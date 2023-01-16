#include "Pch.h"
#include "../DeferredRenderer.h"
#include "../../Renderer.h"
#include "../../RenderSceneData.h"
#include "../../SceneTextures.h"
#include "../../RenderSettingContext.h"

namespace Flower
{
    struct DofDepthRange
    {
        uint32_t minDepth;
        uint32_t maxDepth;
        uint32_t sumPmxDepth;
        uint32_t pmxPixelCount;
    };

    struct DofPush
    {
        int bNearBlur;
        int bFocusPMXCharacter; // 0 off, 1 use min depth,  2 use max depth, 3 use avg depth.

        float distanceF; // focus distance
        float lensCoeff;
        float maxCoc;
        float maxCoCRcp;
        float aspectRcp;

        float focusLen;
        float filmHeight;
        float fStop;
        float pmxFoucusMinOffset;
    };

    // Dof effect, baisc idea from Doom External.
    class DofPass : public PassInterface
    {
    public:
        VkPipeline downsamplePipeline = VK_NULL_HANDLE;
        VkPipeline gatherPipeline = VK_NULL_HANDLE;
        VkPipeline fillPipeline = VK_NULL_HANDLE;
        VkPipeline combinePipeline = VK_NULL_HANDLE;

        VkPipeline pmxFocusPipeline = VK_NULL_HANDLE;

        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;

    public:
        virtual void init() override
        {
            CHECK(setLayout == VK_NULL_HANDLE);
            CHECK(pipelineLayout == VK_NULL_HANDLE);

            // Config code.
            RHI::get()->descriptorFactoryBegin()
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0) // inDepth
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1) // inHDRSceneColor
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 2) // downSampleHDRImage
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 3) // inDownSampleHDRImage
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 4) // HDRSceneColorImage
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 5) // gatherImage
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 6) // inGatherImage
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 7) // expandFillImage
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 8) // inExpandFillImage
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 9) // inGbufferA
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 10) // Depth Range
                .buildNoInfoPush(setLayout);

            std::vector<VkDescriptorSetLayout> setLayouts =
            {
                setLayout, // Owner setlayout.
                GetLayoutStatic(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER),  // viewData
                GetLayoutStatic(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER), // frameData
                RHI::SamplerManager->getCommonDescriptorSetLayout(), // sampler
                BlueNoiseMisc::getSetLayout() // Bluenoise
                , StaticTexturesManager::get()->globalBlueNoise.spp_1_buffer.setLayouts // All blue noise set layout is same.
            };

            VkPushConstantRange pushRange{ .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(DofPush) };

            VkPipelineLayoutCreateInfo plci = RHIPipelineLayoutCreateInfo();
            plci.pushConstantRangeCount = 1;
            plci.pPushConstantRanges = &pushRange;
            plci.setLayoutCount = (uint32_t)setLayouts.size();
            plci.pSetLayouts = setLayouts.data();
            pipelineLayout = RHI::get()->createPipelineLayout(plci);


            VkPipelineShaderStageCreateInfo shaderStageCI{};
            shaderStageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            shaderStageCI.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            shaderStageCI.pName = "main";
            VkComputePipelineCreateInfo computePipelineCreateInfo{};
            computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            computePipelineCreateInfo.layout = pipelineLayout;
            computePipelineCreateInfo.flags = 0;

            {
                CHECK(pmxFocusPipeline == VK_NULL_HANDLE);
                auto shaderModule = RHI::ShaderManager->getShader("Dof_FocusEvaluate.comp.spv", true);

                shaderStageCI.module = shaderModule;
                computePipelineCreateInfo.stage = shaderStageCI;

                RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &pmxFocusPipeline));
            }
            {
                CHECK(downsamplePipeline == VK_NULL_HANDLE);
                auto shaderModule = RHI::ShaderManager->getShader("Dof_Prepare.comp.spv", true);

                shaderStageCI.module = shaderModule;
                computePipelineCreateInfo.stage = shaderStageCI;

                RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &downsamplePipeline));
            }
            {
                CHECK(gatherPipeline == VK_NULL_HANDLE);
                auto shaderModule = RHI::ShaderManager->getShader("Dof_Gather.comp.spv", true);

                shaderStageCI.module = shaderModule;
                computePipelineCreateInfo.stage = shaderStageCI;

                RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &gatherPipeline));
            }
            {
                CHECK(fillPipeline == VK_NULL_HANDLE);
                auto shaderModule = RHI::ShaderManager->getShader("Dof_ExpandFill.comp.spv", true);

                shaderStageCI.module = shaderModule;
                computePipelineCreateInfo.stage = shaderStageCI;

                RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &fillPipeline));
            }
            {
                CHECK(combinePipeline == VK_NULL_HANDLE);
                auto shaderModule = RHI::ShaderManager->getShader("Dof_Combine.comp.spv", true);

                shaderStageCI.module = shaderModule;
                computePipelineCreateInfo.stage = shaderStageCI;

                RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &combinePipeline));
            }
        }

        virtual void release() override
        {
            RHISafeRelease(pipelineLayout);
            setLayout = VK_NULL_HANDLE;

            RHISafeRelease(downsamplePipeline);
            RHISafeRelease(gatherPipeline);
            RHISafeRelease(fillPipeline);
            RHISafeRelease(combinePipeline);

            RHISafeRelease(pmxFocusPipeline);
        }
    };


    void Flower::DeferredRenderer::renderDof(
        VkCommandBuffer cmd, 
        Renderer* renderer, 
        SceneTextures* inTextures, 
        RenderSceneData* scene, 
        BufferParamRefPointer& viewData, 
        BufferParamRefPointer& frameData,
        BlueNoiseMisc& inBlueNoise)
    {
        // Dof config.
        const auto& postProcessVolumeSetting = scene->getPostprocessVolumeSetting();
        if (!postProcessVolumeSetting.bDofEnable)
        {
            return;
        }


        auto& sceneDepthZ = inTextures->getDepth()->getImage();
        auto& hdrSceneColor = inTextures->getHdrSceneColorUpscale()->getImage();
        auto& gbufferA = inTextures->getGbufferA()->getImage();

        int32_t pmxTrackMode = 0;

        float focusDistance = 0.0f;
        if(postProcessVolumeSetting.dof_focusMode == 0)
        {
            glm::vec4 pView = m_cacheViewData.camView * glm::vec4(postProcessVolumeSetting.dof_focusPoint, 1.0f);

            focusDistance = -pView.z;
        }
        else if (postProcessVolumeSetting.dof_focusMode == 1)
        {
            focusDistance = postProcessVolumeSetting.dof_focusDistance;
        }
        else if (postProcessVolumeSetting.dof_focusMode == 2)
        {
            // Auto track pmx
            pmxTrackMode = postProcessVolumeSetting.dof_trackPMXMode + 1;
        }
        else
        {
            CHECK_ENTRY();
        }

        float focuseLen = postProcessVolumeSetting.dof_focusLength * 0.001f; // MM to meter
        if (postProcessVolumeSetting.dof_bUseCameraFOV)
        {
            float fovy = m_cacheViewData.camInfo.x;
            focuseLen =  0.5f * postProcessVolumeSetting.dof_FilmHeight / glm::tan(0.5f * fovy);
        }

        // Focus distance should max than focus len.
        focusDistance = glm::max(focusDistance, focuseLen);

        float lensCoeff = focuseLen * focuseLen / (postProcessVolumeSetting.dof_aperture * (focusDistance - focuseLen) * postProcessVolumeSetting.dof_FilmHeight * 2);

        float maxCoC = glm::min(0.05f, (postProcessVolumeSetting.dof_kernelSize * 4.0f + 6.0f) / float((hdrSceneColor.getExtent().height)));


        DofPush pushConst
        {
            .bNearBlur = postProcessVolumeSetting.dof_bNearBlur,
            .bFocusPMXCharacter = pmxTrackMode,
            .distanceF = focusDistance,
            .lensCoeff = lensCoeff,
            .maxCoc = maxCoC,
            .maxCoCRcp = 1.0f / maxCoC,
            .aspectRcp = float(hdrSceneColor.getExtent().height) / float(hdrSceneColor.getExtent().width),
            .focusLen = focuseLen,
            .filmHeight = postProcessVolumeSetting.dof_FilmHeight,
            .fStop = postProcessVolumeSetting.dof_aperture,
            .pmxFoucusMinOffset = postProcessVolumeSetting.dof_pmxFoucusMinOffset,
        };

        sceneDepthZ.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));
        hdrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        gbufferA.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

        VkDescriptorImageInfo hdrImageInfo = RHIDescriptorImageInfoStorage(hdrSceneColor.getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo hdrSampleInfo = RHIDescriptorImageInfoSample(hdrSceneColor.getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo depthInfo = RHIDescriptorImageInfoSample(sceneDepthZ.getView(RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT)));

        VkDescriptorImageInfo gbufferAInfo = RHIDescriptorImageInfoSample(gbufferA.getView(buildBasicImageSubresource()));

        auto dowmsampleImage = m_rtPool->createPoolImage(
            "Dof downsample",
            hdrSceneColor.getExtent().width / 2,
            hdrSceneColor.getExtent().height / 2,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

        auto gatherImage = m_rtPool->createPoolImage(
            "Dof gather",
            dowmsampleImage->getImage().getExtent().width,
            dowmsampleImage->getImage().getExtent().height,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

        auto fillImage = m_rtPool->createPoolImage(
            "Dof Fill",
            gatherImage->getImage().getExtent().width,
            gatherImage->getImage().getExtent().height,
            gatherImage->getImage().getFormat(),
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

        auto rangeBuffer = getBuffers()->getStaticStorageGPUOnly("DofRangeBuffer", sizeof(DofDepthRange));
        VkDescriptorBufferInfo rangeBufferInfo = VkDescriptorBufferInfo{ .buffer = rangeBuffer->buffer.getBuffer()->getVkBuffer(), .offset = 0, .range = rangeBuffer->buffer.getBuffer()->getSize() };

        VkDescriptorImageInfo downImageInfo = RHIDescriptorImageInfoStorage(dowmsampleImage->getImage().getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo downSampleInfo = RHIDescriptorImageInfoSample(dowmsampleImage->getImage().getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo gatherImageInfo = RHIDescriptorImageInfoStorage(gatherImage->getImage().getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo gatherSampleInfo = RHIDescriptorImageInfoSample(gatherImage->getImage().getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo fillImageInfo = RHIDescriptorImageInfoStorage(fillImage->getImage().getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo fillSampleInfo = RHIDescriptorImageInfoSample(fillImage->getImage().getView(buildBasicImageSubresource()));

        std::vector<VkWriteDescriptorSet> writes
        {
            RHIPushWriteDescriptorSetImage(0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &depthInfo),
            RHIPushWriteDescriptorSetImage(1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &hdrSampleInfo),
            RHIPushWriteDescriptorSetImage(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &downImageInfo),
            RHIPushWriteDescriptorSetImage(3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &downSampleInfo),
            RHIPushWriteDescriptorSetImage(4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &hdrImageInfo),
            RHIPushWriteDescriptorSetImage(5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &gatherImageInfo),
            RHIPushWriteDescriptorSetImage(6, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &gatherSampleInfo),
            RHIPushWriteDescriptorSetImage(7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &fillImageInfo),
            RHIPushWriteDescriptorSetImage(8, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &fillSampleInfo),
            RHIPushWriteDescriptorSetImage(9, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &gbufferAInfo),
            RHIPushWriteDescriptorSetBuffer(10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &rangeBufferInfo),
        };

        auto* pass = getPasses()->getPass<DofPass>();
        RHI::PushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->pipelineLayout, 0, uint32_t(writes.size()), writes.data());

        std::vector<VkDescriptorSet> passSets =
        {
            viewData->buffer.getSet(),  // viewData
            frameData->buffer.getSet(), // frameData
            RHI::SamplerManager->getCommonDescriptorSet(), // samplers.
            inBlueNoise.getSet()
            , StaticTexturesManager::get()->globalBlueNoise.spp_1_buffer.set // 1spp is good.
        };

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->pipelineLayout,
            1, (uint32_t)passSets.size(), passSets.data(), 0, nullptr);


        vkCmdPushConstants(cmd, pass->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConst), &pushConst);

        {
            RHI::ScopePerframeMarker marker(cmd, "Dof Auto focus", { 1.0f, 1.0f, 0.0f, 1.0f });

            DofDepthRange clearRangeValue = { .minDepth = ~0u, .maxDepth = 0u, .sumPmxDepth = 0, .pmxPixelCount = 0 };
            vkCmdUpdateBuffer(cmd, *rangeBuffer->buffer.getBuffer(), 0, rangeBuffer->buffer.getBuffer()->getSize(), &clearRangeValue);
            std::array<VkBufferMemoryBarrier2, 1> fillBarriers
            {
                RHIBufferBarrier(rangeBuffer->buffer.getBuffer()->getVkBuffer(),
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT),
            };
            RHIPipelineBarrier(cmd, 0, (uint32_t)fillBarriers.size(), fillBarriers.data(), 0, nullptr);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->pmxFocusPipeline);

            // Block dim is 3x3.
            vkCmdDispatch(cmd,
                getGroupCount(sceneDepthZ.getExtent().width / 3 + 1, 8),
                getGroupCount(sceneDepthZ.getExtent().height / 3 + 1, 8), 1);

            VkBufferMemoryBarrier2 endBufferBarrier = RHIBufferBarrier(rangeBuffer->buffer.getBuffer()->getVkBuffer(),
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
            RHIPipelineBarrier(cmd, 0, 1, &endBufferBarrier, 0, nullptr);

        }
        {
            RHI::ScopePerframeMarker marker(cmd, "Dof Downsample", { 1.0f, 1.0f, 0.0f, 1.0f });

            dowmsampleImage->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->downsamplePipeline);

            vkCmdDispatch(cmd, getGroupCount(
                dowmsampleImage->getImage().getExtent().width, 8),
                getGroupCount(dowmsampleImage->getImage().getExtent().height, 8), 1);

            dowmsampleImage->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        }
        {
            RHI::ScopePerframeMarker marker(cmd, "Dof Gather", { 1.0f, 1.0f, 0.0f, 1.0f });

            gatherImage->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->gatherPipeline);

            vkCmdDispatch(cmd, getGroupCount(
                gatherImage->getImage().getExtent().width, 8), 
                getGroupCount(gatherImage->getImage().getExtent().height, 8), 1);

            gatherImage->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        }
        {
            RHI::ScopePerframeMarker marker(cmd, "Dof Fill", { 1.0f, 1.0f, 0.0f, 1.0f });

            fillImage->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->fillPipeline);

            vkCmdDispatch(cmd, 
                getGroupCount(fillImage->getImage().getExtent().width, 8), 
                getGroupCount(fillImage->getImage().getExtent().height, 8), 1);

            fillImage->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        }
        {
            RHI::ScopePerframeMarker marker(cmd, "Dof Combine", { 1.0f, 1.0f, 0.0f, 1.0f });

            hdrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->combinePipeline);

            vkCmdDispatch(cmd, 
                getGroupCount(hdrSceneColor.getExtent().width, 8), 
                getGroupCount(hdrSceneColor.getExtent().height, 8), 1);

            hdrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        }

        m_gpuTimer.getTimeStamp(cmd, "Dof");
    }
}