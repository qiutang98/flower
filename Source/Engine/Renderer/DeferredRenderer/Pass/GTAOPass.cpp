#include "Pch.h"
#include "../DeferredRenderer.h"
#include "../../Renderer.h"
#include "../../RenderSceneData.h"
#include "../../SceneTextures.h"
#include "../../RenderSettingContext.h"

namespace Flower
{
    struct GTAOPush
    {
        uint32_t sliceNum;
        uint32_t stepNum;
        float radius;
        float thickness;
        float power;
        float intensity;
    };

    class GTAOPass : public PassInterface
    {
    public:
        VkPipeline evaluatePipeline = VK_NULL_HANDLE;
        VkPipeline filterPipeline = VK_NULL_HANDLE;
        VkPipeline tempFilter = VK_NULL_HANDLE;
        

        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;

    public:
        virtual void init() override
        {
            CHECK(setLayout == VK_NULL_HANDLE);
            CHECK(pipelineLayout == VK_NULL_HANDLE);

            // Config code.
            RHI::get()->descriptorFactoryBegin()
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0) // inHiz
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1) // inDepth
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 2) // inGbufferA
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 3) // inGbufferB
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 4) // inGbufferS
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 5) // GTAOImage
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 6) // inGTAO
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 7) // GTAOFilterImageX
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 8) // inGTAOFilterImageX
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 9) // GTAO temp filter.
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 10) // in GTAO temp filter.
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 11) // GTAO history
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 12) // in GTAO history
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 13) // in Velocity
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 14) // inPrevDepth
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 15) // inPrevGBufferB
                .buildNoInfoPush(setLayout);

            std::vector<VkDescriptorSetLayout> setLayouts =
            {
                setLayout, // Owner setlayout.
                GetLayoutStatic(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER),  // viewData
                GetLayoutStatic(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER), // frameData
                RHI::SamplerManager->getCommonDescriptorSetLayout(), // sampler
            };

            VkPushConstantRange pushRange{ .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(GTAOPush) };

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
                CHECK(evaluatePipeline == VK_NULL_HANDLE);
                auto shaderModule = RHI::ShaderManager->getShader("GTAOEvaluate.comp.spv", true);

                shaderStageCI.module = shaderModule;
                computePipelineCreateInfo.stage = shaderStageCI;

                RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &evaluatePipeline));
            }

            {
                CHECK(tempFilter == VK_NULL_HANDLE);
                auto shaderModule = RHI::ShaderManager->getShader("GTAOTemporalFilter.comp.spv", true);

                shaderStageCI.module = shaderModule;
                computePipelineCreateInfo.stage = shaderStageCI;

                RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &tempFilter));
            }

            {
                CHECK(filterPipeline == VK_NULL_HANDLE);
                auto shaderModule = RHI::ShaderManager->getShader("GTAOSpatialFilter.comp.spv", true);

                shaderStageCI.module = shaderModule;
                computePipelineCreateInfo.stage = shaderStageCI;

                RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &filterPipeline));
            }
        }

        virtual void release() override
        {
            RHISafeRelease(pipelineLayout);
            setLayout = VK_NULL_HANDLE;

            RHISafeRelease(evaluatePipeline);
            RHISafeRelease(filterPipeline);
            RHISafeRelease(tempFilter);
        }
    };

    PoolImageSharedRef DeferredRenderer::renderGTAO(
        VkCommandBuffer cmd, 
        Renderer* renderer, 
        SceneTextures* inTextures, 
        RenderSceneData* scene, 
        BufferParamRefPointer& viewData, 
        BufferParamRefPointer& frameData, 
        PoolImageSharedRef inHiz,
        BlueNoiseMisc& inBlueNoise)
    {
        auto& gbufferA = inTextures->getGbufferA()->getImage();
        auto& gbufferB = inTextures->getGbufferB()->getImage();
        auto& gbufferS = inTextures->getGbufferS()->getImage();
        auto& gbufferV = inTextures->getGbufferV()->getImage();
        auto& sceneDepthZ = inTextures->getDepth()->getImage();

        gbufferA.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        gbufferB.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        gbufferS.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        gbufferV.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        sceneDepthZ.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));

        VkDescriptorImageInfo hizInfo = RHIDescriptorImageInfoSample(inHiz->getImage().getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo depthInfo = RHIDescriptorImageInfoSample(sceneDepthZ.getView(RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT)));
        VkDescriptorImageInfo gbufferAInfo = RHIDescriptorImageInfoSample(gbufferA.getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo gbufferBInfo = RHIDescriptorImageInfoSample(gbufferB.getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo gbufferSInfo = RHIDescriptorImageInfoSample(gbufferS.getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo gbufferVInfo = RHIDescriptorImageInfoSample(gbufferV.getView(buildBasicImageSubresource()));

        VkDescriptorImageInfo prevDepthInfo = depthInfo;
        if (m_prevDepth)
        {
            prevDepthInfo = RHIDescriptorImageInfoSample(m_prevDepth->getImage().getView(RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT)));
        }

        VkDescriptorImageInfo preGBufferBInfo = gbufferBInfo;
        if (m_prevGBufferB)
        {
            preGBufferBInfo = RHIDescriptorImageInfoSample(m_prevGBufferB->getImage().getView(buildBasicImageSubresource()));
        }

        auto imageGTAOEvaluate = m_rtPool->createPoolImage(
            "GTAOEvaluate",
            sceneDepthZ.getExtent().width,
            sceneDepthZ.getExtent().height,
            VK_FORMAT_R8_UNORM,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

        auto imageGTAOFilter = m_rtPool->createPoolImage(
            "GTAOFilter",
            imageGTAOEvaluate->getImage().getExtent().width,
            imageGTAOEvaluate->getImage().getExtent().height,
            VK_FORMAT_R8_UNORM,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

        auto imageGTAOTempFilter = m_rtPool->createPoolImage(
            "GTAOTempFilter",
            imageGTAOEvaluate->getImage().getExtent().width,
            imageGTAOEvaluate->getImage().getExtent().height,
            VK_FORMAT_R8_UNORM,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

        bool bShouldCreateNewGTAO = !m_gtaoHistory;
        if (m_gtaoHistory)
        {
            bShouldCreateNewGTAO =
                m_gtaoHistory->getImage().getExtent().width != imageGTAOEvaluate->getImage().getExtent().width ||
                m_gtaoHistory->getImage().getExtent().height != imageGTAOEvaluate->getImage().getExtent().height;

        }
        else
        {
            bShouldCreateNewGTAO = true;
        }

        if (bShouldCreateNewGTAO)
        {
            m_gtaoHistory = m_rtPool->createPoolImage(
                "GTAOHistory",
                imageGTAOEvaluate->getImage().getExtent().width,
                imageGTAOEvaluate->getImage().getExtent().height,
                VK_FORMAT_R8_UNORM,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

            m_gtaoHistory->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        }

        VkDescriptorImageInfo gtaoEvaluateImageInfo = RHIDescriptorImageInfoStorage(imageGTAOEvaluate->getImage().getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo gtaoEvaluateInfo = RHIDescriptorImageInfoSample(imageGTAOEvaluate->getImage().getView(buildBasicImageSubresource()));

        VkDescriptorImageInfo filterImage = RHIDescriptorImageInfoStorage(imageGTAOFilter->getImage().getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo filterImageInfo = RHIDescriptorImageInfoSample(imageGTAOFilter->getImage().getView(buildBasicImageSubresource()));

        VkDescriptorImageInfo tempfilterImage = RHIDescriptorImageInfoStorage(imageGTAOTempFilter->getImage().getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo tempfilterImageInfo = RHIDescriptorImageInfoSample(imageGTAOTempFilter->getImage().getView(buildBasicImageSubresource()));

        VkDescriptorImageInfo historyImage =  RHIDescriptorImageInfoStorage(m_gtaoHistory->getImage().getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo historyImageInfo =  RHIDescriptorImageInfoSample(m_gtaoHistory->getImage().getView(buildBasicImageSubresource()));

        std::vector<VkWriteDescriptorSet> writes
        {
            RHIPushWriteDescriptorSetImage(0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &hizInfo),
            RHIPushWriteDescriptorSetImage(1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &depthInfo),
            RHIPushWriteDescriptorSetImage(2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &gbufferAInfo),
            RHIPushWriteDescriptorSetImage(3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &gbufferBInfo),
            RHIPushWriteDescriptorSetImage(4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &gbufferSInfo),
            RHIPushWriteDescriptorSetImage(5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &gtaoEvaluateImageInfo),
            RHIPushWriteDescriptorSetImage(6, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &gtaoEvaluateInfo),
            RHIPushWriteDescriptorSetImage(7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &filterImage),
            RHIPushWriteDescriptorSetImage(8, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &filterImageInfo),
            RHIPushWriteDescriptorSetImage(9, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &tempfilterImage),
            RHIPushWriteDescriptorSetImage(10, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &tempfilterImageInfo),
            RHIPushWriteDescriptorSetImage(11, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &historyImage),
            RHIPushWriteDescriptorSetImage(12, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &historyImageInfo),
            RHIPushWriteDescriptorSetImage(13, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &gbufferVInfo),
            RHIPushWriteDescriptorSetImage(14, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &prevDepthInfo),
            RHIPushWriteDescriptorSetImage(15, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &preGBufferBInfo),
        };


        auto* pass = getPasses()->getPass<GTAOPass>();

        RHI::PushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->pipelineLayout, 0, uint32_t(writes.size()), writes.data());

        std::vector<VkDescriptorSet> passSets =
        {
            viewData->buffer.getSet(),  // viewData
            frameData->buffer.getSet(), // frameData
            RHI::SamplerManager->getCommonDescriptorSet(), // samplers.
        };

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->pipelineLayout,
            1, (uint32_t)passSets.size(), passSets.data(), 0, nullptr);

        const auto& postProcessVolumeSetting = scene->getPostprocessVolumeSetting();

        GTAOPush pushConst
        {
            .sliceNum = (uint32_t)postProcessVolumeSetting.gtaoSliceNum,
            .stepNum = (uint32_t)postProcessVolumeSetting.gtaoStepNum,
            .radius = postProcessVolumeSetting.gtaoRadius,
            .thickness = postProcessVolumeSetting.gtaoThickness,
            .power = postProcessVolumeSetting.gtaoPower,
            .intensity = postProcessVolumeSetting.gtaoIntensity,
        };
        vkCmdPushConstants(cmd, pass->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConst), &pushConst);


        {
            RHI::ScopePerframeMarker marker(cmd, "Compute GTAO", { 1.0f, 1.0f, 0.0f, 1.0f });

            imageGTAOEvaluate->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->evaluatePipeline);

            vkCmdDispatch(cmd, getGroupCount(imageGTAOEvaluate->getImage().getExtent().width, 8), getGroupCount(imageGTAOEvaluate->getImage().getExtent().height, 8), 1);

            imageGTAOEvaluate->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        }

       

        {
            RHI::ScopePerframeMarker marker(cmd, "Filter GTAO", { 1.0f, 1.0f, 0.0f, 1.0f });
            imageGTAOFilter->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->filterPipeline);

            vkCmdDispatch(cmd, getGroupCount(imageGTAOFilter->getImage().getExtent().width, 16), getGroupCount(imageGTAOFilter->getImage().getExtent().height, 16), 1);

            imageGTAOFilter->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        }

        {
            RHI::ScopePerframeMarker marker(cmd, "GTAO TempFilter", { 1.0f, 1.0f, 0.0f, 1.0f });

            imageGTAOTempFilter->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->tempFilter);

            vkCmdDispatch(cmd, getGroupCount(imageGTAOTempFilter->getImage().getExtent().width, 8), getGroupCount(imageGTAOTempFilter->getImage().getExtent().height, 8), 1);

            imageGTAOTempFilter->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        }

        m_gtaoHistory = imageGTAOTempFilter;

        m_gpuTimer.getTimeStamp(cmd, "GTAO");

        return imageGTAOTempFilter;
    }
}