#include "Pch.h"
#include "../DeferredRenderer.h"
#include "../../Renderer.h"
#include "../../RenderSceneData.h"
#include "../../SceneTextures.h"
#include "../../RenderSettingContext.h"
#include "BloomCommon.h"

namespace Flower
{
    struct HDREffectPush
    {
        int bEnableVignette = 0;
        float vignetteFalloff = 0.5f;

        int bEnableFringeMode = 0;
        float fringe_barrelStrength;
        float fringe_zoomStrength;
        float fringe_lateralShift;
    };

    class HDREffectPass : public PassInterface
    {
    public:
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;

    public:
        virtual void init() override
        {
            CHECK(pipeline == VK_NULL_HANDLE);
            CHECK(pipelineLayout == VK_NULL_HANDLE);
            CHECK(setLayout == VK_NULL_HANDLE);

            // Config code.
            RHI::get()->descriptorFactoryBegin()
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0) // inHdr
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1) // outHdr
                .buildNoInfoPush(setLayout);

            std::vector<VkDescriptorSetLayout> setLayouts =
            {
                setLayout, // Owner setlayout.
                GetLayoutStatic(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER),  // viewData
                GetLayoutStatic(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER),  // frameData
                RHI::SamplerManager->getCommonDescriptorSetLayout(),
                BlueNoiseMisc::getSetLayout() // Bluenoise
                , StaticTexturesManager::get()->globalBlueNoise.spp_1_buffer.setLayouts // All blue noise set layout is same.
            };
            auto shaderModule = RHI::ShaderManager->getShader("Post_HDREffect.comp.spv", true);

            // Vulkan buid functions.
            VkPipelineLayoutCreateInfo plci = RHIPipelineLayoutCreateInfo();
            VkPushConstantRange pushRange{ .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(HDREffectPush) };
            plci.pushConstantRangeCount = 1;
            plci.pPushConstantRanges = &pushRange;

            plci.setLayoutCount = (uint32_t)setLayouts.size();
            plci.pSetLayouts = setLayouts.data();
            pipelineLayout = RHI::get()->createPipelineLayout(plci);
            VkPipelineShaderStageCreateInfo shaderStageCI{};
            shaderStageCI.module = shaderModule;
            shaderStageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            shaderStageCI.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            shaderStageCI.pName = "main";
            VkComputePipelineCreateInfo computePipelineCreateInfo{};
            computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            computePipelineCreateInfo.layout = pipelineLayout;
            computePipelineCreateInfo.flags = 0;
            computePipelineCreateInfo.stage = shaderStageCI;
            RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &pipeline));
        }

        virtual void release() override
        {
            RHISafeRelease(pipeline);
            RHISafeRelease(pipelineLayout);
            setLayout = VK_NULL_HANDLE;
        }
    };

    void DeferredRenderer::renderHDREffect(
        VkCommandBuffer cmd, 
        Renderer* renderer, 
        SceneTextures* inTextures, 
        RenderSceneData* scene, 
        BufferParamRefPointer& viewData, 
        BufferParamRefPointer& frameData,
        BlueNoiseMisc& inBlueNoise)
    {
        const auto& postProcessVolumeSetting = scene->getPostprocessVolumeSetting();

        if ((postProcessVolumeSetting.bEnableFringeMode == 0)
            && (!postProcessVolumeSetting.bEnableVignette))
        {
            return;
        }

        auto& hdrSceneColor = inTextures->getHdrSceneColorUpscale()->getImage();

        auto resultHDR = m_rtPool->createPoolImage(
            "HDR effect",
            hdrSceneColor.getExtent().width,
            hdrSceneColor.getExtent().height,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

        hdrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        resultHDR->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

        {
            RHI::ScopePerframeMarker marker(cmd, "HDREffect", { 1.0f, 1.0f, 0.0f, 1.0f });

            auto* pass = getPasses()->getPass<HDREffectPass>();

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->pipeline);

            VkDescriptorImageInfo hdrImageInfo = RHIDescriptorImageInfoSample(hdrSceneColor.getView(buildBasicImageSubresource()));
            VkDescriptorImageInfo outImageInfo = RHIDescriptorImageInfoStorage(resultHDR->getImage().getView(buildBasicImageSubresource()));

            auto inRange = VkImageSubresourceRange
            {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            };

            std::vector<VkWriteDescriptorSet> writes
            {
                RHIPushWriteDescriptorSetImage(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outImageInfo),
                RHIPushWriteDescriptorSetImage(1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &hdrImageInfo),
            };

            // Push owner set #0.
            RHI::PushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->pipelineLayout, 0, uint32_t(writes.size()), writes.data());

            std::vector<VkDescriptorSet> passSets =
            {
                viewData->buffer.getSet(),
                frameData->buffer.getSet(),
                RHI::SamplerManager->getCommonDescriptorSet(), // samplers.
                inBlueNoise.getSet()
                , StaticTexturesManager::get()->globalBlueNoise.spp_1_buffer.set // 1spp is good.
            };
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->pipelineLayout,
                1, (uint32_t)passSets.size(), passSets.data(), 0, nullptr);

           


            HDREffectPush compositePush
            {
                .bEnableVignette = postProcessVolumeSetting.bEnableVignette ? 1 : 0,
                .vignetteFalloff = postProcessVolumeSetting.vignette_falloff,


                .bEnableFringeMode = postProcessVolumeSetting.bEnableFringeMode,
                .fringe_barrelStrength = postProcessVolumeSetting.fringe_barrelStrength,
                .fringe_zoomStrength = postProcessVolumeSetting.fringe_zoomStrength,
                .fringe_lateralShift = postProcessVolumeSetting.fringe_lateralShift,
            };

            vkCmdPushConstants(cmd, pass->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(compositePush), &compositePush);


            vkCmdDispatch(cmd, 
                getGroupCount(resultHDR->getImage().getExtent().width, 8), 
                getGroupCount(resultHDR->getImage().getExtent().height, 8), 1);

            resultHDR->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            m_gpuTimer.getTimeStamp(cmd, "HDR Effect");
        }

        // Replace with new guy.
        inTextures->setHdrSceneColorUpscale(resultHDR);
    }
}