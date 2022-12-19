#include "Pch.h"
#include "../DeferredRenderer.h"
#include "../../Renderer.h"
#include "../../RenderSceneData.h"
#include "../../SceneTextures.h"
#include "../../RenderSettingContext.h"
#include "BloomCommon.h"

#include <glm/gtc/integer.hpp>

namespace Flower
{
    constexpr uint32_t GMaxDownsampleCount = 6;

    struct BloomDownsample
    {
        glm::vec4 prefilterFactor;
        uint32_t mipLevel;
        
    };

    struct BloomPushUpscale
    {
        uint32_t bBlurX;
        uint32_t bFinalBlur = 0u;
        uint32_t upscaleTime;
        float blurRadius;

    };

	class BloomPass : public PassInterface
	{
    public:
        VkPipeline downsamplePipeline = VK_NULL_HANDLE;
        VkPipelineLayout downsamplePipelineLayout = VK_NULL_HANDLE;
        VkDescriptorSetLayout downsampleSetLayout = VK_NULL_HANDLE;

        VkPipeline upscalePipeline = VK_NULL_HANDLE;
        VkPipelineLayout upscalePipelineLayout = VK_NULL_HANDLE;
        VkDescriptorSetLayout upscaleSetLayout = VK_NULL_HANDLE;

    public:
        virtual void init() override
        {
            // Downsample pipe init.
            {
                CHECK(downsamplePipeline == VK_NULL_HANDLE);
                CHECK(downsamplePipelineLayout == VK_NULL_HANDLE);
                CHECK(downsampleSetLayout == VK_NULL_HANDLE);

                // Config code.
                RHI::get()->descriptorFactoryBegin()
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0) // in
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1) // out
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 2) // lum
                    .buildNoInfoPush(downsampleSetLayout);

                std::vector<VkDescriptorSetLayout> setLayouts =
                {
                    downsampleSetLayout, // Owner setlayout.
                    RHI::SamplerManager->getCommonDescriptorSetLayout(),
                    GetLayoutStatic(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER),  // viewData
                    GetLayoutStatic(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER),  // frameData
                };
                auto shaderModule = RHI::ShaderManager->getShader("BasicBloomDownsample.comp.spv", true);

                // Vulkan buid functions.
                VkPipelineLayoutCreateInfo plci = RHIPipelineLayoutCreateInfo();

                VkPushConstantRange pushRange{ .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(BloomDownsample) };
                plci.pushConstantRangeCount = 1;
                plci.pPushConstantRanges = &pushRange;

                plci.setLayoutCount = (uint32_t)setLayouts.size();
                plci.pSetLayouts = setLayouts.data();
                downsamplePipelineLayout = RHI::get()->createPipelineLayout(plci);
                VkPipelineShaderStageCreateInfo shaderStageCI{};
                shaderStageCI.module = shaderModule;
                shaderStageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                shaderStageCI.stage = VK_SHADER_STAGE_COMPUTE_BIT;
                shaderStageCI.pName = "main";
                VkComputePipelineCreateInfo computePipelineCreateInfo{};
                computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
                computePipelineCreateInfo.layout = downsamplePipelineLayout;
                computePipelineCreateInfo.flags = 0;
                computePipelineCreateInfo.stage = shaderStageCI;
                RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &downsamplePipeline));
            }

            {
                CHECK(upscalePipeline == VK_NULL_HANDLE);
                CHECK(upscalePipelineLayout == VK_NULL_HANDLE);
                CHECK(upscaleSetLayout == VK_NULL_HANDLE);

                // Config code.
                RHI::get()->descriptorFactoryBegin()
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0) // inHdr
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1) // inCurHdr
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 2) // out
                    .buildNoInfoPush(upscaleSetLayout);

                std::vector<VkDescriptorSetLayout> setLayouts =
                {
                    upscaleSetLayout, // Owner setlayout.
                    RHI::SamplerManager->getCommonDescriptorSetLayout(),
                };
                auto shaderModule = RHI::ShaderManager->getShader("BasicBloomUpscale.comp.spv", true);

                // Vulkan buid functions.
                VkPushConstantRange pushRange{ .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(BloomPushUpscale)};
                
                
                VkPipelineLayoutCreateInfo plci = RHIPipelineLayoutCreateInfo();
                plci.pushConstantRangeCount = 1;
                plci.pPushConstantRanges = &pushRange;

                plci.setLayoutCount = (uint32_t)setLayouts.size();
                plci.pSetLayouts = setLayouts.data();
                upscalePipelineLayout = RHI::get()->createPipelineLayout(plci);
                VkPipelineShaderStageCreateInfo shaderStageCI{};
                shaderStageCI.module = shaderModule;
                shaderStageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                shaderStageCI.stage = VK_SHADER_STAGE_COMPUTE_BIT;
                shaderStageCI.pName = "main";
                VkComputePipelineCreateInfo computePipelineCreateInfo{};
                computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
                computePipelineCreateInfo.layout = upscalePipelineLayout;
                computePipelineCreateInfo.flags = 0;
                computePipelineCreateInfo.stage = shaderStageCI;
                RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &upscalePipeline));
            }
        }

        virtual void release() override
        {
            RHISafeRelease(downsamplePipeline);
            RHISafeRelease(downsamplePipelineLayout);
            downsampleSetLayout = VK_NULL_HANDLE;

            RHISafeRelease(upscalePipeline);
            RHISafeRelease(upscalePipelineLayout);
            upscaleSetLayout = VK_NULL_HANDLE;
        }
	};

    PoolImageSharedRef DeferredRenderer::renderBloom(
        VkCommandBuffer cmd, 
        Renderer* renderer, 
        SceneTextures* inTextures, 
        RenderSceneData* scene, 
        BufferParamRefPointer& viewData, 
        BufferParamRefPointer& frameData)
	{
        auto& hdrSceneColor = inTextures->getHdrSceneColorUpscale()->getImage();
        hdrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

        const uint32_t srcHdrColorWidth = hdrSceneColor.getExtent().width;
        const uint32_t srcHdrColorHeight = hdrSceneColor.getExtent().height;

        // Min size is 64x64
        const uint32_t mipStartWidth  = srcHdrColorWidth  >> 1;
        const uint32_t mipStartHeight = srcHdrColorHeight >> 1;

        const uint32_t downsampleMipCount = glm::min(GMaxDownsampleCount, glm::log2(glm::min(mipStartWidth, mipStartHeight)));

        auto* pass = getPasses()->getPass<BloomPass>();

        std::vector<PoolImageSharedRef> downsampleBlurs;
        downsampleBlurs.resize(downsampleMipCount);

        for (uint32_t i = 0; i < downsampleMipCount; i ++)
        {
            downsampleBlurs[i] = m_rtPool->createPoolImage(
                "SceneColorBlurChain",
                mipStartWidth  >> i,
                mipStartHeight >> i,
                hdrSceneColor.getFormat(),
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
        }

        PoolImageSharedRef result;
        {
            RHI::ScopePerframeMarker marker(cmd, "Bloom Basic", { 1.0f, 1.0f, 0.0f, 1.0f });

            std::vector<VkDescriptorSet> passSets =
            {
                RHI::SamplerManager->getCommonDescriptorSet(), // samplers.
                viewData->buffer.getSet(),
                frameData->buffer.getSet(),
            };

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->downsamplePipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->downsamplePipelineLayout,
                1, (uint32_t)passSets.size(), passSets.data(), 0, nullptr);


            CHECK(m_averageLum && "Must call adaptive exposure before tonemapper.");
            VkDescriptorImageInfo lumImgInfo = RHIDescriptorImageInfoSample(m_averageLum->getImage().getView(buildBasicImageSubresource()));

            BloomDownsample downsamplePush{};

            const auto& postProcessVolumeSetting = scene->getPostprocessVolumeSetting();

            downsamplePush.prefilterFactor = getBloomPrefilter(postProcessVolumeSetting.bloomThreshold, postProcessVolumeSetting.bloomThresholdSoft);

            VkDescriptorImageInfo inImageInfo{};
            VkDescriptorImageInfo outImageInfo{};
            for (uint32_t i = 0; i < downsampleMipCount; i++)
            {
                const bool bFirstLevel = (i == 0);
                downsamplePush.mipLevel = i;

                inImageInfo = RHIDescriptorImageInfoSample((bFirstLevel ? hdrSceneColor : downsampleBlurs[i - 1]->getImage()).getView(buildBasicImageSubresource()));

                downsampleBlurs[i]->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

                outImageInfo = RHIDescriptorImageInfoStorage(downsampleBlurs[i]->getImage().getView(buildBasicImageSubresource()));

                std::vector<VkWriteDescriptorSet> writes
                {
                    RHIPushWriteDescriptorSetImage(0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &inImageInfo),
                    RHIPushWriteDescriptorSetImage(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outImageInfo),
                    RHIPushWriteDescriptorSetImage(2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &lumImgInfo),
                };

                RHI::PushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->downsamplePipelineLayout, 0, uint32_t(writes.size()), writes.data());

                vkCmdPushConstants(cmd, pass->downsamplePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(downsamplePush), &downsamplePush);

                vkCmdDispatch(cmd, getGroupCount(downsampleBlurs[i]->getImage().getExtent().width, 8), getGroupCount(downsampleBlurs[i]->getImage().getExtent().height, 8), 1);

                downsampleBlurs[i]->getImage().transitionLayout(cmd,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,buildBasicImageSubresource());
            }

            std::vector<VkDescriptorSet> upscalePassSets =
            {
                RHI::SamplerManager->getCommonDescriptorSet(), // samplers.
            };
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->upscalePipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->upscalePipelineLayout, 1, (uint32_t)upscalePassSets.size(), upscalePassSets.data(), 0, nullptr);

            // Upscale.
            VkDescriptorImageInfo inImageCurInfo{};
            PoolImageSharedRef prevLevelUpscaleResult = nullptr;
            for (uint32_t i = 0; i < downsampleMipCount; i++)
            {
                uint32_t workMip = downsampleMipCount - i;

                uint32_t workWidth  = srcHdrColorWidth  >> workMip;
                uint32_t workHeight = srcHdrColorHeight >> workMip;

                const bool bLowestUpscale  = (i == 0);
                const bool bHighestUpscale = (i == (downsampleMipCount - 1));

                // Prev blur result.
                inImageInfo = RHIDescriptorImageInfoSample((
                    bLowestUpscale ? 
                    downsampleBlurs[downsampleMipCount - 1] : // Input from last downsample texture.
                    prevLevelUpscaleResult // Input from prev upscale result.
                )->getImage().getView(buildBasicImageSubresource()));

                inImageCurInfo = RHIDescriptorImageInfoSample((
                    bHighestUpscale ?
                    hdrSceneColor :
                    downsampleBlurs[workMip - 1]->getImage()
                    ).getView(buildBasicImageSubresource()));
                
                auto blurX = m_rtPool->createPoolImage("blurX", workWidth, workHeight, hdrSceneColor.getFormat(), VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);

                outImageInfo = RHIDescriptorImageInfoStorage(blurX->getImage().getView(buildBasicImageSubresource()));

                std::vector<VkWriteDescriptorSet> writes
                {
                    RHIPushWriteDescriptorSetImage(0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &inImageInfo),
                    RHIPushWriteDescriptorSetImage(1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &inImageCurInfo),
                    RHIPushWriteDescriptorSetImage(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outImageInfo),
                };

                const auto& postProcessVolumeSetting = scene->getPostprocessVolumeSetting();

                BloomPushUpscale upscalePush{ .bBlurX = 1u, .blurRadius = postProcessVolumeSetting.bloomRadius, };
                vkCmdPushConstants(cmd, pass->upscalePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(upscalePush), &upscalePush);
                RHI::PushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->upscalePipelineLayout, 0, uint32_t(writes.size()), writes.data());

                blurX->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                vkCmdDispatch(cmd, getGroupCount(workWidth, 8), getGroupCount(workHeight, 8), 1);
                blurX->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

                inImageInfo = RHIDescriptorImageInfoSample(blurX->getImage().getView(buildBasicImageSubresource()));

                upscalePush = { .bBlurX = 0u, .bFinalBlur = (bHighestUpscale ? 1u : 0u), .upscaleTime = workMip - 1,.blurRadius = postProcessVolumeSetting.bloomRadius,};
                vkCmdPushConstants(cmd, pass->upscalePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(upscalePush), &upscalePush);
                auto blurY = m_rtPool->createPoolImage("blurY", workWidth, workHeight, hdrSceneColor.getFormat(), VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);

                outImageInfo = RHIDescriptorImageInfoStorage(blurY->getImage().getView(buildBasicImageSubresource()));

                writes = 
                {
                    RHIPushWriteDescriptorSetImage(0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &inImageInfo),
                    RHIPushWriteDescriptorSetImage(1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &inImageCurInfo),
                    RHIPushWriteDescriptorSetImage(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outImageInfo),
                };

                RHI::PushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->upscalePipelineLayout, 0, uint32_t(writes.size()), writes.data());

                blurY->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                vkCmdDispatch(cmd, getGroupCount(workWidth, 8), getGroupCount(workHeight, 8), 1);
                blurY->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

                // Update prevUpscale result.
                prevLevelUpscaleResult = blurY;
            }
            result = prevLevelUpscaleResult;
        }

        m_gpuTimer.getTimeStamp(cmd, "BasicBloom");

        return result;
	}
}