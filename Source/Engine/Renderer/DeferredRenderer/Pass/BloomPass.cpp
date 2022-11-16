#include "Pch.h"
#include "../DeferredRenderer.h"
#include "../../Renderer.h"
#include "../../RenderSceneData.h"
#include "../../SceneTextures.h"
#include "../../RenderSettingContext.h"

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

    

    PoolImageSharedRef DeferredRenderer::renderBloom(VkCommandBuffer cmd, Renderer* renderer, SceneTextures* inTextures, RenderSceneData* scene, BufferParamRefPointer& viewData, BufferParamRefPointer& frameData)
	{
        auto& hdrSceneColor = inTextures->getHdrSceneColorUpscale()->getImage();
        hdrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

        const uint32_t srcHdrColorWidth = hdrSceneColor.getExtent().width;
        const uint32_t srcHdrColorHeight = hdrSceneColor.getExtent().height;
        const uint32_t mipStartWidth = getSafeWidthDiv2(srcHdrColorWidth);
        const uint32_t mipStartHeight = getSafeWidthDiv2(srcHdrColorHeight);

        const uint32_t downsampleMipCount = glm::min(GMaxDownsampleCount, glm::log2(glm::min(mipStartWidth, mipStartHeight)));

		auto sceneColoBlurMipChain = m_rtPool->createPoolImage(
            "SceneColorBlurChain", 
            mipStartWidth, 
            mipStartHeight,
            hdrSceneColor.getFormat(),
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
            downsampleMipCount);

        auto* pass = getPasses()->getPass<BloomPass>();

        

        // Upscale.
        auto sceneColoUpscaleMipChain = m_rtPool->createPoolImage(
            "SceneColorUpscaleChain",
            mipStartWidth,
            mipStartHeight,
            hdrSceneColor.getFormat(),
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
            downsampleMipCount - 1);

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

            VkDescriptorImageInfo inImageInfo{};
            VkDescriptorImageInfo outImageInfo{};

            // CHECK(m_averageLum && "Must call adaptive exposure before tonemapper."); // TODO: 
            VkDescriptorImageInfo lumImgInfo = RHIDescriptorImageInfoSample(m_averageLum->getImage().getView(buildBasicImageSubresource()));


            uint32_t workWidth = mipStartWidth;
            uint32_t workHeight = mipStartHeight;
            BloomDownsample downsamplePush{};

            float knee = RenderSettingManager::get()->bloomThreshold * RenderSettingManager::get()->bloomThresholdSoft;

            downsamplePush.prefilterFactor.x = RenderSettingManager::get()->bloomThreshold;
            downsamplePush.prefilterFactor.y = downsamplePush.prefilterFactor.x - knee;
            downsamplePush.prefilterFactor.z = 2.0f * knee;
            downsamplePush.prefilterFactor.w = 0.25f / (knee + 0.00001f);

            for (uint32_t i = 0; i < downsampleMipCount; i++)
            {
                const bool bFirstLevel = (i == 0);
                downsamplePush.mipLevel = i;

                if (bFirstLevel)
                {
                    inImageInfo = RHIDescriptorImageInfoSample(hdrSceneColor.getView(buildBasicImageSubresource()));
                }
                else
                {
                    auto prevRange = VkImageSubresourceRange
                    {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, 
                        .baseMipLevel = i - 1, 
                        .levelCount = 1, 
                        .baseArrayLayer = 0, 
                        .layerCount = 1 
                    };
                    inImageInfo = RHIDescriptorImageInfoSample(sceneColoBlurMipChain->getImage().getView(prevRange));
                }

                auto outRange = VkImageSubresourceRange
                {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = i,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1
                };

                sceneColoBlurMipChain->getImage().transitionLayout(
                    cmd,
                    VK_IMAGE_LAYOUT_GENERAL,
                    outRange
                );
                outImageInfo = RHIDescriptorImageInfoStorage(sceneColoBlurMipChain->getImage().getView(outRange));

                std::vector<VkWriteDescriptorSet> writes
                {
                    RHIPushWriteDescriptorSetImage(0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &inImageInfo),
                    RHIPushWriteDescriptorSetImage(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outImageInfo),
                    RHIPushWriteDescriptorSetImage(2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &lumImgInfo),
                };

                RHI::PushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->downsamplePipelineLayout, 0, uint32_t(writes.size()), writes.data());

                vkCmdPushConstants(cmd, pass->downsamplePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(downsamplePush), &downsamplePush);

                vkCmdDispatch(cmd, getGroupCount(workWidth, 8), getGroupCount(workHeight, 8), 1);

                sceneColoBlurMipChain->getImage().transitionLayout(
                    cmd,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    outRange
                );

                workWidth = getSafeWidthDiv2(workWidth);
                workHeight = getSafeWidthDiv2(workHeight);
            }

            std::vector<VkDescriptorSet> upscalePassSets =
            {
                RHI::SamplerManager->getCommonDescriptorSet(), // samplers.
            };
            
            BloomPushUpscale upscalePush{ .blurRadius = RenderSettingManager::get()->bloomRadius, }; 
            vkCmdPushConstants(cmd, pass->upscalePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(upscalePush), &upscalePush);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->upscalePipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->upscalePipelineLayout, 1, (uint32_t)upscalePassSets.size(), upscalePassSets.data(), 0, nullptr);
            
            VkDescriptorImageInfo inImageCurInfo{};
            for (uint32_t i = 1; i < downsampleMipCount; i++)
            {
                uint32_t workMip = downsampleMipCount - i - 1;

                workWidth = glm::max(1u, mipStartWidth >> workMip);
                workHeight = glm::max(1u, mipStartHeight >> workMip);

                if (i == 1)
                {
                    auto inRange = VkImageSubresourceRange
                    {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel = workMip + 1,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1
                    };

                    // Input from low res mip.
                    inImageInfo = RHIDescriptorImageInfoSample(sceneColoBlurMipChain->getImage().getView(inRange));
                }
                else
                {
                    auto inRange = VkImageSubresourceRange
                    {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel = workMip + 1,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1
                    };

                    // Input from low res mip. upscale mip.
                    inImageInfo = RHIDescriptorImageInfoSample(sceneColoUpscaleMipChain->getImage().getView(inRange));
                }
                
                // Cur input.
                auto inRangeCur = VkImageSubresourceRange
                {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = workMip,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1
                };
                inImageCurInfo = RHIDescriptorImageInfoSample(sceneColoBlurMipChain->getImage().getView(inRangeCur));

                auto outRange = VkImageSubresourceRange
                {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = workMip,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1
                };
                sceneColoUpscaleMipChain->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, outRange);
                outImageInfo = RHIDescriptorImageInfoStorage(sceneColoUpscaleMipChain->getImage().getView(outRange));

                std::vector<VkWriteDescriptorSet> writes
                {
                    RHIPushWriteDescriptorSetImage(0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &inImageInfo),
                    RHIPushWriteDescriptorSetImage(1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &inImageCurInfo),
                    RHIPushWriteDescriptorSetImage(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outImageInfo),
                };

                RHI::PushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->upscalePipelineLayout, 0, uint32_t(writes.size()), writes.data());

                vkCmdDispatch(cmd, getGroupCount(workWidth, 8), getGroupCount(workHeight, 8), 1);

                sceneColoUpscaleMipChain->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, outRange);
            }

        }

        m_gpuTimer.getTimeStamp(cmd, "BasicBloom");

        return sceneColoUpscaleMipChain;
	}
}