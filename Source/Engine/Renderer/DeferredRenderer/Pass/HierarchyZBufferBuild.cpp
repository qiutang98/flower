#include "Pch.h"
#include "../DeferredRenderer.h"
#include "../../Renderer.h"
#include "../../RenderSceneData.h"
#include "../../SceneTextures.h"


namespace Flower
{
    struct HizPush
    {
        uint32_t kInputLevel;
    };

    class HizBuildPass : public PassInterface
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

            RHI::get()->descriptorFactoryBegin()
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0) // hizClosestImage
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1) // hizFurthestImage
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 2) // inDepth
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 3) // inSrcHizClosest
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 4) // inSrcHizFurthest
                .buildNoInfoPush(setLayout);

            std::vector<VkDescriptorSetLayout> setLayouts =
            {
                setLayout, // Owner setlayout.
                RHI::SamplerManager->getCommonDescriptorSetLayout() // Common samplers
            };
            auto shaderModule = RHI::ShaderManager->getShader("SceneHizBuild.comp.spv", true);

            // Vulkan buid functions.
            VkPushConstantRange pushRange{ .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(HizPush) };


            VkPipelineLayoutCreateInfo plci = RHIPipelineLayoutCreateInfo();
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

    PoolImageSharedRef DeferredRenderer::renderHiZ(
        VkCommandBuffer cmd, 
        Renderer* renderer, 
        SceneTextures* inTextures, 
        RenderSceneData* scene, 
        BufferParamRefPointer& viewData, 
        BufferParamRefPointer& frameData)
    {
        auto& depthTex = inTextures->getDepth()->getImage();
        depthTex.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));

        auto* pass = getPasses()->getPass<HizBuildPass>();

        std::vector<VkDescriptorSet> compPassSets =
        {
            RHI::SamplerManager->getCommonDescriptorSet()
        };

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pass->pipelineLayout, 1,
            (uint32_t)compPassSets.size(), compPassSets.data(),
            0, nullptr
        );

        {
            RHI::ScopePerframeMarker marker(cmd, "Hizbuild", { 1.0f, 1.0f, 0.0f, 1.0f });

            uint32_t mipStartWidth = depthTex.getExtent().width;
            uint32_t mipStartHeight = depthTex.getExtent().height;

            auto hizMipChainCloest = m_rtPool->createPoolImage(
                "HizMipchain_closet",
                mipStartWidth,
                mipStartHeight,
                VK_FORMAT_R32_SFLOAT,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                -1);

            auto hizMipChainFurthest = m_rtPool->createPoolImage(
                "HizMipchain_furest",
                mipStartWidth,
                mipStartHeight,
                VK_FORMAT_R32_SFLOAT,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                -1);


            HizPush push{ .kInputLevel = 0 };
            vkCmdPushConstants(cmd, pass->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

            // Build from src.
            {
                VkImageSubresourceRange rangeMip0{ .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1 };


                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->pipeline);

                hizMipChainCloest->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, rangeMip0);
                hizMipChainFurthest->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, rangeMip0);
                
                VkDescriptorImageInfo inDepth = RHIDescriptorImageInfoSample(depthTex.getView(RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT)));
                VkDescriptorImageInfo outHiz0_cloest = RHIDescriptorImageInfoStorage(hizMipChainCloest->getImage().getView(rangeMip0));
                VkDescriptorImageInfo outHiz0_far = RHIDescriptorImageInfoStorage(hizMipChainFurthest->getImage().getView(rangeMip0));
                
                std::vector<VkWriteDescriptorSet> writes
                {
                    RHIPushWriteDescriptorSetImage(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outHiz0_cloest),
                    RHIPushWriteDescriptorSetImage(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outHiz0_far),
                    RHIPushWriteDescriptorSetImage(2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &inDepth),
                    RHIPushWriteDescriptorSetImage(3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &inDepth),
                    RHIPushWriteDescriptorSetImage(4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &inDepth),
                };


                RHI::PushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->pipelineLayout, 0, uint32_t(writes.size()), writes.data());

                vkCmdDispatch(cmd, getGroupCount(mipStartWidth, 8), getGroupCount(mipStartHeight, 8), 1);

                hizMipChainCloest->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, rangeMip0);
                hizMipChainFurthest->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, rangeMip0);
            }
           

            push.kInputLevel = 1;
            vkCmdPushConstants(cmd, pass->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

            // Build from hiz.
            if(hizMipChainCloest->getImage().getInfo().mipLevels > 1)
            {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->pipeline);
                uint32_t loopWidth = mipStartWidth;
                uint32_t loopHeight = mipStartHeight;

                for (uint32_t i = 1; i < hizMipChainCloest->getImage().getInfo().mipLevels; i++)
                {
                    VkImageSubresourceRange rangeMipN_1{ .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = i - 1, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1 };
                    VkImageSubresourceRange rangeMipN{ .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = i, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1 };

                    hizMipChainCloest->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, rangeMipN);
                    hizMipChainFurthest->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, rangeMipN);

                    VkDescriptorImageInfo inHiz_cloest = RHIDescriptorImageInfoSample(hizMipChainCloest->getImage().getView(rangeMipN_1));
                    VkDescriptorImageInfo outHiz_cloest = RHIDescriptorImageInfoStorage(hizMipChainCloest->getImage().getView(rangeMipN));

                    VkDescriptorImageInfo inHiz_far = RHIDescriptorImageInfoSample(hizMipChainFurthest->getImage().getView(rangeMipN_1));
                    VkDescriptorImageInfo outHiz_far = RHIDescriptorImageInfoStorage(hizMipChainFurthest->getImage().getView(rangeMipN));


                    std::vector<VkWriteDescriptorSet> writes
                    {
                        RHIPushWriteDescriptorSetImage(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outHiz_cloest),
                        RHIPushWriteDescriptorSetImage(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outHiz_far),
                        RHIPushWriteDescriptorSetImage(2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &inHiz_cloest),
                        RHIPushWriteDescriptorSetImage(3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &inHiz_cloest),
                        RHIPushWriteDescriptorSetImage(4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &inHiz_far),
                    };


                    RHI::PushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->pipelineLayout, 0, uint32_t(writes.size()), writes.data());

                    loopWidth = getSafeWidthDiv2(loopWidth);
                    loopHeight = getSafeWidthDiv2(loopHeight);

                    vkCmdDispatch(cmd, getGroupCount(loopWidth, 8), getGroupCount(loopHeight, 8), 1);

                    hizMipChainCloest->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, rangeMipN);
                    hizMipChainFurthest->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, rangeMipN);
                }
            }
            m_gpuTimer.getTimeStamp(cmd, "Hzbuild");

            return hizMipChainCloest;
        }
    }
}