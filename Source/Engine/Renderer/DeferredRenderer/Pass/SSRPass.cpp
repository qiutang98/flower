#include "Pch.h"
#include "../DeferredRenderer.h"
#include "../../Renderer.h"
#include "../../RenderSceneData.h"
#include "../../SceneTextures.h"
#include "../../RenderSettingContext.h"

namespace Flower
{
    struct SSRPush
    {
        uint32_t samplesPerQuad;
        uint32_t temporalVarianceGuidedTracingEnabled;
        uint32_t mostDetailedMip = 0;
        float roughnessThreshold; // Max roughness stop to reflection sample.
        float temporalVarianceThreshold;
    };

    struct SSRRayCounterSSBO
    {
        uint32_t rayCount;
        uint32_t denoiseTileCount;
    };

    class SSRPass : public PassInterface
    {
    public:
        VkPipeline ssrTileClassifyPipeline = VK_NULL_HANDLE;
        VkPipeline ssrArgsPrepare = VK_NULL_HANDLE;
        VkPipeline ssrIntersect = VK_NULL_HANDLE;

        VkPipeline ssrReproject = VK_NULL_HANDLE;
        VkPipeline ssrPrefilter = VK_NULL_HANDLE;
        VkPipeline ssrTemporal = VK_NULL_HANDLE;

        VkPipeline ssrApplyPipeline = VK_NULL_HANDLE;

        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;

        VkDescriptorSet sets[3];

        // SSR history.
        PoolImageSharedRef rt_ssrPrevRadiance = nullptr;
        PoolImageSharedRef rt_ssrPrevVariance = nullptr;
        PoolImageSharedRef rt_ssrPrevRoughness = nullptr; // history roughness, from ssr.
        PoolImageSharedRef rt_ssrPrevSampleCount = nullptr; // 

        PoolImageSharedRef rt_ssrRadiance = nullptr;
        PoolImageSharedRef rt_ssrVariance = nullptr;
        PoolImageSharedRef rt_ssrRoughness = nullptr; // history roughness, from ssr.
        PoolImageSharedRef rt_ssrSampleCount = nullptr; // 

        PoolImageSharedRef rt_ssrReproject = nullptr; // 
        PoolImageSharedRef rt_ssrAverageRadiance = nullptr; // 
    public:
        void updateRTsBeforeRender(uint32_t width, uint32_t height, RenderTexturePool* pool, VkCommandBuffer cmd)
        {
            bool bShouldRebuild = false;

            if (rt_ssrReproject == nullptr)
            {
                bShouldRebuild = true;
            }
            else if (rt_ssrReproject->getImage().getExtent().width != width || rt_ssrReproject->getImage().getExtent().height != height)
            {
                bShouldRebuild = true;
            }

            if (!bShouldRebuild)
            {
                return;
            }

            rt_ssrRadiance = pool->createPoolImage(
                "SSR Radiance 0",
                width,
                height,
                VK_FORMAT_R16G16B16A16_SFLOAT,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

            rt_ssrPrevRadiance = pool->createPoolImage(
                "SSR Radiance 0",
                width,
                height,
                VK_FORMAT_R16G16B16A16_SFLOAT,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);




            rt_ssrReproject = pool->createPoolImage(
                "SSR reproject",
                width,
                height,
                VK_FORMAT_R16G16B16A16_SFLOAT,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);


            rt_ssrAverageRadiance = pool->createPoolImage(
                "SSR Average Radiance",
                divideRoundingUp(width, 8u),
                divideRoundingUp(height, 8u),
                VK_FORMAT_B10G11R11_UFLOAT_PACK32,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

            rt_ssrRoughness = pool->createPoolImage(
                "SSRExtractRoughness 0",
                width,
                height,
                VK_FORMAT_R8_UNORM,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

            rt_ssrPrevRoughness = pool->createPoolImage(
                "SSRExtractRoughness - 1",
                width,
                height,
                VK_FORMAT_R8_UNORM,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

            rt_ssrVariance = pool->createPoolImage(
                "SSR Variance 0",
                width,
                height,
                VK_FORMAT_R16_SFLOAT,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

            rt_ssrPrevVariance = pool->createPoolImage(
                "SSR Variance - 1",
                width,
                height,
                VK_FORMAT_R16_SFLOAT,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

            


            rt_ssrSampleCount = pool->createPoolImage(
                "SSR SampleCount",
                width,
                height,
                VK_FORMAT_R16_SFLOAT,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

            rt_ssrPrevSampleCount = pool->createPoolImage(
                "SSR SampleCount - 1",
                width,
                height,
                VK_FORMAT_R16_SFLOAT,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

            rt_ssrRadiance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            rt_ssrPrevRadiance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            rt_ssrReproject->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            rt_ssrAverageRadiance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            rt_ssrPrevRoughness->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            rt_ssrRoughness->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            rt_ssrVariance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            rt_ssrPrevVariance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            rt_ssrSampleCount->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            rt_ssrPrevSampleCount->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

            VkClearColorValue clearValue = {};
            clearValue.float32[0] = 0;
            clearValue.float32[1] = 0;
            clearValue.float32[2] = 0;
            clearValue.float32[3] = 0;

            VkImageSubresourceRange subresourceRange = buildBasicImageSubresource();

            // Initial resource clears
            vkCmdClearColorImage(cmd, rt_ssrRadiance->getImage().getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &subresourceRange);
            vkCmdClearColorImage(cmd, rt_ssrPrevRadiance->getImage().getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &subresourceRange);
            vkCmdClearColorImage(cmd, rt_ssrReproject->getImage().getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &subresourceRange);
            vkCmdClearColorImage(cmd, rt_ssrAverageRadiance->getImage().getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &subresourceRange);
            vkCmdClearColorImage(cmd, rt_ssrPrevRoughness->getImage().getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &subresourceRange);
            vkCmdClearColorImage(cmd, rt_ssrRoughness->getImage().getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &subresourceRange);
            vkCmdClearColorImage(cmd, rt_ssrVariance->getImage().getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &subresourceRange);
            vkCmdClearColorImage(cmd, rt_ssrPrevVariance->getImage().getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &subresourceRange);
            vkCmdClearColorImage(cmd, rt_ssrSampleCount->getImage().getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &subresourceRange);
            vkCmdClearColorImage(cmd, rt_ssrPrevSampleCount->getImage().getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &subresourceRange);

                
            rt_ssrRadiance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            rt_ssrPrevRadiance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            rt_ssrReproject->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            rt_ssrAverageRadiance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            rt_ssrPrevRoughness->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            rt_ssrRoughness->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            rt_ssrVariance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            rt_ssrPrevVariance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            rt_ssrSampleCount->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            rt_ssrPrevSampleCount->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        }

        void swapRTAfterRender()
        {
            auto tempRadiance = rt_ssrPrevRadiance;
            auto tempVariance = rt_ssrPrevVariance;
            auto tempRoughness = rt_ssrPrevRoughness;
            auto tempSampleCount = rt_ssrPrevSampleCount;

            rt_ssrPrevRadiance = rt_ssrRadiance;
            rt_ssrPrevVariance = rt_ssrVariance;
            rt_ssrPrevRoughness = rt_ssrRoughness;
            rt_ssrPrevSampleCount = rt_ssrSampleCount;

            rt_ssrRadiance = tempRadiance;
            rt_ssrVariance = tempVariance;
            rt_ssrRoughness = tempRoughness;
            rt_ssrSampleCount = tempSampleCount;
        }

        virtual void init() override
        {
            CHECK(setLayout == VK_NULL_HANDLE);
            CHECK(pipelineLayout == VK_NULL_HANDLE);

            for (auto i = 0; i < 3; i++)
            {
                // Config code.
                RHI::get()->descriptorFactoryBegin()
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0) // inHiz
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1) // inDepth
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 2) // inGbufferA
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 3) // inGbufferB
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 4) // inGbufferS
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 5) // in Velocity
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 6) // inPrevDepth
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 7) // inPrevGBufferB
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 8) // inHDRSceneColor
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 9) // inBRDFLut
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 10) // SSRRayCounterSSBO
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 11) // SSRRayListSSBO
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 12) // SSRDenoiseTileListSSBO
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 13) // HDRSceneColorImage
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 14) // inCubeGlobalPrefilter
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 15) // inGTAO
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 16) // SSRIntersectCmdSSBO
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 17) // SSRDenoiseCmdSSBO

                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 18) // SSRExtractRoughness
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 19) // inSSRExtractRoughness
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 20) // SSRIntersection
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 21) // inSSRIntersection

                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 22) // SSR prev frame roughness
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 23) // SSR prev frame radiance
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 24) // SSR prev frame sample count

                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 25) // SSR reproject radiance
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 26) // SSR average radiance
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 27) // SSR variance
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 28) // SSR sample count

                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 29) // in SSR reproject radiance
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 30) // in SSR average radiance
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 31) // in SSR variance
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 32) // in SSR variance history

                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 33) // SSR prefilter radiance
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 34) // SSR prefilter variance

                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 35) // in SSR prefilter radiance
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 36) // in SSR prefilter variance

                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 37) // SSR temporal radiance
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 38) // SSR temporal variance

                    .buildNoInfo(setLayout, sets[i]);
            }

         

            std::vector<VkDescriptorSetLayout> setLayouts =
            {
                setLayout, // Owner setlayout.
                GetLayoutStatic(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER),  // viewData
                GetLayoutStatic(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER), // frameData
                RHI::SamplerManager->getCommonDescriptorSetLayout(), // sampler
                BlueNoiseMisc::getSetLayout(), // Bluenoise
                StaticTexturesManager::get()->globalBlueNoise.spp_1_buffer.setLayouts, // All blue noise set layout is same.
            };

            VkPushConstantRange pushRange{ .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(SSRPush) };

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
                CHECK(ssrTileClassifyPipeline == VK_NULL_HANDLE);
                auto shaderModule = RHI::ShaderManager->getShader("SSRTileClassify.comp.spv", true);

                shaderStageCI.module = shaderModule;
                computePipelineCreateInfo.stage = shaderStageCI;

                RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &ssrTileClassifyPipeline));
            }
            {
                CHECK(ssrArgsPrepare == VK_NULL_HANDLE);
                auto shaderModule = RHI::ShaderManager->getShader("SSRIntersectArgs.comp.spv", true);

                shaderStageCI.module = shaderModule;
                computePipelineCreateInfo.stage = shaderStageCI;

                RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &ssrArgsPrepare));
            }
            {
                CHECK(ssrIntersect == VK_NULL_HANDLE);
                auto shaderModule = RHI::ShaderManager->getShader("SSRIntersect.comp.spv", true);

                shaderStageCI.module = shaderModule;
                computePipelineCreateInfo.stage = shaderStageCI;

                RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &ssrIntersect));
            }

            {
                CHECK(ssrReproject == VK_NULL_HANDLE);
                auto shaderModule = RHI::ShaderManager->getShader("SSRReproject.comp.spv", true);

                shaderStageCI.module = shaderModule;
                computePipelineCreateInfo.stage = shaderStageCI;

                RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &ssrReproject));
            }
            {
                CHECK(ssrPrefilter == VK_NULL_HANDLE);
                auto shaderModule = RHI::ShaderManager->getShader("SSRPrefilter.comp.spv", true);

                shaderStageCI.module = shaderModule;
                computePipelineCreateInfo.stage = shaderStageCI;

                RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &ssrPrefilter));
            }
            {
                CHECK(ssrTemporal == VK_NULL_HANDLE);
                auto shaderModule = RHI::ShaderManager->getShader("SSRTemporalFilter.comp.spv", true);

                shaderStageCI.module = shaderModule;
                computePipelineCreateInfo.stage = shaderStageCI;

                RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &ssrTemporal));
            }
            {
                CHECK(ssrApplyPipeline == VK_NULL_HANDLE);
                auto shaderModule = RHI::ShaderManager->getShader("SSRApply.comp.spv", true);

                shaderStageCI.module = shaderModule;
                computePipelineCreateInfo.stage = shaderStageCI;

                RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &ssrApplyPipeline));
            }
        }

        virtual void release() override
        {
            RHISafeRelease(pipelineLayout);
            setLayout = VK_NULL_HANDLE;

            RHISafeRelease(ssrArgsPrepare);
            RHISafeRelease(ssrTileClassifyPipeline);
            RHISafeRelease(ssrApplyPipeline);
            RHISafeRelease(ssrIntersect);

            RHISafeRelease(ssrReproject);
            RHISafeRelease(ssrPrefilter);
            RHISafeRelease(ssrTemporal);
        }
    };

    void DeferredRenderer::renderSSR(
        VkCommandBuffer cmd,
        Renderer* renderer,
        SceneTextures* inTextures,
        RenderSceneData* scene,
        BufferParamRefPointer& viewData,
        BufferParamRefPointer& frameData,
        PoolImageSharedRef inHiz,
        PoolImageSharedRef inGTAO,
        BlueNoiseMisc& inBlueNoise)
    {
        auto& hdrSceneColor = inTextures->getHdrSceneColor()->getImage();
        auto& gbufferA = inTextures->getGbufferA()->getImage();
        auto& gbufferB = inTextures->getGbufferB()->getImage();
        auto& gbufferS = inTextures->getGbufferS()->getImage();
        auto& gbufferV = inTextures->getGbufferV()->getImage();
        auto& sceneDepthZ = inTextures->getDepth()->getImage();
        auto& skyPrefilterImage = inTextures->getSkyPrefilter()->getImage();

        hdrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        gbufferA.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        gbufferB.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        gbufferS.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        gbufferV.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        sceneDepthZ.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));
        skyPrefilterImage.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresourceCube());

        VkDescriptorImageInfo hizInfo = RHIDescriptorImageInfoSample(inHiz->getImage().getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo depthInfo = RHIDescriptorImageInfoSample(sceneDepthZ.getView(RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT)));
        VkDescriptorImageInfo gbufferAInfo = RHIDescriptorImageInfoSample(gbufferA.getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo gbufferBInfo = RHIDescriptorImageInfoSample(gbufferB.getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo gbufferSInfo = RHIDescriptorImageInfoSample(gbufferS.getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo gbufferVInfo = RHIDescriptorImageInfoSample(gbufferV.getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo brdfLutInfo = RHIDescriptorImageInfoSample(StaticTexturesManager::get()->getBRDFLut()->getImage().getView(buildBasicImageSubresource()));

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

        VkImageView globalPrefilterView = skyPrefilterImage.getView(buildBasicImageSubresourceCube(), VK_IMAGE_VIEW_TYPE_CUBE);
        VkDescriptorImageInfo globalPrefilterInfo = RHIDescriptorImageInfoSample(globalPrefilterView);

        VkDescriptorImageInfo hdrImageInfo = RHIDescriptorImageInfoStorage(hdrSceneColor.getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo hdrSampleInfo = RHIDescriptorImageInfoSample(hdrSceneColor.getView(buildBasicImageSubresource()));
        if (m_prevHDR)
        {
            hdrSampleInfo = RHIDescriptorImageInfoSample(m_prevHDR->getImage().getView(buildBasicImageSubresource()));
        }

        auto* pass = getPasses()->getPass<SSRPass>();

        pass->updateRTsBeforeRender(sceneDepthZ.getExtent().width, sceneDepthZ.getExtent().height, m_rtPool.get(), cmd);

        
        VkDescriptorImageInfo ssrReprojectImageInfo = RHIDescriptorImageInfoStorage(pass->rt_ssrReproject->getImage().getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo ssrReprojectInfo = RHIDescriptorImageInfoSample(pass->rt_ssrReproject->getImage().getView(buildBasicImageSubresource()));

        
        VkDescriptorImageInfo ssrVarianceImageInfo = RHIDescriptorImageInfoStorage(pass->rt_ssrVariance->getImage().getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo ssrVarianceInfo = RHIDescriptorImageInfoSample(pass->rt_ssrVariance->getImage().getView(buildBasicImageSubresource()));


        
        VkDescriptorImageInfo ssrVarianceHistoryImageInfo = RHIDescriptorImageInfoStorage(pass->rt_ssrPrevVariance->getImage().getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo ssrVarianceHistoryInfo = RHIDescriptorImageInfoSample(pass->rt_ssrPrevVariance->getImage().getView(buildBasicImageSubresource()));

        
       
        VkDescriptorImageInfo ssrSampleCountImageInfo = RHIDescriptorImageInfoStorage(pass->rt_ssrSampleCount->getImage().getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo ssrSampleCountHistoryInfo = RHIDescriptorImageInfoSample(pass->rt_ssrPrevSampleCount->getImage().getView(buildBasicImageSubresource()));


       
        VkDescriptorImageInfo ssrIntersectImageInfo = RHIDescriptorImageInfoStorage(pass->rt_ssrRadiance->getImage().getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo ssrIntersectInfo = RHIDescriptorImageInfoSample(pass->rt_ssrRadiance->getImage().getView(buildBasicImageSubresource()));


        
        VkDescriptorImageInfo ssrIntersectHistoryImageInfo = RHIDescriptorImageInfoStorage(pass->rt_ssrPrevRadiance->getImage().getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo ssrIntersectHistoryInfo = RHIDescriptorImageInfoSample(pass->rt_ssrPrevRadiance->getImage().getView(buildBasicImageSubresource()));

        
        VkDescriptorImageInfo ssrAverageImageInfo = RHIDescriptorImageInfoStorage(pass->rt_ssrAverageRadiance->getImage().getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo ssrAverageInfo = RHIDescriptorImageInfoSample(pass->rt_ssrAverageRadiance->getImage().getView(buildBasicImageSubresource()));

        
        VkDescriptorImageInfo roughnessExtractHistoryInfo = RHIDescriptorImageInfoSample(pass->rt_ssrPrevRoughness->getImage().getView(buildBasicImageSubresource()));

        
        VkDescriptorImageInfo roughnessExtractImageInfo = RHIDescriptorImageInfoStorage(pass->rt_ssrRoughness->getImage().getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo roughnessExtractInfo = RHIDescriptorImageInfoSample(pass->rt_ssrRoughness->getImage().getView(buildBasicImageSubresource()));



        const uint32_t maxRayCount = sceneDepthZ.getExtent().width* sceneDepthZ.getExtent().height; // Max case is one pixel one ray.
        const uint32_t maxDenoiseListCount = maxRayCount / (8 * 8) + 1; // Tile run in 8x8.

        auto ssboCounterBuffer = getBuffers()->getStaticStorage("ssrCounterBuffer", sizeof(SSRRayCounterSSBO));
        auto ssboRayListBuffer = getBuffers()->getStaticStorage("ssrRayListBuffer", sizeof(uint32_t) * maxRayCount);
        auto ssboDenoiseListBuffer = getBuffers()->getStaticStorage("ssrDenoiseListBuffer", sizeof(uint32_t) * maxDenoiseListCount);

        auto ssboIntersectCmdBuffer = getBuffers()->getIndirectStorage("ssboIntersectCmdBuffer", sizeof(GPUDispatchIndirectCommand));
        auto ssboDenoiseCmdBuffer = getBuffers()->getIndirectStorage("ssboDenoiseCmdBuffer", sizeof(GPUDispatchIndirectCommand));


        // Clear count buffer. list buffer don't care and it will update by thread.
        vkCmdFillBuffer(cmd, *ssboCounterBuffer->buffer.getBuffer(), 0, ssboCounterBuffer->buffer.getBuffer()->getSize(), 0u);
        std::array<VkBufferMemoryBarrier2, 1> fillBarriers
        {
            RHIBufferBarrier(ssboCounterBuffer->buffer.getBuffer()->getVkBuffer(),
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT),
        };
        RHIPipelineBarrier(cmd, 0, (uint32_t)fillBarriers.size(), fillBarriers.data(), 0, nullptr);

        VkDescriptorBufferInfo ssboCounterBufferInfo = ssboCounterBuffer->buffer.getBufferInfo();
        VkDescriptorBufferInfo ssboRayListBufferInfo = ssboRayListBuffer->buffer.getBufferInfo();
        VkDescriptorBufferInfo ssboDenoiseListBufferInfo = ssboDenoiseListBuffer->buffer.getBufferInfo();
        VkDescriptorBufferInfo ssboArgsIntersectInfo = ssboIntersectCmdBuffer->buffer.getBufferInfo();
        VkDescriptorBufferInfo ssboArgsDenoiseInfo = ssboDenoiseCmdBuffer->buffer.getBufferInfo();
        VkDescriptorImageInfo gtaoInfo = RHIDescriptorImageInfoSample(inGTAO->getImage().getView(buildBasicImageSubresource()));


        std::vector<VkWriteDescriptorSet> writes
        {
            RHIPushWriteDescriptorSetImage(0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &hizInfo),
            RHIPushWriteDescriptorSetImage(1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &depthInfo),
            RHIPushWriteDescriptorSetImage(2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &gbufferAInfo),
            RHIPushWriteDescriptorSetImage(3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &gbufferBInfo),
            RHIPushWriteDescriptorSetImage(4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &gbufferSInfo),
            RHIPushWriteDescriptorSetImage(5, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &gbufferVInfo),
            RHIPushWriteDescriptorSetImage(6, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &prevDepthInfo),
            RHIPushWriteDescriptorSetImage(7, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &preGBufferBInfo),
            RHIPushWriteDescriptorSetImage(8, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &hdrSampleInfo),
            RHIPushWriteDescriptorSetImage(9, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &brdfLutInfo),
            RHIPushWriteDescriptorSetBuffer(10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &ssboCounterBufferInfo),
            RHIPushWriteDescriptorSetBuffer(11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &ssboRayListBufferInfo),
            RHIPushWriteDescriptorSetBuffer(12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &ssboDenoiseListBufferInfo),
            RHIPushWriteDescriptorSetImage(13, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &hdrImageInfo),
            RHIPushWriteDescriptorSetImage(14, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &globalPrefilterInfo),
            RHIPushWriteDescriptorSetImage(15, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &gtaoInfo),
            RHIPushWriteDescriptorSetBuffer(16, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &ssboArgsIntersectInfo),
            RHIPushWriteDescriptorSetBuffer(17, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &ssboArgsDenoiseInfo),

            RHIPushWriteDescriptorSetImage(18, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &roughnessExtractImageInfo), // extract roughness
            RHIPushWriteDescriptorSetImage(19, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &roughnessExtractInfo),

            RHIPushWriteDescriptorSetImage(20, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ssrIntersectImageInfo), // ssr intersect result
            RHIPushWriteDescriptorSetImage(21, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &ssrIntersectInfo),

            RHIPushWriteDescriptorSetImage(22, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &roughnessExtractHistoryInfo), // inPrevSSRExtractRoughness
            RHIPushWriteDescriptorSetImage(23, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &ssrIntersectHistoryInfo), // inPrevSSRRadiance
            RHIPushWriteDescriptorSetImage(24, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &ssrSampleCountHistoryInfo), // inPrevSampleCount

            RHIPushWriteDescriptorSetImage(25, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ssrReprojectImageInfo), // ssr reproject
            RHIPushWriteDescriptorSetImage(26, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ssrAverageImageInfo), // ssr average
            RHIPushWriteDescriptorSetImage(27, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ssrVarianceImageInfo), // ssr variance
            RHIPushWriteDescriptorSetImage(28, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ssrSampleCountImageInfo), // ssr sample count

            RHIPushWriteDescriptorSetImage(29, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &ssrReprojectInfo),
            RHIPushWriteDescriptorSetImage(30, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &ssrAverageInfo), 
            RHIPushWriteDescriptorSetImage(31, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &ssrVarianceInfo), 
            RHIPushWriteDescriptorSetImage(32, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &ssrVarianceHistoryInfo), // ssr variance history

            RHIPushWriteDescriptorSetImage(33, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ssrIntersectHistoryImageInfo), // ssr prefilter radiance.
            RHIPushWriteDescriptorSetImage(34, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ssrVarianceHistoryImageInfo), // ssr prefilter variance.
            RHIPushWriteDescriptorSetImage(35, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &ssrIntersectHistoryInfo), // in ssr prefilter radiance.
            RHIPushWriteDescriptorSetImage(36, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &ssrVarianceHistoryInfo), // in ssr prefilter variance.

            RHIPushWriteDescriptorSetImage(37, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ssrIntersectImageInfo), // ssr temporal radiance.
            RHIPushWriteDescriptorSetImage(38, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ssrVarianceImageInfo), // ssr temporal variance.
        };



        auto& setActive = pass->sets[m_renderIndex % 3];

        for (auto& write : writes)
        {
            write.dstSet = setActive;
        }

        {
            vkUpdateDescriptorSets(RHI::Device, (uint32_t)writes.size(), writes.data(), 0, nullptr);
        }

        std::vector<VkDescriptorSet> passSets =
        {
            setActive,
            viewData->buffer.getSet(),  // viewData
            frameData->buffer.getSet(), // frameData
            RHI::SamplerManager->getCommonDescriptorSet(), // samplers.
            inBlueNoise.getSet(),
            StaticTexturesManager::get()->globalBlueNoise.spp_1_buffer.set // 1spp is good.
        };

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->pipelineLayout,
            0, (uint32_t)passSets.size(), passSets.data(), 0, nullptr);

        // TODO: Fill by context.
        SSRPush pushConst = 
        {
            .samplesPerQuad = 1,
            .temporalVarianceGuidedTracingEnabled = 1,
            .mostDetailedMip = 0,
            .roughnessThreshold = 0.2f,
            .temporalVarianceThreshold = 0.0f,
        };
        vkCmdPushConstants(cmd, pass->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConst), &pushConst);

        {
            RHI::ScopePerframeMarker marker(cmd, "classify ssr", { 1.0f, 1.0f, 0.0f, 1.0f });

            
            pass->rt_ssrRoughness->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            pass->rt_ssrRadiance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->ssrTileClassifyPipeline);

            vkCmdDispatch(cmd, getGroupCount(pass->rt_ssrRadiance->getImage().getExtent().width, 8), getGroupCount(pass->rt_ssrRadiance->getImage().getExtent().height, 8), 1);

            pass->rt_ssrRoughness->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            pass->rt_ssrRadiance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

            // 
            std::array<VkBufferMemoryBarrier2, 3> endBufferBarriers
            {
                RHIBufferBarrier(ssboCounterBuffer->buffer.getBuffer()->getVkBuffer(),
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  VK_ACCESS_SHADER_READ_BIT),

                RHIBufferBarrier(ssboRayListBuffer->buffer.getBuffer()->getVkBuffer(),
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_MEMORY_READ_BIT),

                RHIBufferBarrier(ssboDenoiseListBuffer->buffer.getBuffer()->getVkBuffer(),
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_MEMORY_READ_BIT),
            };
            RHIPipelineBarrier(cmd, 0, (uint32_t)endBufferBarriers.size(), endBufferBarriers.data(), 0, nullptr);
        }

        {
            RHI::ScopePerframeMarker marker(cmd, "ssr prepare args", { 1.0f, 1.0f, 0.0f, 1.0f });
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->ssrArgsPrepare);
            vkCmdDispatch(cmd, 1,1,1);


            std::array<VkBufferMemoryBarrier2, 2> endBufferBarriers
            {
                RHIBufferBarrier(ssboIntersectCmdBuffer->buffer.getBuffer()->getVkBuffer(),
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
                    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,  VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT),

                RHIBufferBarrier(ssboDenoiseCmdBuffer->buffer.getBuffer()->getVkBuffer(),
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
                    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,  VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT),
            };
            RHIPipelineBarrier(cmd, 0, (uint32_t)endBufferBarriers.size(), endBufferBarriers.data(), 0, nullptr);
        }

        {
            RHI::ScopePerframeMarker marker(cmd, "ssr intersect", { 1.0f, 1.0f, 0.0f, 1.0f });


            pass->rt_ssrRadiance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->ssrIntersect);

            vkCmdDispatchIndirect(cmd, ssboIntersectCmdBuffer->buffer.getBuffer()->getVkBuffer(), 0);

            pass->rt_ssrRadiance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        }

        {
            RHI::ScopePerframeMarker marker(cmd, "ssr reproject", { 1.0f, 1.0f, 0.0f, 1.0f });

            
            pass->rt_ssrReproject->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            pass->rt_ssrAverageRadiance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            pass->rt_ssrVariance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            pass->rt_ssrSampleCount->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->ssrReproject);

            vkCmdDispatchIndirect(cmd, ssboDenoiseCmdBuffer->buffer.getBuffer()->getVkBuffer(), 0);


            pass->rt_ssrReproject->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            pass->rt_ssrAverageRadiance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            pass->rt_ssrVariance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            pass->rt_ssrSampleCount->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        }

        {
            RHI::ScopePerframeMarker marker(cmd, "ssr prefilter", { 1.0f, 1.0f, 0.0f, 1.0f });

            pass->rt_ssrPrevRadiance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            pass->rt_ssrPrevVariance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->ssrPrefilter);
            vkCmdDispatchIndirect(cmd, ssboDenoiseCmdBuffer->buffer.getBuffer()->getVkBuffer(), 0);

            pass->rt_ssrPrevRadiance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            pass->rt_ssrPrevVariance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        }

        {
            RHI::ScopePerframeMarker marker(cmd, "ssr temporal", { 1.0f, 1.0f, 0.0f, 1.0f });

            pass->rt_ssrRadiance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            pass->rt_ssrVariance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->ssrTemporal);
            vkCmdDispatchIndirect(cmd, ssboDenoiseCmdBuffer->buffer.getBuffer()->getVkBuffer(), 0);

            pass->rt_ssrRadiance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            pass->rt_ssrVariance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        }

        {
            RHI::ScopePerframeMarker marker(cmd, "apply reflection", { 1.0f, 1.0f, 0.0f, 1.0f });

            hdrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->ssrApplyPipeline);

            vkCmdDispatch(cmd, getGroupCount(hdrSceneColor.getExtent().width, 8), getGroupCount(hdrSceneColor.getExtent().height, 8), 1);

            hdrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        }

        pass->swapRTAfterRender();

        m_gpuTimer.getTimeStamp(cmd, "SSR");
    }
}