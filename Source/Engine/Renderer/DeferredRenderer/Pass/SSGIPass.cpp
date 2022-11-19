#include "Pch.h"
#include "../DeferredRenderer.h"
#include "../../Renderer.h"
#include "../../RenderSceneData.h"
#include "../../SceneTextures.h"
#include "../../RenderSettingContext.h"

namespace Flower
{


    struct SSGIPush
    {
        float intensity;
    };

    class SSGIPass : public PassInterface
    {
    public:
        VkPipeline ssgi_Intersect = VK_NULL_HANDLE;
        VkPipeline ssgi_Reproject = VK_NULL_HANDLE;
        VkPipeline ssgi_Prefilter = VK_NULL_HANDLE;
        VkPipeline ssgi_Temporal = VK_NULL_HANDLE;
        VkPipeline ssgi_ApplyPipeline = VK_NULL_HANDLE;

        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;

        VkDescriptorSet sets[3];

        // ssgi history.
        PoolImageSharedRef rt_ssgiPrevRadiance = nullptr;
        PoolImageSharedRef rt_ssgiPrevVariance = nullptr;
        PoolImageSharedRef rt_ssgiPrevSampleCount = nullptr; // 

        PoolImageSharedRef rt_ssgiRadiance = nullptr;
        PoolImageSharedRef rt_ssgiVariance = nullptr;
        PoolImageSharedRef rt_ssgiSampleCount = nullptr; // 

        PoolImageSharedRef rt_ssgiReproject = nullptr; // 
        PoolImageSharedRef rt_ssgiAverageRadiance = nullptr; // 

    public:
        void updateRTsBeforeRender(uint32_t width, uint32_t height, RenderTexturePool* pool, VkCommandBuffer cmd)
        {
            bool bShouldRebuild = false;

            if (rt_ssgiReproject == nullptr)
            {
                bShouldRebuild = true;
            }
            else if (rt_ssgiReproject->getImage().getExtent().width != width || rt_ssgiReproject->getImage().getExtent().height != height)
            {
                bShouldRebuild = true;
            }

            if (!bShouldRebuild)
            {
                return;
            }

            rt_ssgiRadiance = pool->createPoolImage(
                "ssgi Radiance 0",
                width,
                height,
                VK_FORMAT_R16G16B16A16_SFLOAT,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

            rt_ssgiPrevRadiance = pool->createPoolImage(
                "ssgi Radiance 0",
                width,
                height,
                VK_FORMAT_R16G16B16A16_SFLOAT,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);




            rt_ssgiReproject = pool->createPoolImage(
                "ssgi reproject",
                width,
                height,
                VK_FORMAT_R16G16B16A16_SFLOAT,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);


            rt_ssgiAverageRadiance = pool->createPoolImage(
                "ssgi Average Radiance",
                divideRoundingUp(width, 8u),
                divideRoundingUp(height, 8u),
                VK_FORMAT_B10G11R11_UFLOAT_PACK32,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

            rt_ssgiVariance = pool->createPoolImage(
                "ssgi Variance 0",
                width,
                height,
                VK_FORMAT_R16_SFLOAT,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

            rt_ssgiPrevVariance = pool->createPoolImage(
                "ssgi Variance - 1",
                width,
                height,
                VK_FORMAT_R16_SFLOAT,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);




            rt_ssgiSampleCount = pool->createPoolImage(
                "ssgi SampleCount",
                width,
                height,
                VK_FORMAT_R16_SFLOAT,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

            rt_ssgiPrevSampleCount = pool->createPoolImage(
                "ssgi SampleCount - 1",
                width,
                height,
                VK_FORMAT_R16_SFLOAT,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

            rt_ssgiRadiance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            rt_ssgiPrevRadiance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            rt_ssgiReproject->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            rt_ssgiAverageRadiance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            rt_ssgiVariance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            rt_ssgiPrevVariance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            rt_ssgiSampleCount->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            rt_ssgiPrevSampleCount->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

            VkClearColorValue clearValue = {};
            clearValue.float32[0] = 0;
            clearValue.float32[1] = 0;
            clearValue.float32[2] = 0;
            clearValue.float32[3] = 0;

            VkImageSubresourceRange subresourceRange = buildBasicImageSubresource();

            // Initial resource clears
            vkCmdClearColorImage(cmd, rt_ssgiRadiance->getImage().getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &subresourceRange);
            vkCmdClearColorImage(cmd, rt_ssgiPrevRadiance->getImage().getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &subresourceRange);
            vkCmdClearColorImage(cmd, rt_ssgiReproject->getImage().getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &subresourceRange);
            vkCmdClearColorImage(cmd, rt_ssgiAverageRadiance->getImage().getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &subresourceRange);

            vkCmdClearColorImage(cmd, rt_ssgiVariance->getImage().getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &subresourceRange);
            vkCmdClearColorImage(cmd, rt_ssgiPrevVariance->getImage().getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &subresourceRange);
            vkCmdClearColorImage(cmd, rt_ssgiSampleCount->getImage().getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &subresourceRange);
            vkCmdClearColorImage(cmd, rt_ssgiPrevSampleCount->getImage().getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &subresourceRange);


            rt_ssgiRadiance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            rt_ssgiPrevRadiance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            rt_ssgiReproject->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            rt_ssgiAverageRadiance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            rt_ssgiVariance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            rt_ssgiPrevVariance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            rt_ssgiSampleCount->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            rt_ssgiPrevSampleCount->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        }

        void swapRTAfterRender()
        {
            auto tempRadiance = rt_ssgiPrevRadiance;
            auto tempVariance = rt_ssgiPrevVariance;
            auto tempSampleCount = rt_ssgiPrevSampleCount;

            rt_ssgiPrevRadiance = rt_ssgiRadiance;
            rt_ssgiPrevVariance = rt_ssgiVariance;
            rt_ssgiPrevSampleCount = rt_ssgiSampleCount;

            rt_ssgiRadiance = tempRadiance;
            rt_ssgiVariance = tempVariance;
            rt_ssgiSampleCount = tempSampleCount;
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
   
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 9) // HDRSceneColorImage
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 10) // inCubeGlobalPrefilter
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 11) // inGTAO
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 12) // SSRIntersection
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 13) // inSSRIntersection
                   
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 14) // SSR prev frame radiance
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 15) // SSR prev frame sample count

                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 16) // SSR reproject radiance
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 17) // SSR average radiance
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 18) // SSR variance
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 19) // SSR sample count

                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 20) // in SSR reproject radiance
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 21) // in SSR average radiance
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 22) // in SSR variance
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 23) // in SSR variance history

                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 24) // SSR prefilter radiance
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 25) // SSR prefilter variance

                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 26) // in SSR prefilter radiance
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 27) // in SSR prefilter variance

                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 28) // SSR temporal radiance
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 29) // SSR temporal variance

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

            VkPushConstantRange pushRange{ .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(SSGIPush) };

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
                CHECK(ssgi_Intersect == VK_NULL_HANDLE);
                auto shaderModule = RHI::ShaderManager->getShader("SSGI_Intersect.comp.spv", true);

                shaderStageCI.module = shaderModule;
                computePipelineCreateInfo.stage = shaderStageCI;

                RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &ssgi_Intersect));
            }

            {
                CHECK(ssgi_Reproject == VK_NULL_HANDLE);
                auto shaderModule = RHI::ShaderManager->getShader("SSGI_Reproject.comp.spv", true);

                shaderStageCI.module = shaderModule;
                computePipelineCreateInfo.stage = shaderStageCI;

                RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &ssgi_Reproject));
            }
            {
                CHECK(ssgi_Prefilter == VK_NULL_HANDLE);
                auto shaderModule = RHI::ShaderManager->getShader("SSGI_Prefilter.comp.spv", true);

                shaderStageCI.module = shaderModule;
                computePipelineCreateInfo.stage = shaderStageCI;

                RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &ssgi_Prefilter));
            }
            {
                CHECK(ssgi_Temporal == VK_NULL_HANDLE);
                auto shaderModule = RHI::ShaderManager->getShader("SSGI_Temporal.comp.spv", true);

                shaderStageCI.module = shaderModule;
                computePipelineCreateInfo.stage = shaderStageCI;

                RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &ssgi_Temporal));
            }
            {
                CHECK(ssgi_ApplyPipeline == VK_NULL_HANDLE);
                auto shaderModule = RHI::ShaderManager->getShader("SSGI_Apply.comp.spv", true);

                shaderStageCI.module = shaderModule;
                computePipelineCreateInfo.stage = shaderStageCI;

                RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &ssgi_ApplyPipeline));
            }
        }

        virtual void release() override
        {
            RHISafeRelease(pipelineLayout);
            setLayout = VK_NULL_HANDLE;

            RHISafeRelease(ssgi_ApplyPipeline);
            RHISafeRelease(ssgi_Intersect);

            RHISafeRelease(ssgi_Reproject);
            RHISafeRelease(ssgi_Prefilter);
            RHISafeRelease(ssgi_Temporal);
        }
    };

    void DeferredRenderer::renderSSGI(
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
        auto& hdrSceneColor = inTextures->getHdrSceneColor()->getImage(); // Direct light color buffer.

        auto& gbufferA = inTextures->getGbufferA()->getImage();
        auto& gbufferB = inTextures->getGbufferB()->getImage();
        auto& gbufferS = inTextures->getGbufferS()->getImage();
        auto& gbufferV = inTextures->getGbufferV()->getImage();
        auto& sceneDepthZ = inTextures->getDepth()->getImage();
        auto& atmosphereEnvCubeImage = inTextures->getAtmosphereEnvCapture()->getImage();

        hdrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        gbufferA.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        gbufferB.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        gbufferS.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        gbufferV.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        sceneDepthZ.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));
        atmosphereEnvCubeImage.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresourceCube());

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

        VkImageView globalPrefilterView = atmosphereEnvCubeImage.getView(buildBasicImageSubresourceCube(), VK_IMAGE_VIEW_TYPE_CUBE);
        if (StaticTexturesManager::get()->isIBLReady())
        {
            globalPrefilterView = StaticTexturesManager::get()->getIBLIrradiance()->getImage().getView(buildBasicImageSubresourceCube(), VK_IMAGE_VIEW_TYPE_CUBE);
        }
        VkDescriptorImageInfo globalPrefilterInfo = RHIDescriptorImageInfoSample(globalPrefilterView);

        VkDescriptorImageInfo hdrImageInfo = RHIDescriptorImageInfoStorage(hdrSceneColor.getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo hdrSampleInfo = RHIDescriptorImageInfoSample(hdrSceneColor.getView(buildBasicImageSubresource()));
        if (m_prevHDR)
        {
            hdrSampleInfo = RHIDescriptorImageInfoSample(m_prevHDR->getImage().getView(buildBasicImageSubresource()));
        }


        auto* pass = getPasses()->getPass<SSGIPass>();

        pass->updateRTsBeforeRender(sceneDepthZ.getExtent().width, sceneDepthZ.getExtent().height, m_rtPool.get(), cmd);


        VkDescriptorImageInfo ssrReprojectImageInfo = RHIDescriptorImageInfoStorage(pass->rt_ssgiReproject->getImage().getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo ssrReprojectInfo = RHIDescriptorImageInfoSample(pass->rt_ssgiReproject->getImage().getView(buildBasicImageSubresource()));


        VkDescriptorImageInfo ssrVarianceImageInfo = RHIDescriptorImageInfoStorage(pass->rt_ssgiVariance->getImage().getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo ssrVarianceInfo = RHIDescriptorImageInfoSample(pass->rt_ssgiVariance->getImage().getView(buildBasicImageSubresource()));



        VkDescriptorImageInfo ssrVarianceHistoryImageInfo = RHIDescriptorImageInfoStorage(pass->rt_ssgiPrevVariance->getImage().getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo ssrVarianceHistoryInfo = RHIDescriptorImageInfoSample(pass->rt_ssgiPrevVariance->getImage().getView(buildBasicImageSubresource()));



        VkDescriptorImageInfo ssrSampleCountImageInfo = RHIDescriptorImageInfoStorage(pass->rt_ssgiSampleCount->getImage().getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo ssrSampleCountHistoryInfo = RHIDescriptorImageInfoSample(pass->rt_ssgiPrevSampleCount->getImage().getView(buildBasicImageSubresource()));



        VkDescriptorImageInfo ssrIntersectImageInfo = RHIDescriptorImageInfoStorage(pass->rt_ssgiRadiance->getImage().getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo ssrIntersectInfo = RHIDescriptorImageInfoSample(pass->rt_ssgiRadiance->getImage().getView(buildBasicImageSubresource()));



        VkDescriptorImageInfo ssrIntersectHistoryImageInfo = RHIDescriptorImageInfoStorage(pass->rt_ssgiPrevRadiance->getImage().getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo ssrIntersectHistoryInfo = RHIDescriptorImageInfoSample(pass->rt_ssgiPrevRadiance->getImage().getView(buildBasicImageSubresource()));


        VkDescriptorImageInfo ssrAverageImageInfo = RHIDescriptorImageInfoStorage(pass->rt_ssgiAverageRadiance->getImage().getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo ssrAverageInfo = RHIDescriptorImageInfoSample(pass->rt_ssgiAverageRadiance->getImage().getView(buildBasicImageSubresource()));


        const uint32_t maxRayCount = sceneDepthZ.getExtent().width * sceneDepthZ.getExtent().height; // Max case is one pixel one ray.
        const uint32_t maxDenoiseListCount = maxRayCount / (8 * 8) + 1; // Tile run in 8x8.

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
            RHIPushWriteDescriptorSetImage(9, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &hdrImageInfo),
            RHIPushWriteDescriptorSetImage(10, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &globalPrefilterInfo),
            RHIPushWriteDescriptorSetImage(11, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &gtaoInfo),

            RHIPushWriteDescriptorSetImage(12, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ssrIntersectImageInfo), // ssr intersect result
            RHIPushWriteDescriptorSetImage(13, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &ssrIntersectInfo),

            RHIPushWriteDescriptorSetImage(14, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &ssrIntersectHistoryInfo), // inPrevSSRRadiance
            RHIPushWriteDescriptorSetImage(15, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &ssrSampleCountHistoryInfo), // inPrevSampleCount

            RHIPushWriteDescriptorSetImage(16, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ssrReprojectImageInfo), // ssr reproject
            RHIPushWriteDescriptorSetImage(17, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ssrAverageImageInfo), // ssr average
            RHIPushWriteDescriptorSetImage(18, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ssrVarianceImageInfo), // ssr variance
            RHIPushWriteDescriptorSetImage(19, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ssrSampleCountImageInfo), // ssr sample count

            RHIPushWriteDescriptorSetImage(20, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &ssrReprojectInfo),
            RHIPushWriteDescriptorSetImage(21, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &ssrAverageInfo),
            RHIPushWriteDescriptorSetImage(22, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &ssrVarianceInfo),
            RHIPushWriteDescriptorSetImage(23, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &ssrVarianceHistoryInfo), // ssr variance history

            RHIPushWriteDescriptorSetImage(24, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ssrIntersectHistoryImageInfo), // ssr prefilter radiance.
            RHIPushWriteDescriptorSetImage(25, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ssrVarianceHistoryImageInfo), // ssr prefilter variance.
            RHIPushWriteDescriptorSetImage(26, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &ssrIntersectHistoryInfo), // in ssr prefilter radiance.
            RHIPushWriteDescriptorSetImage(27, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &ssrVarianceHistoryInfo), // in ssr prefilter variance.

            RHIPushWriteDescriptorSetImage(28, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ssrIntersectImageInfo), // ssr temporal radiance.
            RHIPushWriteDescriptorSetImage(29, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ssrVarianceImageInfo), // ssr temporal variance.
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
        SSGIPush pushConst =
        {
            .intensity = 1.0,
        };
        vkCmdPushConstants(cmd, pass->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConst), &pushConst);

        {
            RHI::ScopePerframeMarker marker(cmd, "ssgi intersect", { 1.0f, 1.0f, 0.0f, 1.0f });


            pass->rt_ssgiRadiance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->ssgi_Intersect);

            vkCmdDispatch(cmd, getGroupCount(hdrSceneColor.getExtent().width, 8), getGroupCount(hdrSceneColor.getExtent().height, 8), 1);

            pass->rt_ssgiRadiance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        }

        {
            RHI::ScopePerframeMarker marker(cmd, "ssgi reproject", { 1.0f, 1.0f, 0.0f, 1.0f });


            pass->rt_ssgiReproject->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            pass->rt_ssgiAverageRadiance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            pass->rt_ssgiVariance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            pass->rt_ssgiSampleCount->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->ssgi_Reproject);

            vkCmdDispatch(cmd, getGroupCount(hdrSceneColor.getExtent().width, 8), getGroupCount(hdrSceneColor.getExtent().height, 8), 1);


            pass->rt_ssgiReproject->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            pass->rt_ssgiAverageRadiance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            pass->rt_ssgiVariance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            pass->rt_ssgiSampleCount->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        }

        {
            RHI::ScopePerframeMarker marker(cmd, "ssgi prefilter", { 1.0f, 1.0f, 0.0f, 1.0f });

            pass->rt_ssgiPrevRadiance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            pass->rt_ssgiPrevVariance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->ssgi_Prefilter);
            vkCmdDispatch(cmd, getGroupCount(hdrSceneColor.getExtent().width, 8), getGroupCount(hdrSceneColor.getExtent().height, 8), 1);

            pass->rt_ssgiPrevRadiance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            pass->rt_ssgiPrevVariance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        }

        {
            RHI::ScopePerframeMarker marker(cmd, "ssgi temporal", { 1.0f, 1.0f, 0.0f, 1.0f });

            pass->rt_ssgiRadiance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            pass->rt_ssgiVariance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->ssgi_Temporal);
            vkCmdDispatch(cmd, getGroupCount(hdrSceneColor.getExtent().width, 8), getGroupCount(hdrSceneColor.getExtent().height, 8), 1);

            pass->rt_ssgiRadiance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            pass->rt_ssgiVariance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        }

        {
            RHI::ScopePerframeMarker marker(cmd, "apply gi", { 1.0f, 1.0f, 0.0f, 1.0f });

            hdrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->ssgi_ApplyPipeline);

            vkCmdDispatch(cmd, getGroupCount(hdrSceneColor.getExtent().width, 8), getGroupCount(hdrSceneColor.getExtent().height, 8), 1);

            hdrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        }

        pass->swapRTAfterRender();

        m_gpuTimer.getTimeStamp(cmd, "SSGI");
    }
}