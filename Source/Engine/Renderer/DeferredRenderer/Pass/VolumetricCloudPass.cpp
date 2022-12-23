#include "Pch.h"
#include "../DeferredRenderer.h"
#include "../../Renderer.h"
#include "../../RenderSceneData.h"
#include "../../SceneTextures.h"
#include "../../../AssetSystem/TextureManager.h"

namespace Flower
{
    static AutoCVarCmd cVarUpdateCloudNoise("cmd.Cloud.NoiseUpdate", "Update cloud noise lut.");

    class VolumetricCloudPass : public PassInterface
    {
    public:
        VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;

        VkPipeline computeCloudPipeline = VK_NULL_HANDLE;
        VkPipelineLayout computeCloudPipelineLayout = VK_NULL_HANDLE;

        VkPipeline shadowMapPipeline = VK_NULL_HANDLE;
        VkPipeline reconstructionPipeline = VK_NULL_HANDLE;
        VkPipeline compositeCloudPipeline = VK_NULL_HANDLE;
        

    public:
        virtual void init() override
        {
            RHI::get()->descriptorFactoryBegin()
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, GCommonShaderStage, 0) // imageHdrSceneColor
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, GCommonShaderStage, 1) // inHdrSceneColor
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, GCommonShaderStage, 2) // imageCloudRenderTexture
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, GCommonShaderStage, 3) // inCloudRenderTexture
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, GCommonShaderStage, 4) // inDepth
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, GCommonShaderStage, 5) // inGBufferA
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, GCommonShaderStage, 6) // inBasicNoise
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, GCommonShaderStage, 7) // inDetailNoise
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, GCommonShaderStage, 8) // inCloudWeather
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, GCommonShaderStage, 9) // inCloudGradient
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, GCommonShaderStage, 10) // inTransmittanceLut
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, GCommonShaderStage, 11) // inFroxelScatter
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, GCommonShaderStage, 12) // imageShadowMapCloud
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, GCommonShaderStage, 13) // inShadowMapCloud
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, GCommonShaderStage, 14) // imageCloudReconstructionTexture
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, GCommonShaderStage, 15) // inCloudReconstructionTexture
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, GCommonShaderStage, 16) // imageCloudReconstructionTexture
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, GCommonShaderStage, 17) // inCloudReconstructionTexture
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, GCommonShaderStage, 18) // imageCloudReconstructionTexture
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, GCommonShaderStage, 19) // inCloudReconstructionTexture
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, GCommonShaderStage, 20) // inCloudReconstructionTexture
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, GCommonShaderStage, 21) // inCloudReconstructionTexture
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, GCommonShaderStage, 22) // imageCloudRenderTexture
                .buildNoInfoPush(setLayout);

            std::vector<VkDescriptorSetLayout> setLayouts =
            {
                  setLayout // Owner layout.
                , GetLayoutStatic(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)  // viewData
                , GetLayoutStatic(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)  // frameData
                , RHI::SamplerManager->getCommonDescriptorSetLayout() // Common samplers
                , BlueNoiseMisc::getSetLayout() // Bluenoise
                , StaticTexturesManager::get()->globalBlueNoise.spp_1_buffer.setLayouts, // All blue noise set layout is same.
            };

            // Cloud compute.
            {
                CHECK(computeCloudPipeline == VK_NULL_HANDLE);
                CHECK(computeCloudPipelineLayout == VK_NULL_HANDLE);

                auto shaderModule = RHI::ShaderManager->getShader("VolumetricCloudRayMarching.comp.spv", true);

                // Vulkan build functions.
                VkPipelineLayoutCreateInfo plci = RHIPipelineLayoutCreateInfo();
                plci.setLayoutCount = (uint32_t)setLayouts.size();
                plci.pSetLayouts = setLayouts.data();
                computeCloudPipelineLayout = RHI::get()->createPipelineLayout(plci);

                VkPipelineShaderStageCreateInfo shaderStageCI{};
                shaderStageCI.module = shaderModule;
                shaderStageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                shaderStageCI.stage = VK_SHADER_STAGE_COMPUTE_BIT;
                shaderStageCI.pName = "main";
                VkComputePipelineCreateInfo computePipelineCreateInfo{};
                computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
                computePipelineCreateInfo.layout = computeCloudPipelineLayout;
                computePipelineCreateInfo.flags = 0;
                computePipelineCreateInfo.stage = shaderStageCI;
                RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &computeCloudPipeline));
            }

            {
                auto shaderModule = RHI::ShaderManager->getShader("Cloud_ShadowMap.comp.spv", true);
                VkPipelineShaderStageCreateInfo shaderStageCI{};
                shaderStageCI.module = shaderModule;
                shaderStageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                shaderStageCI.stage = VK_SHADER_STAGE_COMPUTE_BIT;
                shaderStageCI.pName = "main";
                VkComputePipelineCreateInfo computePipelineCreateInfo{};
                computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
                computePipelineCreateInfo.layout = computeCloudPipelineLayout;
                computePipelineCreateInfo.flags = 0;
                computePipelineCreateInfo.stage = shaderStageCI;
                RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &shadowMapPipeline));
            }

            {
                CHECK(reconstructionPipeline == VK_NULL_HANDLE);

                auto shaderModule = RHI::ShaderManager->getShader("Cloud_Reconstruction.comp.spv", true);


                VkPipelineShaderStageCreateInfo shaderStageCI{};
                shaderStageCI.module = shaderModule;
                shaderStageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                shaderStageCI.stage = VK_SHADER_STAGE_COMPUTE_BIT;
                shaderStageCI.pName = "main";
                VkComputePipelineCreateInfo computePipelineCreateInfo{};
                computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
                computePipelineCreateInfo.layout = computeCloudPipelineLayout;
                computePipelineCreateInfo.flags = 0;
                computePipelineCreateInfo.stage = shaderStageCI;
                RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &reconstructionPipeline));
            }

            // Cloud composition compute.
            {
                CHECK(compositeCloudPipeline == VK_NULL_HANDLE);

                auto shaderModule = RHI::ShaderManager->getShader("VolumetricCompositeWithScreen.comp.spv", true);


                VkPipelineShaderStageCreateInfo shaderStageCI{};
                shaderStageCI.module = shaderModule;
                shaderStageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                shaderStageCI.stage = VK_SHADER_STAGE_COMPUTE_BIT;
                shaderStageCI.pName = "main";
                VkComputePipelineCreateInfo computePipelineCreateInfo{};
                computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
                computePipelineCreateInfo.layout = computeCloudPipelineLayout;
                computePipelineCreateInfo.flags = 0;
                computePipelineCreateInfo.stage = shaderStageCI;
                RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &compositeCloudPipeline));
            }
        }

        virtual void release() override
        {
            RHISafeRelease(computeCloudPipeline);
            RHISafeRelease(computeCloudPipelineLayout);

            RHISafeRelease(compositeCloudPipeline);
            RHISafeRelease(shadowMapPipeline);
            RHISafeRelease(reconstructionPipeline);

            setLayout = VK_NULL_HANDLE;
        }
    };

    void DeferredRenderer::renderVolumetricCloud(
        VkCommandBuffer cmd,
        Renderer* renderer,
        SceneTextures* inTextures,
        RenderSceneData* scene,
        BufferParamRefPointer& viewData,
        BufferParamRefPointer& frameData,
        BlueNoiseMisc& inBlueNoise)
    {
        // Skip if no directional light.
        if (scene->getImportanceLights().directionalLightCount <= 0)
        {
            return;
        }

        CVarCmdHandle(cVarUpdateCloudNoise, [&]()
        {
            StaticTexturesManager::get()->rebuildCloudTexture(cmd);
        });

        auto& gbufferTranslucentMask = inTextures->getGbufferUpscaleReactive()->getImage();


        auto& sceneColorHdr = inTextures->getHdrSceneColor()->getImage();
        auto& sceneDepthZ = inTextures->getDepth()->getImage();
        auto& gbufferA = inTextures->getGbufferA()->getImage();

        auto& basicNoise = StaticTexturesManager::get()->getCloudBasicNoise()->getImage();
        auto& detailNoise = StaticTexturesManager::get()->getCloudWorleyNoise()->getImage();

        auto weatherTexture = TextureManager::get()->getImage(EngineTextures::GCloudWeatherUUID);
        auto gradientTexture = TextureManager::get()->getImage(EngineTextures::GCloudGradientUUID);

        // Quater resolution evaluate.
        auto computeCloud = m_rtPool->createPoolImage(
            "CloudCompute",
            inTextures->getDepth()->getImage().getExtent().width  / 4,
            inTextures->getDepth()->getImage().getExtent().height / 4,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
        );
        auto computeCloudDepth = m_rtPool->createPoolImage(
            "CloudComputeDepth",
            inTextures->getDepth()->getImage().getExtent().width / 4,
            inTextures->getDepth()->getImage().getExtent().height / 4,
            VK_FORMAT_R32_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
        );

        auto* pass = getPasses()->getPass<VolumetricCloudPass>();

        VkDescriptorImageInfo hdrSceneColorImageInfo = RHIDescriptorImageInfoStorage(sceneColorHdr.getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo hdrSceneColorInfo = RHIDescriptorImageInfoSample(sceneColorHdr.getView(buildBasicImageSubresource()));

        VkDescriptorImageInfo translucentMaskInfo = RHIDescriptorImageInfoStorage(gbufferTranslucentMask.getView(buildBasicImageSubresource()));

        VkDescriptorImageInfo computeCloudImageInfo = RHIDescriptorImageInfoStorage(computeCloud->getImage().getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo computeCloudInfo = RHIDescriptorImageInfoSample(computeCloud->getImage().getView(buildBasicImageSubresource()));

        VkDescriptorImageInfo computeCloudImageInfoDepth = RHIDescriptorImageInfoStorage(computeCloudDepth->getImage().getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo computeCloudInfoDepth = RHIDescriptorImageInfoSample(computeCloudDepth->getImage().getView(buildBasicImageSubresource()));

        VkDescriptorImageInfo sceneDepthZInfo = RHIDescriptorImageInfoSample(sceneDepthZ.getView(RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT)));

        VkDescriptorImageInfo gbufferAInfo = RHIDescriptorImageInfoSample(gbufferA.getView(buildBasicImageSubresource()));

        VkDescriptorImageInfo basicNoiseInfo = RHIDescriptorImageInfoSample(basicNoise.getView(buildBasicImageSubresource(), VK_IMAGE_VIEW_TYPE_3D));
        VkDescriptorImageInfo detailNoiseInfo = RHIDescriptorImageInfoSample(detailNoise.getView(buildBasicImageSubresource(), VK_IMAGE_VIEW_TYPE_3D));

        VkDescriptorImageInfo weatherInfo = RHIDescriptorImageInfoSample(weatherTexture->getImage().getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo gradientInfo = RHIDescriptorImageInfoSample(gradientTexture->getImage().getView(buildBasicImageSubresource()));

        auto& transmittanceLut = inTextures->getAtmosphereTransmittance()->getImage();
        auto& froxelScatterLut = inTextures->getAtmosphereFroxelScatter()->getImage();
        transmittanceLut.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        froxelScatterLut.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());


        VkDescriptorImageInfo froxelScatterLutInfo = RHIDescriptorImageInfoSample(froxelScatterLut.getView(buildBasicImageSubresource(), VK_IMAGE_VIEW_TYPE_3D));
        VkDescriptorImageInfo tansmittanceLutInfo = RHIDescriptorImageInfoSample(transmittanceLut.getView(buildBasicImageSubresource()));

        auto cloudShadowMap = m_rtPool->createPoolImage(
            "CloudShadowMap",
            1u, // 768,
            1u, // 768,
            VK_FORMAT_R32_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
        );
        VkDescriptorImageInfo shadowMapImageInfo = RHIDescriptorImageInfoStorage(cloudShadowMap->getImage().getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo shadowMapColorInfo = RHIDescriptorImageInfoSample(cloudShadowMap->getImage().getView(buildBasicImageSubresource()));

        auto newCloudReconstruction = m_rtPool->createPoolImage(
            "NewCloudReconstruction",
            sceneDepthZ.getExtent().width,
            sceneDepthZ.getExtent().height,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

        auto newCloudReconstructionDepth = m_rtPool->createPoolImage(
            "NewCloudReconstructionDepth",
            sceneDepthZ.getExtent().width,
            sceneDepthZ.getExtent().height,
            VK_FORMAT_R32_SFLOAT,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

        if (!m_cloudReconstruction)
        {
            m_cloudReconstruction = m_rtPool->createPoolImage(
                "CloudReconstruction",
                sceneDepthZ.getExtent().width,
                sceneDepthZ.getExtent().height,
                VK_FORMAT_R16G16B16A16_SFLOAT,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
            m_cloudReconstruction->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        }
        if (!m_cloudReconstructionDepth)
        {
            m_cloudReconstructionDepth = m_rtPool->createPoolImage(
                "CloudReconstructionDepth",
                sceneDepthZ.getExtent().width,
                sceneDepthZ.getExtent().height,
                VK_FORMAT_R32_SFLOAT,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
            m_cloudReconstructionDepth->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        }

        VkDescriptorImageInfo reconstructImageInfo = RHIDescriptorImageInfoStorage(newCloudReconstruction->getImage().getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo reconstructInfo = RHIDescriptorImageInfoSample(newCloudReconstruction->getImage().getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo reconstructImageDepthInfo = RHIDescriptorImageInfoStorage(newCloudReconstructionDepth->getImage().getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo reconstructDepthInfo = RHIDescriptorImageInfoSample(newCloudReconstructionDepth->getImage().getView(buildBasicImageSubresource()));

        VkDescriptorImageInfo hisRecInfo = RHIDescriptorImageInfoSample(m_cloudReconstruction->getImage().getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo hisRecDepthInfo = RHIDescriptorImageInfoSample(m_cloudReconstructionDepth->getImage().getView(buildBasicImageSubresource()));

        std::vector<VkWriteDescriptorSet> writes
        {
            RHIPushWriteDescriptorSetImage(0,  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &hdrSceneColorImageInfo),
            RHIPushWriteDescriptorSetImage(1,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &hdrSceneColorInfo),
            RHIPushWriteDescriptorSetImage(2,  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &computeCloudImageInfo),
            RHIPushWriteDescriptorSetImage(3,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &computeCloudInfo),
            RHIPushWriteDescriptorSetImage(4,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &sceneDepthZInfo),
            RHIPushWriteDescriptorSetImage(5,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &gbufferAInfo),
            RHIPushWriteDescriptorSetImage(6,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &basicNoiseInfo),
            RHIPushWriteDescriptorSetImage(7,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &detailNoiseInfo),
            RHIPushWriteDescriptorSetImage(8,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &weatherInfo),
            RHIPushWriteDescriptorSetImage(9,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &gradientInfo),
            RHIPushWriteDescriptorSetImage(10,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &tansmittanceLutInfo),
            RHIPushWriteDescriptorSetImage(11,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &froxelScatterLutInfo),
            RHIPushWriteDescriptorSetImage(12,  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &shadowMapImageInfo),
            RHIPushWriteDescriptorSetImage(13,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &shadowMapColorInfo),
            RHIPushWriteDescriptorSetImage(14,  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &reconstructImageInfo),
            RHIPushWriteDescriptorSetImage(15,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &reconstructInfo),
            RHIPushWriteDescriptorSetImage(16,  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &computeCloudImageInfoDepth),
            RHIPushWriteDescriptorSetImage(17,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &computeCloudInfoDepth),
            RHIPushWriteDescriptorSetImage(18,  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &reconstructImageDepthInfo),
            RHIPushWriteDescriptorSetImage(19,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &reconstructDepthInfo),
            RHIPushWriteDescriptorSetImage(20,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &hisRecInfo),
            RHIPushWriteDescriptorSetImage(21,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &hisRecDepthInfo),
            RHIPushWriteDescriptorSetImage(22,  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &translucentMaskInfo),
            
        };

        std::vector<VkDescriptorSet> compPassSets =
        {
              viewData->buffer.getSet()
            , frameData->buffer.getSet()
            , RHI::SamplerManager->getCommonDescriptorSet()
            , inBlueNoise.getSet()
            , StaticTexturesManager::get()->globalBlueNoise.spp_1_buffer.set // 1spp is good.
        };
        // Push owner set #0.
        RHI::PushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->computeCloudPipelineLayout, 0, uint32_t(writes.size()), writes.data());

        // Set #1..3
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pass->computeCloudPipelineLayout, 1,
            (uint32_t)compPassSets.size(), compPassSets.data(),
            0, nullptr
        );

        {
            RHI::ScopePerframeMarker marker(cmd, "CloudCompute", { 1.0f, 1.0f, 0.0f, 1.0f });

            computeCloud->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            computeCloudDepth->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->computeCloudPipeline);

            vkCmdDispatch(cmd, 
                getGroupCount(computeCloud->getImage().getExtent().width, 8), 
                getGroupCount(computeCloud->getImage().getExtent().height, 8), 1);

            computeCloud->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            computeCloudDepth->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        }
        {
            RHI::ScopePerframeMarker marker(cmd, "CloudReconstruction", { 1.0f, 1.0f, 0.0f, 1.0f });

            newCloudReconstruction->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            newCloudReconstructionDepth->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->reconstructionPipeline);

            vkCmdDispatch(cmd,
                getGroupCount(newCloudReconstruction->getImage().getExtent().width, 8),
                getGroupCount(newCloudReconstruction->getImage().getExtent().height, 8), 1);

            newCloudReconstruction->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            newCloudReconstructionDepth->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        }
        {
            RHI::ScopePerframeMarker marker(cmd, "CloudComposite", { 1.0f, 1.0f, 0.0f, 1.0f });
            gbufferTranslucentMask.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            sceneColorHdr.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->compositeCloudPipeline);

            vkCmdDispatch(cmd, getGroupCount(sceneColorHdr.getExtent().width, 8), getGroupCount(sceneColorHdr.getExtent().height, 8), 1);
            sceneColorHdr.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            gbufferTranslucentMask.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        }

        m_cloudReconstruction = newCloudReconstruction;
        m_cloudReconstructionDepth = newCloudReconstructionDepth;

        m_gpuTimer.getTimeStamp(cmd, "Volumetric Cloud");
    }


    class CloudNoiseComputePass : public PassInterface
    {
    public:
        VkDescriptorSetLayout basicNoiseSetLayout = VK_NULL_HANDLE;
        VkPipeline basicNoisePipeline = VK_NULL_HANDLE;
        VkPipelineLayout basicNoisePipelineLayout = VK_NULL_HANDLE;
        
        VkDescriptorSetLayout worleyNoiseSetLayout = VK_NULL_HANDLE;
        VkPipeline worleyNoisePipeline = VK_NULL_HANDLE;
        VkPipelineLayout worleyNoisePipelineLayout = VK_NULL_HANDLE;

    public:
        virtual void init() override
        {
            

            {
                CHECK(basicNoiseSetLayout == VK_NULL_HANDLE);
                CHECK(basicNoisePipeline == VK_NULL_HANDLE);
                CHECK(basicNoisePipelineLayout == VK_NULL_HANDLE);

                // Config code.
                RHI::get()->descriptorFactoryBegin()
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0) // outImage
                    .buildNoInfoPush(basicNoiseSetLayout);

                std::vector<VkDescriptorSetLayout> setLayouts =
                {
                    basicNoiseSetLayout, // Owner setlayout.
                };
                auto shaderModule = RHI::ShaderManager->getShader("VolumetricCloudNoiseBasic.comp.spv", true);

                // Vulkan buid functions.
                VkPipelineLayoutCreateInfo plci = RHIPipelineLayoutCreateInfo();
                plci.setLayoutCount = (uint32_t)setLayouts.size();
                plci.pSetLayouts = setLayouts.data();
                basicNoisePipelineLayout = RHI::get()->createPipelineLayout(plci);
                VkPipelineShaderStageCreateInfo shaderStageCI{};
                shaderStageCI.module = shaderModule;
                shaderStageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                shaderStageCI.stage = VK_SHADER_STAGE_COMPUTE_BIT;
                shaderStageCI.pName = "main";
                VkComputePipelineCreateInfo computePipelineCreateInfo{};
                computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
                computePipelineCreateInfo.layout = basicNoisePipelineLayout;
                computePipelineCreateInfo.flags = 0;
                computePipelineCreateInfo.stage = shaderStageCI;
                RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &basicNoisePipeline));
            }
            
            {
                CHECK(worleyNoiseSetLayout == VK_NULL_HANDLE);
                CHECK(worleyNoisePipeline == VK_NULL_HANDLE);
                CHECK(worleyNoisePipelineLayout == VK_NULL_HANDLE);

                // Config code.
                RHI::get()->descriptorFactoryBegin()
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0) // outImage
                    .buildNoInfoPush(worleyNoiseSetLayout);

                std::vector<VkDescriptorSetLayout> setLayouts =
                {
                    worleyNoiseSetLayout, // Owner setlayout.
                };
                auto shaderModule = RHI::ShaderManager->getShader("VolumetricCloudNoiseWorley.comp.spv", true);

                // Vulkan buid functions.
                VkPipelineLayoutCreateInfo plci = RHIPipelineLayoutCreateInfo();
                plci.setLayoutCount = (uint32_t)setLayouts.size();
                plci.pSetLayouts = setLayouts.data();
                worleyNoisePipelineLayout = RHI::get()->createPipelineLayout(plci);
                VkPipelineShaderStageCreateInfo shaderStageCI{};
                shaderStageCI.module = shaderModule;
                shaderStageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                shaderStageCI.stage = VK_SHADER_STAGE_COMPUTE_BIT;
                shaderStageCI.pName = "main";
                VkComputePipelineCreateInfo computePipelineCreateInfo{};
                computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
                computePipelineCreateInfo.layout = worleyNoisePipelineLayout;
                computePipelineCreateInfo.flags = 0;
                computePipelineCreateInfo.stage = shaderStageCI;
                RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &worleyNoisePipeline));
            }
        }

        virtual void release() override
        {
            RHISafeRelease(basicNoisePipeline);
            RHISafeRelease(basicNoisePipelineLayout);
            basicNoiseSetLayout = VK_NULL_HANDLE;

            RHISafeRelease(worleyNoisePipeline);
            RHISafeRelease(worleyNoisePipelineLayout);
            worleyNoiseSetLayout = VK_NULL_HANDLE;
        }
    };

    void StaticTextures::initCloudTexture(VkCommandBuffer cmd)
    {
        CHECK(m_cloudBasicNoise == nullptr);
        m_cloudBasicNoise = m_rtPool->createPoolImage(
            "CloudBasicNoise",
            128u,
            128u,
            VK_FORMAT_R8_UNORM,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            1,
            128u
        );

        auto* pass = m_passCollector->getPass<CloudNoiseComputePass>();
        m_cloudBasicNoise->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
        {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->basicNoisePipeline);

            VkDescriptorImageInfo imageInfo = RHIDescriptorImageInfoStorage(m_cloudBasicNoise->getImage().getView(buildBasicImageSubresource(), VK_IMAGE_VIEW_TYPE_3D));
            std::vector<VkWriteDescriptorSet> writes
            {
                RHIPushWriteDescriptorSetImage(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &imageInfo),
            };

            // Push owner set #0.
            RHI::PushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->basicNoisePipelineLayout, 0, uint32_t(writes.size()), writes.data());

            vkCmdDispatch(cmd, getGroupCount(m_cloudBasicNoise->getImage().getExtent().width, 8), getGroupCount(m_cloudBasicNoise->getImage().getExtent().height, 8), m_cloudBasicNoise->getImage().getExtent().depth);
        }
        m_cloudBasicNoise->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

        CHECK(m_cloudWorleyNoise == nullptr);
        m_cloudWorleyNoise = m_rtPool->createPoolImage(
            "CloudWorleyNoise",
            64u,
            64u,
            VK_FORMAT_R8_UNORM,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            1,
            64u
        );
        m_cloudWorleyNoise->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
        {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->worleyNoisePipeline);

            VkDescriptorImageInfo imageInfo = RHIDescriptorImageInfoStorage(m_cloudWorleyNoise->getImage().getView(buildBasicImageSubresource(), VK_IMAGE_VIEW_TYPE_3D));
            std::vector<VkWriteDescriptorSet> writes
            {
                RHIPushWriteDescriptorSetImage(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &imageInfo),
            };

            // Push owner set #0.
            RHI::PushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->worleyNoisePipelineLayout, 0, uint32_t(writes.size()), writes.data());

            vkCmdDispatch(cmd, getGroupCount(
                m_cloudWorleyNoise->getImage().getExtent().width, 8), 
                getGroupCount(m_cloudWorleyNoise->getImage().getExtent().height, 8),
                m_cloudWorleyNoise->getImage().getExtent().depth);
        }
        m_cloudWorleyNoise->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
    }
}