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

        VkPipeline compositeCloudPipeline = VK_NULL_HANDLE;
        VkPipelineLayout compositeCloudPipelineLayout = VK_NULL_HANDLE;
        

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
                .buildNoInfoPush(setLayout);

            std::vector<VkDescriptorSetLayout> setLayouts =
            {
                  setLayout // Owner layout.
                , GetLayoutStatic(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)  // viewData
                , GetLayoutStatic(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)  // frameData
                , RHI::SamplerManager->getCommonDescriptorSetLayout() // Common samplers
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

            // Cloud composition compute.
            {
                CHECK(compositeCloudPipeline == VK_NULL_HANDLE);
                CHECK(compositeCloudPipelineLayout == VK_NULL_HANDLE);

                auto shaderModule = RHI::ShaderManager->getShader("VolumetricCompositeWithScreen.comp.spv", true);

                // Vulkan build functions.
                VkPipelineLayoutCreateInfo plci = RHIPipelineLayoutCreateInfo();
                plci.setLayoutCount = (uint32_t)setLayouts.size();
                plci.pSetLayouts = setLayouts.data();
                compositeCloudPipelineLayout = RHI::get()->createPipelineLayout(plci);

                VkPipelineShaderStageCreateInfo shaderStageCI{};
                shaderStageCI.module = shaderModule;
                shaderStageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                shaderStageCI.stage = VK_SHADER_STAGE_COMPUTE_BIT;
                shaderStageCI.pName = "main";
                VkComputePipelineCreateInfo computePipelineCreateInfo{};
                computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
                computePipelineCreateInfo.layout = compositeCloudPipelineLayout;
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
            RHISafeRelease(compositeCloudPipelineLayout);

            setLayout = VK_NULL_HANDLE;
        }
    };

    void DeferredRenderer::renderVolumetricCloud(
        VkCommandBuffer cmd,
        Renderer* renderer,
        SceneTextures* inTextures,
        RenderSceneData* scene,
        BufferParamRefPointer& viewData,
        BufferParamRefPointer& frameData)
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

        auto& sceneColorHdr = inTextures->getHdrSceneColor()->getImage();
        auto& computeCloud = inTextures->getCloudImage()->getImage();
        auto& sceneDepthZ = inTextures->getDepth()->getImage();
        auto& gbufferA = inTextures->getGbufferA()->getImage();

        auto& basicNoise = StaticTexturesManager::get()->getCloudBasicNoise()->getImage();
        auto& detailNoise = StaticTexturesManager::get()->getCloudWorleyNoise()->getImage();

        auto weatherTexture = TextureManager::get()->getImage(EngineTextures::GCloudWeatherUUID);
        auto gradientTexture = TextureManager::get()->getImage(EngineTextures::GCloudGradientUUID);

        auto* pass = getPasses()->getPass<VolumetricCloudPass>();

        VkDescriptorImageInfo hdrSceneColorImageInfo = RHIDescriptorImageInfoStorage(sceneColorHdr.getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo hdrSceneColorInfo = RHIDescriptorImageInfoSample(sceneColorHdr.getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo computeCloudImageInfo = RHIDescriptorImageInfoStorage(computeCloud.getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo computeCloudInfo = RHIDescriptorImageInfoSample(computeCloud.getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo sceneDepthZInfo = RHIDescriptorImageInfoSample(sceneDepthZ.getView(RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT)));
        VkDescriptorImageInfo gbufferAInfo = RHIDescriptorImageInfoSample(gbufferA.getView(buildBasicImageSubresource()));

        VkDescriptorImageInfo basicNoiseInfo = RHIDescriptorImageInfoSample(basicNoise.getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo detailNoiseInfo = RHIDescriptorImageInfoSample(detailNoise.getView(buildBasicImageSubresource(), VK_IMAGE_VIEW_TYPE_3D));

        VkDescriptorImageInfo weatherInfo = RHIDescriptorImageInfoSample(weatherTexture->getImage().getView(buildBasicImageSubresource()));
        VkDescriptorImageInfo gradientInfo = RHIDescriptorImageInfoSample(gradientTexture->getImage().getView(buildBasicImageSubresource()));

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
        };

        std::vector<VkDescriptorSet> compPassSets =
        {
              viewData->buffer.getSet()
            , frameData->buffer.getSet()
            , RHI::SamplerManager->getCommonDescriptorSet()
        };
        {
            RHI::ScopePerframeMarker marker(cmd, "CloudCompute", { 1.0f, 1.0f, 0.0f, 1.0f });

            computeCloud.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->computeCloudPipeline);

            // Push owner set #0.
            RHI::PushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->computeCloudPipelineLayout, 0, uint32_t(writes.size()), writes.data());

            // Set #1..3
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                pass->computeCloudPipelineLayout, 1,
                (uint32_t)compPassSets.size(), compPassSets.data(),
                0, nullptr
            );

            vkCmdDispatch(cmd, getGroupCount(computeCloud.getExtent().width, 8), getGroupCount(computeCloud.getExtent().height, 8), 1);
            computeCloud.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        }
        {
            RHI::ScopePerframeMarker marker(cmd, "CloudComposite", { 1.0f, 1.0f, 0.0f, 1.0f });

            sceneColorHdr.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->compositeCloudPipeline);

            // Push owner set #0.
            RHI::PushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->compositeCloudPipelineLayout, 0, uint32_t(writes.size()), writes.data());

            // Set #1..3
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                pass->compositeCloudPipelineLayout, 1,
                (uint32_t)compPassSets.size(), compPassSets.data(),
                0, nullptr
            );

            vkCmdDispatch(cmd, getGroupCount(sceneColorHdr.getExtent().width, 8), getGroupCount(sceneColorHdr.getExtent().height, 8), 1);
            sceneColorHdr.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        }

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
            1024u,
            1024u,
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
        );

        auto* pass = m_passCollector->getPass<CloudNoiseComputePass>();
        m_cloudBasicNoise->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
        {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->basicNoisePipeline);

            VkDescriptorImageInfo imageInfo = RHIDescriptorImageInfoStorage(m_cloudBasicNoise->getImage().getView(buildBasicImageSubresource()));
            std::vector<VkWriteDescriptorSet> writes
            {
                RHIPushWriteDescriptorSetImage(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &imageInfo),
            };

            // Push owner set #0.
            RHI::PushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->basicNoisePipelineLayout, 0, uint32_t(writes.size()), writes.data());

            vkCmdDispatch(cmd, getGroupCount(m_cloudBasicNoise->getImage().getExtent().width, 8), getGroupCount(m_cloudBasicNoise->getImage().getExtent().height, 8), 1);
        }
        m_cloudBasicNoise->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

        CHECK(m_cloudWorleyNoise == nullptr);
        m_cloudWorleyNoise = m_rtPool->createPoolImage(
            "CloudWorleyNoise",
            32u,
            32u,
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            1,
            32u
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