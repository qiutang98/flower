#include "Pch.h"
#include "../DeferredRenderer.h"
#include "../../Renderer.h"
#include "../../RenderSceneData.h"
#include "../../SceneTextures.h"
#include "../../RenderSettingContext.h"

namespace Flower
{
    struct BasicLightingPushConst
    {
        uint32_t directionalLightValid;
    };

    class BasicLightingPass : public PassInterface
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
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0) // Hdr
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1) // inDepth
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 2) // inGbufferA
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 3) // inGbufferB
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 4) // inGbufferS
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 5) // inSDSMShadowMask
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 6) // inBRDFLut
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 7) // inTransmittance
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 8) // inMultiScatter
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 9) // inSkyViewLut
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 10) // inEnvCube
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 11) // inGlobalIrradiance
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 12) // inGlobalPrefilter
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 13) // inGTAO
                .buildNoInfoPush(setLayout);

            std::vector<VkDescriptorSetLayout> setLayouts =
            {
                  setLayout // Owner setlayout.
                , GetLayoutStatic(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) // viewData
                , GetLayoutStatic(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) // frameData
                , RHI::SamplerManager->getCommonDescriptorSetLayout()
            };
            auto shaderModule = RHI::ShaderManager->getShader("BasicLighting.comp.spv", true);

            // Vulkan buid functions.
            VkPipelineLayoutCreateInfo plci = RHIPipelineLayoutCreateInfo();
            VkPushConstantRange pushConstant{};
            pushConstant.offset = 0;
            pushConstant.size = sizeof(BasicLightingPushConst);
            pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            plci.pPushConstantRanges = &pushConstant;
            plci.pushConstantRangeCount = 1;

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

    void DeferredRenderer::renderBasicLighting(
        VkCommandBuffer cmd,
        Renderer* renderer,
        SceneTextures* inTextures,
        RenderSceneData* scene,
        BufferParamRefPointer& viewData,
        BufferParamRefPointer& frameData,
        PoolImageSharedRef inGTAO)
    {
        auto& hdrSceneColor = inTextures->getHdrSceneColor()->getImage();
        auto& gbufferA = inTextures->getGbufferA()->getImage();
        auto& gbufferB = inTextures->getGbufferB()->getImage();
        auto& gbufferS = inTextures->getGbufferS()->getImage();
        auto& sceneDepthZ = inTextures->getDepth()->getImage();
        auto& atmosphereEnvCubeImage = inTextures->getAtmosphereEnvCapture()->getImage();
        
        {
            hdrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            gbufferA.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            gbufferB.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            gbufferS.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            sceneDepthZ.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));

            atmosphereEnvCubeImage.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresourceCube());
        }

        VkImageView globalIrradianceView = atmosphereEnvCubeImage.getView(buildBasicImageSubresourceCube(), VK_IMAGE_VIEW_TYPE_CUBE);
        VkImageView globalPrefilterView = atmosphereEnvCubeImage.getView(buildBasicImageSubresourceCube(), VK_IMAGE_VIEW_TYPE_CUBE);
        if (StaticTexturesManager::get()->isIBLReady())
        {
            globalIrradianceView = StaticTexturesManager::get()->getIBLIrradiance()->getImage().getView(buildBasicImageSubresourceCube(), VK_IMAGE_VIEW_TYPE_CUBE);

            globalPrefilterView = StaticTexturesManager::get()->getIBLPrefilter()->getImage().getView(buildBasicImageSubresourceCube(), VK_IMAGE_VIEW_TYPE_CUBE);
        }

        const bool bExistDirectionalLight = scene->getImportanceLights().directionalLightCount > 0;

        VkImageView transmittanceLutView = gbufferA.getView(RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
        VkImageView multiScatterLutView = gbufferA.getView(RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
        VkImageView skyViewLutView = gbufferA.getView(RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
        if (bExistDirectionalLight)
        {
            auto& transmittanceLut = inTextures->getAtmosphereTransmittance()->getImage();
            auto& multiScatterLut = inTextures->getAtmosphereMultiScatter()->getImage();
            auto& skyviewLut = inTextures->getAtmosphereSkyView()->getImage();

            transmittanceLut.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            multiScatterLut.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            skyviewLut.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

            transmittanceLutView = transmittanceLut.getView(RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
            multiScatterLutView = multiScatterLut.getView(RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
            skyViewLutView = skyviewLut.getView(RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
        }

        const bool bExistDirectionalLightSDSM = 
            scene->getImportanceLights().directionalLightCount > 0 &&
            scene->getCollectStaticMeshes().size() > 0;

        VkImageView sdsmShadowMaskView = gbufferA.getView(RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));

        if (bExistDirectionalLightSDSM)
        {
            auto& sdsmShadowMask = inTextures->getSDSMShadowMask()->getImage();
            sdsmShadowMask.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
            sdsmShadowMaskView = sdsmShadowMask.getView(RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
        }

        {
            RHI::ScopePerframeMarker tonemapperMarker(cmd, "BasicLighting", { 1.0f, 1.0f, 0.0f, 1.0f });

            auto* pass = getPasses()->getPass<BasicLightingPass>();

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->pipeline);

            BasicLightingPushConst gpuPushConstant =
            {
                .directionalLightValid = bExistDirectionalLightSDSM ? 1u : 0u,
            };
            vkCmdPushConstants(cmd, pass->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(gpuPushConstant), &gpuPushConstant);


            VkDescriptorImageInfo hdrImageInfo = RHIDescriptorImageInfoStorage(hdrSceneColor.getView(buildBasicImageSubresource()));
            VkDescriptorImageInfo depthInfo = RHIDescriptorImageInfoSample(sceneDepthZ.getView(RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT)));
            VkDescriptorImageInfo gbufferAInfo = RHIDescriptorImageInfoSample(gbufferA.getView(buildBasicImageSubresource()));
            VkDescriptorImageInfo gbufferBInfo = RHIDescriptorImageInfoSample(gbufferB.getView(buildBasicImageSubresource()));
            VkDescriptorImageInfo gbufferSInfo = RHIDescriptorImageInfoSample(gbufferS.getView(buildBasicImageSubresource()));
            VkDescriptorImageInfo sdsmShadowMask = RHIDescriptorImageInfoSample(sdsmShadowMaskView);
            VkDescriptorImageInfo brdfLutInfo = RHIDescriptorImageInfoSample(StaticTexturesManager::get()->getBRDFLut()->getImage().getView(buildBasicImageSubresource()));
            VkDescriptorImageInfo transmittanceLutInfo = RHIDescriptorImageInfoSample(transmittanceLutView);
            VkDescriptorImageInfo multiScatterLutInfo = RHIDescriptorImageInfoSample(multiScatterLutView);
            VkDescriptorImageInfo skyviewLutInfo = RHIDescriptorImageInfoSample(skyViewLutView);
            VkDescriptorImageInfo envCubeInfo = RHIDescriptorImageInfoSample(atmosphereEnvCubeImage.getView(buildBasicImageSubresourceCube(), VK_IMAGE_VIEW_TYPE_CUBE));
            VkDescriptorImageInfo globalIrradianceInfo = RHIDescriptorImageInfoSample(globalIrradianceView);
            VkDescriptorImageInfo globalPrefilterInfo = RHIDescriptorImageInfoSample(globalPrefilterView);
            VkDescriptorImageInfo gtaoInfo = RHIDescriptorImageInfoSample(inGTAO->getImage().getView(buildBasicImageSubresource()));
            
            std::vector<VkWriteDescriptorSet> writes
            {
                RHIPushWriteDescriptorSetImage(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &hdrImageInfo),
                RHIPushWriteDescriptorSetImage(1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &depthInfo),
                RHIPushWriteDescriptorSetImage(2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &gbufferAInfo),
                RHIPushWriteDescriptorSetImage(3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &gbufferBInfo),
                RHIPushWriteDescriptorSetImage(4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &gbufferSInfo),
                RHIPushWriteDescriptorSetImage(5, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &sdsmShadowMask),
                RHIPushWriteDescriptorSetImage(6, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &brdfLutInfo),
                RHIPushWriteDescriptorSetImage(7, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &transmittanceLutInfo),
                RHIPushWriteDescriptorSetImage(8, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &multiScatterLutInfo),
                RHIPushWriteDescriptorSetImage(9, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &skyviewLutInfo),
                RHIPushWriteDescriptorSetImage(10, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &envCubeInfo),
                RHIPushWriteDescriptorSetImage(11, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &globalIrradianceInfo),
                RHIPushWriteDescriptorSetImage(12, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &globalPrefilterInfo),
                RHIPushWriteDescriptorSetImage(13, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &gtaoInfo),
            };

            // Push owner set #0.
            RHI::PushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->pipelineLayout, 0, uint32_t(writes.size()), writes.data());

            std::vector<VkDescriptorSet> passSets =
            {
                  viewData->buffer.getSet()  // viewData
                , frameData->buffer.getSet() // frameData
                , RHI::SamplerManager->getCommonDescriptorSet() // samplers.
            };
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->pipelineLayout,
                1, (uint32_t)passSets.size(), passSets.data(), 0, nullptr);

            vkCmdDispatch(cmd, getGroupCount(hdrSceneColor.getExtent().width, 8), getGroupCount(hdrSceneColor.getExtent().height, 8), 1);

            m_gpuTimer.getTimeStamp(cmd, "BasicLighting");
        }
    }
   
}