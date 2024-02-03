#include "../deferred_renderer.h"
#include "../render_scene.h"
#include "../renderer.h"
#include "../scene_textures.h"

namespace engine
{
    class LightingPass : public PassInterface
    {
    public:
        std::unique_ptr<ComputePipeResources> pipe;

    public:
        virtual void onInit() override
        {
            VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
            getContext()->descriptorFactoryBegin()
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 0) // Hdr
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1) // inDepth
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 2) // inGbufferA
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 3) // inGbufferB
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 4) // inGbufferS
                .bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,5) // inFrameData
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 6) // inBRDFLut
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 7) // inTransmittanceLut
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 8) // inSunShadowMask
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 9) // inSunShadowMask
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,10)
                .buildNoInfoPush(setLayout);

            pipe = std::make_unique<ComputePipeResources>("shader/direct_lighting.glsl", 0,
                std::vector<VkDescriptorSetLayout>{ 
                setLayout, 
                m_context->getSamplerCache().getCommonDescriptorSetLayout() });
        }

        virtual void release() override
        {
            pipe.reset();
        }
    };

    void engine::renderDirectLighting(
        VkCommandBuffer cmd,
        GBufferTextures* inGBuffers,
        RenderScene* scene,
        BufferParameterHandle perFrameGPU,
        AtmosphereTextures& atmosphere,
        SDSMInfos& sunSdsmInfos,
        SDSMInfos& moonSdsmInfos,
        PoolImageSharedRef bentNormalSSAO,
        GPUTimestamps* timer,
        PoolImageSharedRef exposure)
    {
        auto* rtPool = &getContext()->getRenderTargetPools();
        auto* pass = getContext()->getPasses().get<LightingPass>();

        auto& hdrSceneColor = inGBuffers->hdrSceneColor->getImage();
        auto& gbufferA = inGBuffers->gbufferA->getImage();
        auto& gbufferB = inGBuffers->gbufferB->getImage();
        auto& gbufferS = inGBuffers->gbufferS->getImage();
        auto& sceneDepthZ = inGBuffers->depthTexture->getImage();


        sceneDepthZ.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));
        gbufferA.transitionLayout(cmd, 
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        gbufferB.transitionLayout(cmd, 
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        gbufferS.transitionLayout(cmd, 
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        hdrSceneColor.transitionLayout(cmd, 
            VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

        sunSdsmInfos.shadowMask->getImage().transitionShaderReadOnly(cmd);

        {
            ScopePerframeMarker marker(cmd, "Deferred Lighting", { 1.0f, 1.0f, 0.0f, 1.0f }, timer);

            pass->pipe->bind(cmd);
            PushSetBuilder(cmd)
                .addUAV(hdrSceneColor)
                .addSRV(sceneDepthZ, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT))
                .addSRV(gbufferA)
                .addSRV(gbufferB)
                .addSRV(gbufferS)
                .addBuffer(perFrameGPU)
                .addSRV(*getRenderer()->getSharedTextures().brdfLut)
                .addSRV(atmosphere.transmittance ? 
                    atmosphere.transmittance->getImage() : 
                    getContext()->getBuiltinTextureWhite()->getSelfImage())
                .addSRV(sunSdsmInfos.shadowMask)
                .addSRV(bentNormalSSAO ? bentNormalSSAO->getImage()
                    : getContext()->getBuiltinTextureWhite()->getSelfImage())
                .addSRV(exposure ? exposure->getImage() : getContext()->getBuiltinTextureWhite()->getSelfImage())
                .push(pass->pipe.get());

            pass->pipe->bindSet(cmd, std::vector<VkDescriptorSet>{ getContext()->getSamplerCache().getCommonDescriptorSet() }, 1);

            vkCmdDispatch(cmd, 
                getGroupCount(hdrSceneColor.getExtent().width, 8), 
                getGroupCount(hdrSceneColor.getExtent().height, 8), 1);
        }
    }

}