#include "../renderer_interface.h"
#include "../render_scene.h"
#include "../renderer.h"
#include "../scene_textures.h"

namespace engine
{
    class LightingPass : public PassInterface
    {
    public:
        std::unique_ptr<ComputePipeResources> pipe;
        VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;

    public:
        virtual void onInit() override
        {
            getContext()->descriptorFactoryBegin()
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0) // Hdr
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1) // inDepth
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 2) // inGbufferA
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 3) // inGbufferB
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 4) // inGbufferS
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 5) // inSDSMShadowMask
                .bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 6) // inFrameData
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 7) // inBRDFLut
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 8) // inTransmittanceLut
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 9) // inGTAO
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 10) // inSkylight
                .buildNoInfoPush(setLayout);

            pipe = std::make_unique<ComputePipeResources>("shader/deferred_lighting.comp.spv", 0, 
                std::vector<VkDescriptorSetLayout>{ setLayout, m_context->getSamplerCache().getCommonDescriptorSetLayout() });
        }

        virtual void release() override
        {
            pipe.reset();
        }
    };

    void engine::RendererInterface::deferredLighting(
        VkCommandBuffer cmd, 
        GBufferTextures* inGBuffers, 
        RenderScene* scene, 
        BufferParameterHandle perFrameGPU, 
        PoolImageSharedRef inSDSMMask,
        AtmosphereTextures& atmosphere,
        PoolImageSharedRef inGTAO)
    {
        auto& hdrSceneColor = inGBuffers->hdrSceneColor->getImage();
        auto& gbufferA = inGBuffers->gbufferA->getImage();
        auto& gbufferB = inGBuffers->gbufferB->getImage();
        auto& gbufferS = inGBuffers->gbufferS->getImage();
        auto& sceneDepthZ = inGBuffers->depthTexture->getImage();

        hdrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
        gbufferA.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        gbufferB.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        gbufferS.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        sceneDepthZ.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));

        VulkanImage* sdsmMask = &m_context->getEngineTextureWhite()->getImage();
        if (scene->shouldRenderSDSM())
        {
            sdsmMask = &inSDSMMask->getImage();
            sdsmMask->transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        }

        {
            ScopePerframeMarker marker(cmd, "Deferred Lighting", { 1.0f, 1.0f, 0.0f, 1.0f });
            auto* pass = getContext()->getPasses().get<LightingPass>();
            pass->pipe->bind(cmd);
            PushSetBuilder(cmd)
                .addUAV(hdrSceneColor)
                .addSRV(sceneDepthZ, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT))
                .addSRV(gbufferA)
                .addSRV(gbufferB)
                .addSRV(gbufferS)
                .addSRV(*sdsmMask)
                .addBuffer(perFrameGPU)
                .addSRV(*m_renderer->getSharedTextures().brdfLut)
                .addSRV(atmosphere.transmittance ? atmosphere.transmittance : inGBuffers->gbufferA)
                .addSRV(inGTAO)
                .addSRV(m_skylightRadiance, buildBasicImageSubresourceCube(), VK_IMAGE_VIEW_TYPE_CUBE)
                .push(pass->pipe.get());

            pass->pipe->bindSet(cmd, std::vector<VkDescriptorSet>{
                m_context->getSamplerCache().getCommonDescriptorSet()
            }, 1);

            vkCmdDispatch(cmd, getGroupCount(hdrSceneColor.getExtent().width, 8), getGroupCount(hdrSceneColor.getExtent().height, 8), 1);

            m_gpuTimer.getTimeStamp(cmd, "Deferred Lighting");
        }
    }
}