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

        std::unique_ptr<ComputePipeResources> rt_hardShadow;
        VkDescriptorSetLayout rt_hardShadowSetLayout = VK_NULL_HANDLE;

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
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 11) // inSDSMShadowMask
                .buildNoInfoPush(setLayout);

            pipe = std::make_unique<ComputePipeResources>("shader/deferred_lighting.comp.spv", 0, 
                std::vector<VkDescriptorSetLayout>{ setLayout, m_context->getSamplerCache().getCommonDescriptorSetLayout() });

            getContext()->descriptorFactoryBegin()
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0) // shadow
                .bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1) // inFrameData
                .bindNoInfo(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, VK_SHADER_STAGE_COMPUTE_BIT, 2) // AS
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 3) // inDepth
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 4)
                .buildNoInfoPush(rt_hardShadowSetLayout);

            rt_hardShadow = std::make_unique<ComputePipeResources>("shader/rt_hard_shadow.comp.spv", 0,
                std::vector<VkDescriptorSetLayout>{ 
                    rt_hardShadowSetLayout, 
                         m_context->getBindlessSSBOSetLayout()
                        , m_context->getBindlessSSBOSetLayout()
                        , m_context->getBindlessTextureSetLayout()
                        , m_context->getBindlessSamplerSetLayout(),
                    m_context->getSamplerCache().getCommonDescriptorSetLayout(),
                    getRenderer()->getBlueNoise().spp_1_buffer.setLayouts});
        }

        virtual void release() override
        {
            pipe.reset();
            rt_hardShadow.reset();
        }
    };

    void engine::RendererInterface::deferredLighting(
        VkCommandBuffer cmd, 
        GBufferTextures* inGBuffers, 
        RenderScene* scene, 
        BufferParameterHandle perFrameGPU, 
        PoolImageSharedRef inSDSMMask,
        AtmosphereTextures& atmosphere,
        PoolImageSharedRef inBentNormalSSAO)
    {
        auto& hdrSceneColor = inGBuffers->hdrSceneColor->getImage();
        auto& gbufferA = inGBuffers->gbufferA->getImage();
        auto& gbufferB = inGBuffers->gbufferB->getImage();
        auto& gbufferS = inGBuffers->gbufferS->getImage();
        auto& sceneDepthZ = inGBuffers->depthTexture->getImage();



        sceneDepthZ.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));

        auto* rtPool = &m_context->getRenderTargetPools();
        auto* pass = getContext()->getPasses().get<LightingPass>();

        // 
        PoolImageSharedRef rt_shadow = nullptr;
        const auto& gpuInfo = scene->getSkyGPU();
        const bool bRTshadow = gpuInfo.rayTraceShadow != 0;;
        if (scene->isASValid() && bRTshadow)
        {
            rt_shadow = rtPool->createPoolImage(
                "rt_shadow",
                sceneDepthZ.getExtent().width,
                sceneDepthZ.getExtent().height,
                VK_FORMAT_R8_UNORM,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

            rt_shadow->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            ScopePerframeMarker marker(cmd, "rt hard shadow", { 1.0f, 1.0f, 0.0f, 1.0f });

            pass->rt_hardShadow->bind(cmd);
            PushSetBuilder(cmd)
                .addUAV(rt_shadow)
                .addBuffer(perFrameGPU)
                .addAS(scene->getAS())
                .addSRV(sceneDepthZ, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT))
                .addBuffer(scene->getStaticMeshObjectsGPU())
                .push(pass->rt_hardShadow.get());

            pass->rt_hardShadow->bindSet(cmd, std::vector<VkDescriptorSet>{
                m_context->getBindlessSSBOSet()
                    , m_context->getBindlessSSBOSet()
                    , m_context->getBindlessTextureSet()
                    , m_context->getBindlessSamplerSet(),
                m_context->getSamplerCache().getCommonDescriptorSet(),
                getRenderer()->getBlueNoise().spp_1_buffer.set
            }, 1);

            vkCmdDispatch(cmd, getGroupCount(hdrSceneColor.getExtent().width, 8), getGroupCount(hdrSceneColor.getExtent().height, 8), 1);

            m_gpuTimer.getTimeStamp(cmd, "rt hard shadow");

            rt_shadow->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        }

        gbufferA.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        gbufferB.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        gbufferS.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        hdrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
        VulkanImage* sdsmMask = &m_context->getEngineTextureWhite()->getImage();
        if (scene->shouldRenderSDSM())
        {
            sdsmMask = &inSDSMMask->getImage();
            sdsmMask->transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        }

        {
            ScopePerframeMarker marker(cmd, "Deferred Lighting", { 1.0f, 1.0f, 0.0f, 1.0f });

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
                .addSRV(inBentNormalSSAO)
                .addSRV(m_skylightRadiance, buildBasicImageSubresourceCube(), VK_IMAGE_VIEW_TYPE_CUBE)
                .addSRV(rt_shadow ? rt_shadow->getImage() : *sdsmMask)
                .push(pass->pipe.get());

            pass->pipe->bindSet(cmd, std::vector<VkDescriptorSet>{
                m_context->getSamplerCache().getCommonDescriptorSet()
            }, 1);

            vkCmdDispatch(cmd, getGroupCount(hdrSceneColor.getExtent().width, 8), getGroupCount(hdrSceneColor.getExtent().height, 8), 1);

            m_gpuTimer.getTimeStamp(cmd, "Deferred Lighting");
        }
    }
}