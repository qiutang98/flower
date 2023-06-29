#include "../renderer_interface.h"
#include "../render_scene.h"
#include "../renderer.h"
#include "../scene_textures.h"

namespace engine
{
    struct SSSS_Push
    {
        math::vec2 direction;
        float ssss_width;
        float ssss_maxScale;
        int finalPass = 0;
    };

    struct OutlinePush
    {
        int kContourMethod;
        float kNormalDiffCoeff = 0.5f;
        float kDepthDiffCoeff = 1.0f;
    };

    class LightingPass : public PassInterface
    {
    public:
        std::unique_ptr<ComputePipeResources> pipe;
        VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;

        std::unique_ptr<ComputePipeResources> rt_hardShadow;
        VkDescriptorSetLayout rt_hardShadowSetLayout = VK_NULL_HANDLE;


        std::unique_ptr<ComputePipeResources> ssss_pipe;
        VkDescriptorSetLayout ssss_setLayout = VK_NULL_HANDLE;


        std::unique_ptr<ComputePipeResources> outline_pipe;
        VkDescriptorSetLayout outline_setLayout = VK_NULL_HANDLE;

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
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 12) // Hdr
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 13) // inSDSMShadowMask
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 14) // inSDSMShadowMask
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

            if (getContext()->getGraphicsCardState().bSupportRaytrace)
            {
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


            getContext()->descriptorFactoryBegin()
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0) // 
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1) // 
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 2) // 
                .bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 3) // 
                .buildNoInfoPush(ssss_setLayout);

            ssss_pipe = std::make_unique<ComputePipeResources>("shader/ssss_blur.comp.spv", sizeof(SSSS_Push),
                std::vector<VkDescriptorSetLayout>{
                    ssss_setLayout,
                    m_context->getSamplerCache().getCommonDescriptorSetLayout()});


            getContext()->descriptorFactoryBegin()
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0) // 
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1) // 
                .bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 2) // 
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 3) // 
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 4) //
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 5)
                .buildNoInfoPush(outline_setLayout);

            outline_pipe = std::make_unique<ComputePipeResources>("shader/outline.comp.spv", sizeof(OutlinePush),
                std::vector<VkDescriptorSetLayout>{
                outline_setLayout,
                    m_context->getSamplerCache().getCommonDescriptorSetLayout()});
        }

        virtual void release() override
        {
            pipe.reset();
            rt_hardShadow.reset();

            ssss_pipe.reset();
            outline_pipe.reset();
        }
    };

    void engine::RendererInterface::deferredLighting(
        VkCommandBuffer cmd, 
        GBufferTextures* inGBuffers, 
        RenderScene* scene, 
        BufferParameterHandle perFrameGPU, 
        PoolImageSharedRef inSDSMMask,
        AtmosphereTextures& atmosphere,
        PoolImageSharedRef inBentNormalSSAO,
        SDSMInfos& sdsmInfo)
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


        auto hdrDiffuseSSSS = rtPool->createPoolImage(
            "ssss diffuse",
            hdrSceneColor.getExtent().width,
            hdrSceneColor.getExtent().height,
            hdrSceneColor.getFormat(),
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

        hdrDiffuseSSSS->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
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
                .addUAV(hdrDiffuseSSSS)
                .addSRV(getContext()->getEngineTextureSkinLut()->getImage())
                .addSRV(getContext()->getEngineTextureSkinLutShadow()->getImage())
                .push(pass->pipe.get());

            pass->pipe->bindSet(cmd, std::vector<VkDescriptorSet>{
                m_context->getSamplerCache().getCommonDescriptorSet()
            }, 1);

            vkCmdDispatch(cmd, getGroupCount(hdrSceneColor.getExtent().width, 8), getGroupCount(hdrSceneColor.getExtent().height, 8), 1);

            m_gpuTimer.getTimeStamp(cmd, "Deferred Lighting");
        }

        hdrDiffuseSSSS->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        hdrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        {
            const auto& postProcessVolumeSetting = scene->getPostprocessVolumeSetting();

            auto tempBlur = rtPool->createPoolImage(
                "ssss temp",
                hdrSceneColor.getExtent().width,
                hdrSceneColor.getExtent().height,
                hdrSceneColor.getFormat(),
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

            SSSS_Push push{};
            push.ssss_width = postProcessVolumeSetting.ssss_width;
            push.ssss_maxScale = postProcessVolumeSetting.ssss_maxScale;

            tempBlur->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            {
                ScopePerframeMarker marker(cmd, "SSSS X", { 1.0f, 1.0f, 0.0f, 1.0f });

                push.direction = math::vec2(1.0f, 0.0f);
                pass->ssss_pipe->bindAndPushConst(cmd, &push);
                PushSetBuilder(cmd)
                    .addSRV(hdrDiffuseSSSS)
                    .addSRV(sceneDepthZ, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT))
                    .addUAV(tempBlur)
                    .addBuffer(perFrameGPU)
                    .push(pass->ssss_pipe.get());

                pass->ssss_pipe->bindSet(cmd, std::vector<VkDescriptorSet>{
                    m_context->getSamplerCache().getCommonDescriptorSet()
                }, 1);

                vkCmdDispatch(cmd, getGroupCount(hdrSceneColor.getExtent().width, 8), getGroupCount(hdrSceneColor.getExtent().height, 8), 1);

            }
            tempBlur->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            hdrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

            {
                ScopePerframeMarker marker(cmd, "SSSS Y", { 1.0f, 1.0f, 0.0f, 1.0f });

                push.direction = math::vec2(0.0f, 1.0f);
                push.finalPass = 1;
                pass->ssss_pipe->bindAndPushConst(cmd, &push);
                PushSetBuilder(cmd)
                    .addSRV(tempBlur)
                    .addSRV(sceneDepthZ, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT))
                    .addUAV(hdrSceneColor)
                    .addBuffer(perFrameGPU)
                    .push(pass->ssss_pipe.get());

                pass->ssss_pipe->bindSet(cmd, std::vector<VkDescriptorSet>{
                    m_context->getSamplerCache().getCommonDescriptorSet()
                }, 1);

                vkCmdDispatch(cmd, getGroupCount(hdrSceneColor.getExtent().width, 8), getGroupCount(hdrSceneColor.getExtent().height, 8), 1);

            }

            m_gpuTimer.getTimeStamp(cmd, "SSSS");
        }

        bool bPostOutline = false;
        if (bPostOutline)
        {
            gbufferA.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            gbufferB.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            hdrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            {
                ScopePerframeMarker marker(cmd, "Outline PP", { 1.0f, 1.0f, 0.0f, 1.0f });

                OutlinePush push{};
                push.kContourMethod = 2;
                push.kDepthDiffCoeff = 1.0f;
                push.kNormalDiffCoeff = 0.5f;

                pass->outline_pipe->bindAndPushConst(cmd, &push);
                PushSetBuilder(cmd)
                    .addSRV(gbufferB)
                    .addUAV(hdrSceneColor)
                    .addBuffer(perFrameGPU)
                    .addSRV(gbufferA)
                    .addSRV(sceneDepthZ, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT))
                    .addBuffer(sdsmInfo.rangeBuffer)
                    .push(pass->outline_pipe.get());

                pass->outline_pipe->bindSet(cmd, std::vector<VkDescriptorSet>{
                    m_context->getSamplerCache().getCommonDescriptorSet()
                }, 1);

                vkCmdDispatch(cmd, getGroupCount(hdrSceneColor.getExtent().width, 8), getGroupCount(hdrSceneColor.getExtent().height, 8), 1);

            }
            hdrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        }


    }
}