#include "../renderer_interface.h"
#include "../render_scene.h"
#include "../renderer.h"
#include "../scene_textures.h"

namespace engine
{
    struct GpuSsgiIntersectPush
    {
        float uvRadius =    0.1f;
        uint32_t sliceCount = 2;
        float falloffMul = 1.0;
        float falloffAdd = 0.0f;
        uint32_t stepCount = 6;
        float intensity = 1.0f;
        float power = 1.0f;
    };

    class SSGIPass : public PassInterface
    {
    public:
        VkDescriptorSetLayout intersectLayout = VK_NULL_HANDLE;
        std::unique_ptr<ComputePipeResources> intersect;

    public:
        virtual void onInit() override
        {
            {
                getContext()->descriptorFactoryBegin()
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0) // inHistoryHdr
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1) // inHistoryHdr
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 2) // inHistoryHdr
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 3) // inHistoryHdr
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 4) // inHistoryHdr
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 5) // inHistoryHdr
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 6) // uniform
                    .buildNoInfoPush(intersectLayout);

                std::vector<VkDescriptorSetLayout> intersectLayouts = {
                    intersectLayout,
                    getContext()->getSamplerCache().getCommonDescriptorSetLayout(),
                    getRenderer()->getBlueNoise().spp_1_buffer.setLayouts
                };

                intersect = std::make_unique<ComputePipeResources>("shader/ssgi_intersect.comp.spv", (uint32_t)sizeof(GpuSsgiIntersectPush), intersectLayouts);
            }
            // Config code.
        }

        virtual void release() override
        {
            intersect.reset();
        }
    };


    PoolImageSharedRef RendererInterface::renderSSGI(
        VkCommandBuffer cmd,
        class GBufferTextures* inGBuffers,
        class RenderScene* scene,
        BufferParameterHandle perFrameGPU,
        PoolImageSharedRef inHiz)
    {
        auto* pass = getContext()->getPasses().get<SSGIPass>();
        auto* rtPool = &m_context->getRenderTargetPools();

        auto gbufferB = inGBuffers->gbufferB;
        auto historyColor = m_prevHDR ? m_prevHDR : inGBuffers->hdrSceneColor;

        auto& gbufferA = inGBuffers->gbufferA->getImage();
        auto& gbufferS = inGBuffers->gbufferS->getImage();
        auto& gbufferV = inGBuffers->gbufferV->getImage();
        auto& sceneDepthZ = inGBuffers->depthTexture->getImage();

        gbufferA.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        gbufferB->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        gbufferS.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        gbufferV.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        sceneDepthZ.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));
        inHiz->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

        PoolImageSharedRef ssgiIntersectResultBentNormal = rtPool->createPoolImage(
            "SSGI Intersect bent normal",
            gbufferB->getImage().getExtent().width,
            gbufferB->getImage().getExtent().height,
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);


        const auto& postProcessVolumeSetting = scene->getPostprocessVolumeSetting();
        {
            ScopePerframeMarker tonemapperMarker(cmd, "SSGI-Intersect", { 1.0f, 1.0f, 0.0f, 1.0f });

            gbufferB->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            historyColor->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            ssgiIntersectResultBentNormal->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());



            GpuSsgiIntersectPush intersectPush
            {
                .sliceCount = (uint32_t)postProcessVolumeSetting.ssgiSliceCount,
                .stepCount = (uint32_t)postProcessVolumeSetting.ssgiStepCount,
                .intensity = postProcessVolumeSetting.ssgiIntensity,
                .power = postProcessVolumeSetting.ssgiPower,
            };

            const float viewRadius = postProcessVolumeSetting.ssgiViewRadius;
            const float falloff = postProcessVolumeSetting.ssgiFalloff;

            intersectPush.uvRadius = viewRadius * 0.5f * math::max(m_cacheGPUPerFrameData.camProj[0][0], m_cacheGPUPerFrameData.camProj[1][1]);

            float falloffRange = viewRadius * falloff;
            float falloffFrom = viewRadius * (1.0f - falloff);

            intersectPush.falloffMul = -1.0f / falloffRange;
            intersectPush.falloffAdd = falloffFrom / falloffRange + 1.0f;


            pass->intersect->bindAndPushConst(cmd, &intersectPush);
            PushSetBuilder(cmd)
                .addSRV(inHiz)
                .addSRV(inGBuffers->depthTexture, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT))
                .addSRV(inGBuffers->gbufferA)
                .addSRV(inGBuffers->gbufferB)
                .addUAV(ssgiIntersectResultBentNormal)
                .addSRV(historyColor)
                .addBuffer(perFrameGPU)
                .push(pass->intersect.get());

            pass->intersect->bindSet(cmd, std::vector<VkDescriptorSet>{
                m_context->getSamplerCache().getCommonDescriptorSet()
              , m_renderer->getBlueNoise().spp_1_buffer.set
            }, 1);

            vkCmdDispatch(cmd, 
                getGroupCount(ssgiIntersectResultBentNormal->getImage().getExtent().width, 8), 
                getGroupCount(ssgiIntersectResultBentNormal->getImage().getExtent().height, 8), 1);
            ssgiIntersectResultBentNormal->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        }

        m_gpuTimer.getTimeStamp(cmd, "SSGI");

        return ssgiIntersectResultBentNormal;
    }
}