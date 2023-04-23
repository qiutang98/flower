#include "../renderer_interface.h"
#include "../render_scene.h"
#include "../renderer.h"
#include "../scene_textures.h"

namespace engine
{
    struct GpuSsgiIntersectPush
    {
        float pad0;
    };

    class SSGIPass : public PassInterface
    {
    public:
        VkDescriptorSetLayout intersectLayout = VK_NULL_HANDLE;
        std::unique_ptr<ComputePipeResources> intersect;

    public:
        virtual void onInit() override
        {
            // Config code.
            getContext()->descriptorFactoryBegin()
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0) // inHistoryHdr
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1) // outIntersect
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 2) // inGBufferB
                .bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 3) // uniform
                .buildNoInfoPush(intersectLayout);

            std::vector<VkDescriptorSetLayout> intersectLayouts = {
                intersectLayout,
                getContext()->getSamplerCache().getCommonDescriptorSetLayout(),
                getRenderer()->getBlueNoise().spp_1_buffer.setLayouts
            };

            intersect = std::make_unique<ComputePipeResources>("shader/ssgi_intersect.comp.spv", (uint32_t)sizeof(GpuSsgiIntersectPush), intersectLayouts);
        }

        virtual void release() override
        {
            intersect.reset();
        }
    };


    void RendererInterface::renderSSGI(
        VkCommandBuffer cmd,
        class GBufferTextures* inGBuffers,
        class RenderScene* scene,
        BufferParameterHandle perFrameGPU,
        PoolImageSharedRef inHiz,
        PoolImageSharedRef inGTAO)
    {
        const auto& postProcessVolumeSetting = scene->getPostprocessVolumeSetting();
        auto* pass = getContext()->getPasses().get<SSGIPass>();
        auto* rtPool = &m_context->getRenderTargetPools();

        auto gbufferB = inGBuffers->gbufferB;
        auto historyColor = m_prevHDR ? m_prevHDR : inGBuffers->hdrSceneColor;

        auto ssgiIntersectResult = rtPool->createPoolImage(
            "SSGI Intersect",
            gbufferB->getImage().getExtent().width,
            gbufferB->getImage().getExtent().height,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

        {
            ScopePerframeMarker tonemapperMarker(cmd, "SSGI-Intersect", { 1.0f, 1.0f, 0.0f, 1.0f });

            gbufferB->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            historyColor->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            ssgiIntersectResult->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            GpuSsgiIntersectPush intersectPush
            {

            };

            pass->intersect->bindAndPushConst(cmd, &intersectPush);
            PushSetBuilder(cmd)
                .addSRV(historyColor)
                .addUAV(ssgiIntersectResult)
                .addSRV(gbufferB)
                .addBuffer(perFrameGPU)
                .push(pass->intersect.get());

            pass->intersect->bindSet(cmd, std::vector<VkDescriptorSet>{
                m_context->getSamplerCache().getCommonDescriptorSet()
              , m_renderer->getBlueNoise().spp_1_buffer.set
            }, 1);

            vkCmdDispatch(cmd, getGroupCount(ssgiIntersectResult->getImage().getExtent().width, 8), getGroupCount(ssgiIntersectResult->getImage().getExtent().height, 8), 1);

            ssgiIntersectResult->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        }

        m_gpuTimer.getTimeStamp(cmd, "SSGI");

        m_displayDebug = ssgiIntersectResult;
    }
}