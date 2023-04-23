#include "../renderer_interface.h"
#include "../render_scene.h"
#include "../renderer.h"
#include "../scene_textures.h"

namespace engine
{
    struct TonemapperPushComposite
    {
        math::vec4 prefilterFactor;
        float bloomIntensity;
        float bloomBlur;
    };

    class TonemapperPass : public PassInterface
    {
    public:
        VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
        std::unique_ptr<ComputePipeResources> pipe;

    public:
        virtual void onInit() override
        {
            // Config code.
            getContext()->descriptorFactoryBegin()
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0) // inHdr
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1) // outLdr
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 2) // outLdr
                .bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 3) // uniform
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 4) // inHdr
                .buildNoInfoPush(setLayout);

            std::vector<VkDescriptorSetLayout> setLayouts = {
                setLayout,
                getContext()->getSamplerCache().getCommonDescriptorSetLayout(),
                getRenderer()->getBlueNoise().spp_1_buffer.setLayouts
            };

            pipe = std::make_unique<ComputePipeResources>("shader/tone_mapper.comp.spv", (uint32_t)sizeof(TonemapperPushComposite), setLayouts);
        }

        virtual void release() override
        {
            pipe.reset();
        }
    };


    void RendererInterface::renderTonemapper(VkCommandBuffer cmd, GBufferTextures* inGBuffers, BufferParameterHandle perFrameGPU, RenderScene* scene, PoolImageSharedRef bloomTex)
    {
        auto& hdrSceneColor = inGBuffers->hdrSceneColorUpscale->getImage();
        auto& ldrSceneColor = getDisplayOutput();

        {
            hdrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            ldrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
        }

        {
            ScopePerframeMarker tonemapperMarker(cmd, "Tonemapper", { 1.0f, 1.0f, 0.0f, 1.0f });

            auto* pass = getContext()->getPasses().get<TonemapperPass>();
            const auto& postProcessVolumeSetting = scene->getPostprocessVolumeSetting();
            TonemapperPushComposite compositePush
            {
                .prefilterFactor = getBloomPrefilter(postProcessVolumeSetting.bloomThreshold, postProcessVolumeSetting.bloomThresholdSoft),
                .bloomIntensity = postProcessVolumeSetting.bloomIntensity,
                .bloomBlur = postProcessVolumeSetting.bloomRadius,
            }; 

            pass->pipe->bindAndPushConst(cmd, &compositePush);
            PushSetBuilder(cmd)
                .addSRV(hdrSceneColor)
                .addUAV(ldrSceneColor)
                .addSRV(m_averageLum ? m_averageLum : inGBuffers->hdrSceneColorUpscale)
                .addBuffer(perFrameGPU)
                .addSRV(bloomTex)
                .push(pass->pipe.get());

            pass->pipe->bindSet(cmd, std::vector<VkDescriptorSet>{
                m_context->getSamplerCache().getCommonDescriptorSet()
              , m_renderer->getBlueNoise().spp_1_buffer.set
            }, 1);

            vkCmdDispatch(cmd, getGroupCount(ldrSceneColor.getExtent().width, 8), getGroupCount(ldrSceneColor.getExtent().height, 8), 1);

            m_gpuTimer.getTimeStamp(cmd, "Tonemappering");
        }
    }
}