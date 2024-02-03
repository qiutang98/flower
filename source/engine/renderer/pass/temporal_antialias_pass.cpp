#include "../deferred_renderer.h"
#include "../render_scene.h"
#include "../renderer.h"
#include "../scene_textures.h"

namespace engine
{
    struct PushConstForTAA
    {
        float kAntiFlickerIntensity;
        float kContrastForMaxAntiFlicker;
        float kSampleHistorySharpening;
        float kHistoryContrastBlendLerp;

        float kBaseBlendFactor;
        float kFilterWeights[9];
    };

    class TemporalAntiAliasPass : public PassInterface
    {
    public:
        std::unique_ptr<ComputePipeResources> pipeTemporalAntiAlias;

    public:
        virtual void onInit() override
        {
            VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
            getContext()->descriptorFactoryBegin()
                .bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0)
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1)
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  2)
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  3)
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  4)
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  5)
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  6)
                .buildNoInfoPush(setLayout);

            std::vector<VkDescriptorSetLayout> setLayoutsTone = 
            {
                setLayout,
                getContext()->getSamplerCache().getCommonDescriptorSetLayout()
            };

            pipeTemporalAntiAlias = std::make_unique<ComputePipeResources>("shader/temporal_antialias.glsl", sizeof(PushConstForTAA), setLayoutsTone);
        }

        virtual void release() override
        {
            pipeTemporalAntiAlias.reset();
        }
    };


    void DeferredRenderer::temporalAntiAliasUpscale(
        VkCommandBuffer cmd, 
        GBufferTextures* inGBuffers, 
        BufferParameterHandle perFrameGPU, 
        RenderScene* scene)
    {
        auto* pass = getContext()->getPasses().get<TemporalAntiAliasPass>();

        auto& hdrSceneColor = inGBuffers->hdrSceneColor->getImage();
        auto& sceneDepthZ = inGBuffers->depthTexture->getImage();
        auto& velocity = inGBuffers->gbufferV->getImage();

        auto& taaSceneColor = inGBuffers->hdrSceneColorUpscale->getImage();

        hdrSceneColor.transitionShaderReadOnly(cmd);
        velocity.transitionShaderReadOnly(cmd);
        sceneDepthZ.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));

        m_history.averageLum->getImage().transitionShaderReadOnly(cmd);

        if (m_history.historyUpscale)
        {
            m_history.historyUpscale->getImage().transitionShaderReadOnly(cmd);
        }

        PushConstForTAA taaPush { };
        {
            float taaAntiFlicker     = m_perframe.postprocessing.taaAntiFlicker;
            float taaHistorySharpen  = m_perframe.postprocessing.taaHistorySharpen;
            float taaBaseBlendFactor = m_perframe.postprocessing.taaBaseBlendFactor;

            constexpr float kMinAntiflicker = 0.0f;
            constexpr float kMaxAntiflicker = 3.5f;
            constexpr float historyContrastBlendStart = 0.51f;
            taaPush.kContrastForMaxAntiFlicker = 0.7f - math::mix(0.0f, 0.3f, math::smoothstep(0.5f, 1.0f, taaAntiFlicker));
            taaPush.kAntiFlickerIntensity      = math::mix(kMinAntiflicker, kMaxAntiflicker, taaAntiFlicker);
            taaPush.kSampleHistorySharpening   = taaHistorySharpen;
            taaPush.kHistoryContrastBlendLerp  = math::clamp((taaAntiFlicker - historyContrastBlendStart) / (1.0f - historyContrastBlendStart), 0.0f, 1.0f);
            taaPush.kBaseBlendFactor           = 1.0f - taaBaseBlendFactor;


            // Precompute weights used for the Blackman-Harris filter.
            static const vec2 kTAASampleOffsets[9] =
            {
                vec2( 0,  0),
                vec2( 0,  1),
                vec2( 1,  0),
                vec2(-1,  0),
                vec2( 0, -1),
                vec2(-1,  1),
                vec2( 1, -1),
                vec2( 1,  1),
                vec2(-1, -1)
            };
            float totalWeight = 0;
            for (int i = 0; i < 9; ++i)
            {
                float x = kTAASampleOffsets[i].x + m_perframe.jitterData.x;
                float y = kTAASampleOffsets[i].y + m_perframe.jitterData.y;
                float d = (x * x + y * y);

                taaPush.kFilterWeights[i] = math::exp((-0.5f / (0.22f)) * d);
                totalWeight += taaPush.kFilterWeights[i];
            }

            for (int i = 0; i < 9; ++i)
            {
                taaPush.kFilterWeights[i] /= totalWeight;
            }
        }

        taaSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL);
        {
            ScopePerframeMarker marker(cmd, "TemporalAntiAliasUpscale", { 1.0f, 1.0f, 0.0f, 1.0f }, &m_gpuTimer);
            pass->pipeTemporalAntiAlias->bindAndPushConst(cmd, &taaPush);

            PushSetBuilder(cmd)
                .addBuffer(perFrameGPU)
                .addSRV(hdrSceneColor)
                .addSRV(sceneDepthZ, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT))
                .addSRV(velocity)
                .addSRV(m_history.averageLum)
                .addUAV(taaSceneColor)
                .addSRV(m_history.historyUpscale ? 
                    m_history.historyUpscale->getImage() : 
                    getContext()->getBuiltinTextureTranslucent()->getSelfImage())
                .push(pass->pipeTemporalAntiAlias.get());

            pass->pipeTemporalAntiAlias->bindSet(cmd, std::vector<VkDescriptorSet>{
                getContext()->getSamplerCache().getCommonDescriptorSet()
            }, 1);

            vkCmdDispatch(cmd, 
                getGroupCount(taaSceneColor.getExtent().width, 8), 
                getGroupCount(taaSceneColor.getExtent().height, 8), 1);
        }


        // Swap history.
        m_history.historyUpscale = inGBuffers->hdrSceneColorUpscale;
    }
}