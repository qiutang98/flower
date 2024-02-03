#include "../deferred_renderer.h"
#include "../render_scene.h"
#include "../renderer.h"
#include "../scene_textures.h"
#include "../../scene/component/reflection_probe_component.h"

namespace engine
{

    class GIReflectionPass : public PassInterface
    {
    public:
        std::unique_ptr<ComputePipeResources> reflectionPipe;

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
                .bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 5) // inFrameData
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 6) // inGbufferS
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 7) // inSkyIrradiance
                .buildNoInfoPush(setLayout);

            std::vector<VkDescriptorSetLayout> setLayouts = {
                setLayout,
                m_context->getSamplerCache().getCommonDescriptorSetLayout()
            };

            ShaderVariant shaderVariant("shader/gi_reflection.glsl");
            shaderVariant.setStage(EShaderStage::eComputeShader);

            {
                auto copyVariant = shaderVariant;
                copyVariant.setMacro(L"REFLECTION_COMPOSITE_PASS");
                reflectionPipe = std::make_unique<ComputePipeResources>(copyVariant, 0, setLayouts);
            }
        }

        virtual void release() override
        {
            reflectionPipe.reset();
        }
    };

    void engine::renderGIReflection(
        VkCommandBuffer cmd,
        GBufferTextures* inGBuffers,
        RenderScene* scene,
        const PerFrameData& perframe,
        BufferParameterHandle perFrameGPU,
        SkyLightRenderContext& skylightContext,
        GPUTimestamps* timer)
    {
        if (!skylightContext.skylightReflection)
        {
            return;
        }

        auto* rtPool = &getContext()->getRenderTargetPools();
        auto* pass = getContext()->getPasses().get<GIReflectionPass>();

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
        hdrSceneColor.transitionGeneral(cmd);

        auto probe = skylightContext.skylightReflection;
        {
            ScopePerframeMarker marker(cmd, "GIGlossy", { 1.0f, 1.0f, 0.0f, 1.0f }, timer);

            pass->reflectionPipe->bind(cmd);
            PushSetBuilder(cmd)
                .addUAV(hdrSceneColor)
                .addSRV(sceneDepthZ, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT))
                .addSRV(gbufferA)
                .addSRV(gbufferB)
                .addSRV(gbufferS)
                .addBuffer(perFrameGPU)
                .addSRV(*getRenderer()->getSharedTextures().brdfLut)
                .addSRV(probe, buildBasicImageSubresourceCube(), VK_IMAGE_VIEW_TYPE_CUBE)
                .push(pass->reflectionPipe.get());

            pass->reflectionPipe->bindSet(cmd, std::vector<VkDescriptorSet>{
                getContext()->getSamplerCache().getCommonDescriptorSet() }, 1);

            vkCmdDispatch(cmd, getGroupCount(hdrSceneColor.getExtent().width, 8), getGroupCount(hdrSceneColor.getExtent().height, 8), 1);
        }

        hdrSceneColor.transitionShaderReadOnly(cmd);
    }
}