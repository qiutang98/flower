#include "../deferred_renderer.h"
#include "../render_scene.h"
#include "../renderer.h"
#include "../scene_textures.h"

namespace engine
{
    struct GIDiffusePassPush
    {
        uint bSSGIValid;
    };

    AutoCVarInt32 cVarSSGIComposite(
        "r.ssgi.enableComposite",
        "Enable ssgi composite.",
        "Rendering",
        1,
        CVarFlags::ReadAndWrite);

    class GIDiffusePass : public PassInterface
    {
    public:
        std::unique_ptr<ComputePipeResources> skylightOnlyPipe;

    public:
        virtual void onInit() override
        {
            VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
            getContext()->descriptorFactoryBegin()
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  0) // Hdr
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1) // inDepth
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  2) // inGbufferA
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  3) // inGbufferB
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  4) // inGbufferS
                .bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 5) // inFrameData
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  6) // inSkyIrradiance
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  7) // inSkyIrradiance
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 8) // inSkyIrradiance
                .buildNoInfoPush(setLayout);

            std::vector<VkDescriptorSetLayout> setLayouts = {
				setLayout,
				m_context->getSamplerCache().getCommonDescriptorSetLayout()
			};

            ShaderVariant shaderVariant("shader/gi_diffuse.glsl");
            shaderVariant.setStage(EShaderStage::eComputeShader);

            {
                auto copyVariant = shaderVariant;
                copyVariant.setMacro(L"SKY_LIGHT_ONLY_PASS");
                skylightOnlyPipe = std::make_unique<ComputePipeResources>(copyVariant, sizeof(GIDiffusePassPush), setLayouts);
            }
        }

        virtual void release() override
        {
            skylightOnlyPipe.reset();
        }
    };

    void engine::renderGIDiffuse(
        VkCommandBuffer cmd,
        GBufferTextures* inGBuffers,
        RenderScene* scene,
        BufferParameterHandle perFrameGPU, 
        PoolImageSharedRef bentNormalSSAO,
        PoolImageSharedRef inSSGI,
        SkyLightRenderContext& skylightContext,
        GPUTimestamps* timer)
    {
        PoolImageSharedRef ssgi = inSSGI;
        if (cVarSSGIComposite.get() == 0)
        {
            ssgi = nullptr;
        }

        if (!skylightContext.skylightRadiance)
        {
            return;
        }

        auto* rtPool = &getContext()->getRenderTargetPools();
        auto* pass = getContext()->getPasses().get<GIDiffusePass>();

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

        {
            ScopePerframeMarker marker(cmd, "GIDiffuse", { 1.0f, 1.0f, 0.0f, 1.0f }, timer);

            GIDiffusePassPush push{ .bSSGIValid = (ssgi != nullptr) };

            pass->skylightOnlyPipe->bindAndPushConst(cmd, &push);
            PushSetBuilder(cmd)
                .addUAV(hdrSceneColor)
                .addSRV(sceneDepthZ, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT))
                .addSRV(gbufferA)
                .addSRV(gbufferB)
                .addSRV(gbufferS)
                .addBuffer(perFrameGPU)
                .addSRV(skylightContext.skylightRadiance, buildBasicImageSubresourceCube(), VK_IMAGE_VIEW_TYPE_CUBE)
                .addSRV(bentNormalSSAO ? bentNormalSSAO->getImage()
                    : getContext()->getBuiltinTextureWhite()->getSelfImage())
                .addSRV(ssgi ? ssgi->getImage()
                    : getContext()->getBuiltinTextureTranslucent()->getSelfImage())
                .push(pass->skylightOnlyPipe.get());

            pass->skylightOnlyPipe->bindSet(cmd, std::vector<VkDescriptorSet>{ 
                getContext()->getSamplerCache().getCommonDescriptorSet() }, 1);

            vkCmdDispatch(cmd, getGroupCount(hdrSceneColor.getExtent().width, 8), getGroupCount(hdrSceneColor.getExtent().height, 8), 1);
        }
    }

}