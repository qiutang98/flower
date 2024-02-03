#include "../deferred_renderer.h"
#include "../render_scene.h"
#include "../renderer.h"
#include "../scene_textures.h"

namespace engine
{
    class PostprocessingPass : public PassInterface
    {
    public:
        std::unique_ptr<ComputePipeResources> pipeTonemapper;

    public:
        virtual void onInit() override
        {
            VkDescriptorSetLayout setLayoutTonemapper = VK_NULL_HANDLE;
            getContext()->descriptorFactoryBegin()
                .bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0)
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1)
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  2)
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  3)
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  4)
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5)
                .buildNoInfoPush(setLayoutTonemapper);

            std::vector<VkDescriptorSetLayout> setLayoutsTone = 
            {
                setLayoutTonemapper,
                getContext()->getSamplerCache().getCommonDescriptorSetLayout(),
                getRenderer()->getBlueNoise().spp_1_buffer.setLayouts
            };

            pipeTonemapper = std::make_unique<ComputePipeResources>("shader/post_tonemapper.glsl", 0, setLayoutsTone);
        }

        virtual void release() override
        {
            pipeTonemapper.reset();
        }
    };


    void DeferredRenderer::postprocessing(
        VkCommandBuffer cmd, 
        GBufferTextures* inGBuffers, 
        BufferParameterHandle perFrameGPU, 
        RenderScene* scene,
        PoolImageSharedRef bloomTex,
        PoolImageSharedRef debugTex)
    {
        auto* pass = getContext()->getPasses().get<PostprocessingPass>();

        auto& hdrSceneColor = debugTex ? debugTex->getImage() : inGBuffers->hdrSceneColorUpscale->getImage();
        auto& ldrSceneColor = getOutput()->getImage();

        hdrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        ldrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
        {
            ScopePerframeMarker marker(cmd, "Tonemapper", { 1.0f, 1.0f, 0.0f, 1.0f }, &m_gpuTimer);
            pass->pipeTonemapper->bind(cmd);

            PushSetBuilder(cmd)
                .addBuffer(perFrameGPU)
                .addSRV(hdrSceneColor)
                .addUAV(ldrSceneColor)
                .addSRV(m_history.averageLum)
                .addSRV(bloomTex)
                .addBuffer(inGBuffers->lensBuffer != nullptr ?
                    *inGBuffers->lensBuffer->getBuffer() :
                    *getRenderer()->getSharedTextures().zeroBuffer, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .push(pass->pipeTonemapper.get());

            pass->pipeTonemapper->bindSet(cmd, std::vector<VkDescriptorSet>{
                getContext()->getSamplerCache().getCommonDescriptorSet(), 
                getRenderer()->getBlueNoise().spp_1_buffer.set
            }, 1);

            vkCmdDispatch(cmd, getGroupCount(ldrSceneColor.getExtent().width, 8), getGroupCount(ldrSceneColor.getExtent().height, 8), 1);
        }
    }
}