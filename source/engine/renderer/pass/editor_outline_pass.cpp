#include "../deferred_renderer.h"
#include "../render_scene.h"
#include "../renderer.h"
#include "../scene_textures.h"
#include "../../../editor/editor.h"
#include <editor/widgets/scene_outliner.h>
#include <editor/selection.h>

namespace engine
{
    class SelectionOutlinePass : public PassInterface
    {
    public:
        std::unique_ptr<ComputePipeResources> pipe;

    public:
        virtual void onInit() override
        {
            VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;

            // Config code.
            getContext()->descriptorFactoryBegin()
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  0) // inColor
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1) // inId
                .bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2) // inId
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  3)
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4) // inId
                .buildNoInfoPush(setLayout);

            pipe = std::make_unique<ComputePipeResources>(
                "shader/editor_outline.glsl", 
                0, 
                std::vector<VkDescriptorSetLayout>{ setLayout, getContext()->getSamplerCache().getCommonDescriptorSetLayout() });
        }

        virtual void release() override
        {
            pipe.reset();
        }
    };


    void DeferredRenderer::renderSelectionOutline(
        VkCommandBuffer cmd, 
        GBufferTextures* inGBuffers, 
        BufferParameterHandle perFrameGPU,
        RenderScene* scene)
    {
        if (Editor::get()->getSceneOutlineWidegt()->getSelection().empty())
        {
            return;
        }

        if (scene->getObjectCollector().empty())
        {
            return;
        }

        auto& color = inGBuffers->hdrSceneColor->getImage();
        color.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

        auto& idTex = inGBuffers->gbufferId->getImage();
        idTex.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

        {
            ScopePerframeMarker tonemapperMarker(cmd, "SelectionOutline", { 1.0f, 1.0f, 0.0f, 1.0f }, &m_gpuTimer);

            auto* pass = getContext()->getPasses().get<SelectionOutlinePass>();
            
            pass->pipe->bind(cmd);
            PushSetBuilder(cmd)
                .addUAV(color)
                .addSRV(idTex)
                .addBuffer(perFrameGPU)
                .addSRV(m_history.averageLum ? m_history.averageLum->getImage() : getContext()->getBuiltinTextureWhite()->getSelfImage())
                .addBuffer(scene->getObjectBufferGPU())
                .push(pass->pipe.get());

            pass->pipe->bindSet(cmd, std::vector<VkDescriptorSet>{ getContext()->getSamplerCache().getCommonDescriptorSet() }, 1);

            vkCmdDispatch(cmd, getGroupCount(color.getExtent().width, 8), getGroupCount(color.getExtent().height, 8), 1);
        }
    }
}