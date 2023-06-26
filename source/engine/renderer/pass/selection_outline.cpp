#include "../renderer_interface.h"
#include "../render_scene.h"
#include "../renderer.h"
#include "../scene_textures.h"
#include "../../../editor/editor.h"

namespace engine
{
    // TODO: Occlusion state optimize. also custom draw gizmo.

    class SelectionOutlinePass : public PassInterface
    {
    public:
        VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
        std::unique_ptr<ComputePipeResources> pipe;

    public:
        virtual void onInit() override
        {
            // Config code.
            getContext()->descriptorFactoryBegin()
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0) // inColor
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1) // inId
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 2) // inId
                .bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 3)
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 4) // inId
                .buildNoInfoPush(setLayout);

            pipe = std::make_unique<ComputePipeResources>("shader/selection_outline.comp.spv", 0, std::vector<VkDescriptorSetLayout>{ setLayout, getContext()->getSamplerCache().getCommonDescriptorSetLayout() });
        }

        virtual void release() override
        {
            pipe.reset();
        }
    };


    void RendererInterface::renderSelectionOutline(VkCommandBuffer cmd, GBufferTextures* inGBuffers, BufferParameterHandle perFrameGPU)
    {
        if (Editor::get()->getSceneNodeSelections().getNum() == 0)
        {
            return;
        }

        auto& ldrSceneColor = inGBuffers->hdrSceneColor->getImage();
        ldrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

        auto& selectionOutlineMask = inGBuffers->selectionOutlineMask->getImage();
        selectionOutlineMask.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

        auto& idTexture = inGBuffers->idTexture->getImage();
        idTexture.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

        {
            ScopePerframeMarker tonemapperMarker(cmd, "SelectionOutline", { 1.0f, 1.0f, 0.0f, 1.0f });

            auto* pass = getContext()->getPasses().get<SelectionOutlinePass>();

            pass->pipe->bind(cmd);
            PushSetBuilder(cmd)
                .addUAV(ldrSceneColor)
                .addSRV(selectionOutlineMask)
                .addSRV(idTexture)
                .addBuffer(perFrameGPU)
                .addSRV(m_averageLum ? m_averageLum : inGBuffers->hdrSceneColorUpscale)
                .push(pass->pipe.get());

            pass->pipe->bindSet(cmd, std::vector<VkDescriptorSet>{ getContext()->getSamplerCache().getCommonDescriptorSet() }, 1);

            vkCmdDispatch(cmd, getGroupCount(ldrSceneColor.getExtent().width, 8), getGroupCount(ldrSceneColor.getExtent().height, 8), 1);

            m_gpuTimer.getTimeStamp(cmd, "SelectionOutline");
        }
    }
}