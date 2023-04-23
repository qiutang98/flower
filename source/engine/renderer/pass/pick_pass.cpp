#include "../renderer_interface.h"
#include "../render_scene.h"
#include "../renderer.h"
#include "../scene_textures.h"

namespace engine
{
    struct PickPushComposite
    {
        glm::ivec2 pickPos;
    };

    class PickPass : public PassInterface
    {
    public:
        VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
        std::unique_ptr<ComputePipeResources> pipe;

    public:
        virtual void onInit() override
        {
            // Config code.
            getContext()->descriptorFactoryBegin()
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0) // inHdr
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1) // inIdTexture
                .buildNoInfoPush(setLayout);

            pipe = std::make_unique<ComputePipeResources>("shader/pick.comp.spv", (uint32_t)sizeof(PickPushComposite), std::vector<VkDescriptorSetLayout>{ setLayout });
        }

        virtual void release() override
        {
            pipe.reset();
        }
    };


    void RendererInterface::getPickPixelObject(VkCommandBuffer cmd, GBufferTextures* inGBuffers)
    {
        if (m_pickIdBuffer)
        {
            ASSERT(!m_bPickInThisFrame, "You should no pick in this frame when exist pick id buffer!");
            
            // Sync major graphics queue.
            vkQueueWaitIdle(m_context->getMajorGraphicsQueue());

            uint32_t pickId;
            m_pickIdBuffer->getBuffer()->map();
            {
                m_pickIdBuffer->getBuffer()->invalidate();

                const uint32_t* pickData = (const uint32_t*)m_pickIdBuffer->getBuffer()->getMapped();
                memcpy(&pickId, m_pickIdBuffer->getBuffer()->getMapped(), m_pickIdBuffer->getBuffer()->getSize());
            }
            m_pickIdBuffer->getBuffer()->unmap();

            m_pickCallBack(pickId);

            // Reset pick buffer.
            m_pickIdBuffer = nullptr;
        }


        if (!m_bPickInThisFrame)
        {
            return;
        }

        ASSERT(m_pickIdBuffer == nullptr, "When pick id, id buffer must be null!");

        // Reset state.
        m_bPickInThisFrame = false;

        // Pick texture id in buffer.
        auto& idTexture = inGBuffers->idTexture->getImage();
        idTexture.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        {
            ScopePerframeMarker marker(cmd, "Pick", { 1.0f, 1.0f, 0.0f, 1.0f });

            auto* pass = getContext()->getPasses().get<PickPass>();

            PickPushComposite compositePush
            {
                .pickPos = m_pickPosCurrentFrame,
            }; 

            auto idBuffer = m_context->getBufferParameters().getParameter("IdBuffer", sizeof(uint32_t), 
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, {});

            auto idHostBuffer = m_context->getBufferParameters().getParameter("idHostBuffer", sizeof(uint32_t),
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VulkanBuffer::getReadBackFlags());

            pass->pipe->bindAndPushConst(cmd, &compositePush);
            PushSetBuilder(cmd)
                .addBuffer(idBuffer)
                .addSRV(idTexture)
                .push(pass->pipe.get());

            vkCmdDispatch(cmd, 1, 1, 1);

            // Dispatch barrier.
            auto fillBarriers = RHIBufferBarrier(idBuffer->getBuffer()->getVkBuffer(),
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            RHIPipelineBarrier(cmd, 0, 1, &fillBarriers, 0, nullptr);


            // Read back to host visible buffer
            VkBufferCopy copyRegion = {};
            copyRegion.size = sizeof(uint32_t);
            vkCmdCopyBuffer(cmd, idBuffer->getBuffer()->getVkBuffer(), idHostBuffer->getBuffer()->getVkBuffer(), 1, &copyRegion);

            // Readback barrier.
            auto readbackBarriers = RHIBufferBarrier(idHostBuffer->getBuffer()->getVkBuffer(),
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
            RHIPipelineBarrier(cmd, 0, 1, &readbackBarriers, 0, nullptr);

            m_pickIdBuffer = idHostBuffer;
        }
    }
}