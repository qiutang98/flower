#include "../deferred_renderer.h"
#include "../render_scene.h"
#include "../renderer.h"
#include "../scene_textures.h"

namespace engine
{
    struct PickPushComposite
    {
        glm::vec2 pickUv;
    };

    class PickPass : public PassInterface
    {
    public:

        std::unique_ptr<ComputePipeResources> pipe;

    public:
        virtual void onInit() override
        {
            VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;

            // Config code.
            getContext()->descriptorFactoryBegin()
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0) // 
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1) // 
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2) // 
                .bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3)
                .buildNoInfoPush(setLayout);

            pipe = std::make_unique<ComputePipeResources>(
                "shader/editor_pick.glsl", 
                (uint32_t)sizeof(PickPushComposite), 
                std::vector<VkDescriptorSetLayout>{ setLayout, getContext()->getSamplerCache().getCommonDescriptorSetLayout() });
        }

        virtual void release() override
        {
            pipe.reset();
        }
    };


    void DeferredRenderer::markCurrentFramePick(math::ivec2 pos, std::function<void(uint32_t pickCallback)>&& callback)
    {
        // Only dispatch pick when no exist pick buffer.
        if (m_pickContext.pickIdBuffer == nullptr)
        {
            m_pickContext.bPickInThisFrame = true;
            m_pickContext.pickPosCurrentFrame = pos;
            m_pickContext.pickCallBack = callback;
        }
    }


    void DeferredRenderer::getPickPixelObject(VkCommandBuffer cmd, GBufferTextures* inGBuffers, RenderScene* scene, 
        BufferParameterHandle perFrameGPU)
    {
        if (m_pickContext.pickIdBuffer)
        {
            ASSERT(!m_pickContext.bPickInThisFrame, "You should no pick in this frame when exist pick id buffer!");

            // Sync major graphics queue.
            vkQueueWaitIdle(getContext()->getMajorGraphicsQueue());

            uint32_t pickId;
            m_pickContext.pickIdBuffer->getBuffer()->map();
            {
                m_pickContext.pickIdBuffer->getBuffer()->invalidate();

                const uint32_t* pickData = (const uint32_t*)m_pickContext.pickIdBuffer->getBuffer()->getMapped();
                memcpy(&pickId, m_pickContext.pickIdBuffer->getBuffer()->getMapped(), m_pickContext.pickIdBuffer->getBuffer()->getSize());
            }
            m_pickContext.pickIdBuffer->getBuffer()->unmap();

            // 0 is Root
            if (pickId != 0 && pickId != ~0)
            {
                m_pickContext.pickCallBack(pickId);
            }

            // Reset pick buffer when callback ready.
            m_pickContext.pickIdBuffer = nullptr;
        }


        if (!m_pickContext.bPickInThisFrame)
        {
            return;
        }

        if (scene->getObjectCollector().empty())
        {
            return;
        }

        ASSERT(m_pickContext.pickIdBuffer == nullptr, "When pick id, id buffer must be null!");

        // Reset state.
        m_pickContext.bPickInThisFrame = false;

        // Pick texture id in buffer.
        auto& idTexture = inGBuffers->gbufferId->getImage();
        idTexture.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        {
            ScopePerframeMarker marker(cmd,  "Pick", { 1.0f, 1.0f, 0.0f, 1.0f }, &m_gpuTimer);

            auto* pass = getContext()->getPasses().get<PickPass>();
            
            PickPushComposite compositePush
            {
                .pickUv = { 
                    (m_pickContext.pickPosCurrentFrame.x + 0.5f) / (float)getDimensions().getOutputWidth(),
                    (m_pickContext.pickPosCurrentFrame.y + 0.5f) / (float)getDimensions().getOutputHeight()},
            };

            auto idBuffer = getContext()->getBufferParameters().getParameter("IdBuffer", sizeof(uint32_t),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, {});

            auto idHostBuffer = getContext()->getBufferParameters().getParameter("idHostBuffer", sizeof(uint32_t),
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VulkanBuffer::getReadBackFlags());

            pass->pipe->bindAndPushConst(cmd, &compositePush);
            PushSetBuilder(cmd)
                .addBuffer(idBuffer)
                .addSRV(idTexture)
                .addBuffer(scene->getObjectBufferGPU())
                .addBuffer(perFrameGPU)
                .push(pass->pipe.get());

            pass->pipe->bindSet(cmd, std::vector<VkDescriptorSet>{ getContext()->getSamplerCache().getCommonDescriptorSet() }, 1);

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

            m_pickContext.pickIdBuffer = idHostBuffer;
        }
    }
}