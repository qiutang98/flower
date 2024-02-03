#include "../deferred_renderer.h"
#include "../render_scene.h"
#include "../renderer.h"
#include "../scene_textures.h"

namespace engine
{
    class SceneDepthRangePass : public PassInterface
    {
    public:
        std::unique_ptr<ComputePipeResources> pipe;

    public:
        virtual void onInit() override
        {
            VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
            getContext()->descriptorFactoryBegin()
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  0)
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1)
                .buildNoInfoPush(setLayout);

            std::vector<VkDescriptorSetLayout> setLayouts = 
            {
                setLayout,
            };

            pipe = std::make_unique<ComputePipeResources>("shader/sceneDepthRangeExcludeSky.glsl", 0, setLayouts);
        }

        virtual void release() override
        {
            pipe.reset();
        }
    };

    class ReconstructNormalPass : public PassInterface
    {
    public:
        std::unique_ptr<ComputePipeResources> pipe;
        virtual void onInit() override
        {
            {
                VkDescriptorSetLayout layout = VK_NULL_HANDLE;

                getContext()->descriptorFactoryBegin()
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 0) // 
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1) // 
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,2) // 
                    .buildNoInfoPush(layout);

                std::vector<VkDescriptorSetLayout> layouts =
                {
                    layout,
                    getContext()->getSamplerCache().getCommonDescriptorSetLayout(),
                    getRenderer()->getBlueNoise().spp_1_buffer.setLayouts
                };

                pipe = std::make_unique<ComputePipeResources>("shader/reconstruct_normal.glsl", 0, layouts);
            }
        }

        virtual void release() override
        {
            pipe.reset();
        }

    };

    PoolImageSharedRef engine::reconstructNormal(VkCommandBuffer cmd, GBufferTextures* inGBuffers, BufferParameterHandle perFrameGPU, RenderScene* scene, GPUTimestamps* timer)
    {
        auto* pass = getContext()->getPasses().get<ReconstructNormalPass>();
        auto& sceneDepthZ = inGBuffers->depthTexture->getImage();
        sceneDepthZ.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));
        auto* rtPool = &getContext()->getRenderTargetPools();


        PoolImageSharedRef result = rtPool->createPoolImage(
            "vertex normal",
            sceneDepthZ.getExtent().width,
            sceneDepthZ.getExtent().height,
            VK_FORMAT_A2B10G10R10_UNORM_PACK32,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);

        result->getImage().transitionGeneral(cmd);
        {
            ScopePerframeMarker marker(cmd, "vertex normal build", { 1.0f, 0.0f, 0.0f, 1.0f }, timer);

            pass->pipe->bind(cmd);
            PushSetBuilder(cmd)
                .addSRV(sceneDepthZ, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT))
                .addUAV(result)
                .addBuffer(perFrameGPU)
                .push(pass->pipe.get());

            pass->pipe->bindSet(cmd, std::vector<VkDescriptorSet>{
                getContext()->getSamplerCache().getCommonDescriptorSet() }, 1);

            vkCmdDispatch(cmd, getGroupCount(result->getImage().getExtent().width, 8), getGroupCount(result->getImage().getExtent().height, 8), 1);
        }

        result->getImage().transitionShaderReadOnly(cmd);

        return result;
    }

    BufferParameterHandle engine::sceneDepthRangePass(
        VkCommandBuffer cmd, 
        GBufferTextures* inGBuffers, 
        BufferParameterHandle perFrameGPU, 
        RenderScene* scene,
        GPUTimestamps* timer)
    {
        auto* pass = getContext()->getPasses().get<SceneDepthRangePass>();

        auto& sceneDepthZ = inGBuffers->depthTexture->getImage();
        sceneDepthZ.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));

        auto rangeBuffer = getContext()->getBufferParameters().getStaticStorageGPUOnly(
            "SceneDepthRangeBuffer", sizeof(uint) * 2);

        {
            ScopePerframeMarker marker(cmd, "SceneDepthRangeCompute", { 1.0f, 0.0f, 0.0f, 1.0f }, timer);

            uint clearRangeValue[2] = { ~0u, 0u }; // Min & Max.

            vkCmdUpdateBuffer(cmd, *rangeBuffer->getBuffer(), 0, rangeBuffer->getBuffer()->getSize(), &clearRangeValue);
            std::array<VkBufferMemoryBarrier2, 1> fillBarriers
            {
                RHIBufferBarrier(rangeBuffer->getBuffer()->getVkBuffer(),
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT),
            };
            RHIPipelineBarrier(cmd, 0, (uint32_t)fillBarriers.size(), fillBarriers.data(), 0, nullptr);

            pass->pipe->bind(cmd);
            PushSetBuilder(cmd)
                .addSRV(sceneDepthZ, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT))
                .addBuffer(rangeBuffer)
                .push(pass->pipe.get());

            // Block dim is 3x3.
            vkCmdDispatch(cmd,
                getGroupCount(sceneDepthZ.getExtent().width / 3 + 1, 8),
                getGroupCount(sceneDepthZ.getExtent().height / 3 + 1, 8), 1);

            VkBufferMemoryBarrier2 endBufferBarrier = RHIBufferBarrier(rangeBuffer->getBuffer()->getVkBuffer(),
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
            RHIPipelineBarrier(cmd, 0, 1, &endBufferBarrier, 0, nullptr);
        }

        return rangeBuffer;
    }


}