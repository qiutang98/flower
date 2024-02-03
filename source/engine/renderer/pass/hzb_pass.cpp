#include "../deferred_renderer.h"
#include "../render_scene.h"
#include "../renderer.h"
#include "../scene_textures.h"

namespace engine
{
    struct HzbPassPush
    {
        uint32_t bFromSrcDepth = 1;
    };

    class HzbPass : public PassInterface
    {
    public:
        std::unique_ptr<ComputePipeResources> pipe;
        std::unique_ptr<ComputePipeResources> chessboardPipe;

    public:
        virtual void onInit() override
        {
            {
                // Config code.
                VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
                getContext()->descriptorFactoryBegin()
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 0) // hizClosestImage
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1) // hizFurthestImage
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 2) // inDepth
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 3) // inSrcHizClosest
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 4) // inSrcHizFurthest
                    .buildNoInfoPush(setLayout);

                pipe = std::make_unique<ComputePipeResources>("shader/hzb.glsl", (uint32_t)sizeof(HzbPassPush), std::vector<VkDescriptorSetLayout>{ setLayout });
            }

            {
                // Config code.
                VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
                getContext()->descriptorFactoryBegin()
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 0) // hizClosestImage
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1) // inDepth
                    .buildNoInfoPush(setLayout);

                chessboardPipe = std::make_unique<ComputePipeResources>(
                    "shader/depth_downsample_chessboard.glsl", 0, std::vector<VkDescriptorSetLayout>{ setLayout });
            }
        }

        virtual void release() override
        {
            pipe.reset();

            chessboardPipe.reset();
        }
    };

    void engine::renderHzb(
        PoolImageSharedRef& outClosed,
        PoolImageSharedRef& outFurthest,
        VkCommandBuffer cmd,
        GBufferTextures* inGBuffers,
        RenderScene* scene,
        BufferParameterHandle perFrameGPU,
        GPUTimestamps* timer)
    {
        auto* pass = getContext()->getPasses().get<HzbPass>();
        auto* rtPool = &getContext()->getRenderTargetPools();

        auto& depthTex = inGBuffers->depthTexture->getImage();
        depthTex.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));

        uint32_t mipStartWidth = depthTex.getExtent().width;
        uint32_t mipStartHeight = depthTex.getExtent().height;

        auto hizMipChainCloest = rtPool->createPoolImage(
            "HizMipchain_closet",
            mipStartWidth,
            mipStartHeight,
            VK_FORMAT_R32_SFLOAT,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
            kRenderTextureFullMip);

        auto hizMipChainFurthest = rtPool->createPoolImage(
            "HizMipchain_furest",
            mipStartWidth,
            mipStartHeight,
            VK_FORMAT_R32_SFLOAT,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
            kRenderTextureFullMip);

        {
            ScopePerframeMarker marker(cmd, "Hzb", { 0.8f, 1.0f, 0.0f, 1.0f }, timer);

            pass->pipe->bind(cmd);

            // Build from src.
            HzbPassPush push{ .bFromSrcDepth = 1 };
            pass->pipe->pushConst(cmd, &push);
            {
                VkImageSubresourceRange rangeMip0{ .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1 };
                hizMipChainCloest->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, rangeMip0);
                hizMipChainFurthest->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, rangeMip0);

                PushSetBuilder(cmd)
                    .addUAV(hizMipChainCloest, rangeMip0)
                    .addUAV(hizMipChainFurthest, rangeMip0)
                    .addSRV(depthTex, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT))
                    .addSRV(depthTex, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT))
                    .addSRV(depthTex, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT))
                    .push(pass->pipe.get());

                vkCmdDispatch(cmd, getGroupCount(mipStartWidth, 8), getGroupCount(mipStartHeight, 8), 1);

                hizMipChainCloest->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, rangeMip0);
                hizMipChainFurthest->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, rangeMip0);
            }

            // Build from hiz.
            push.bFromSrcDepth = 0;
            pass->pipe->pushConst(cmd, &push);
            if (hizMipChainCloest->getImage().getInfo().mipLevels > 1)
            {
                uint32_t loopWidth = mipStartWidth;
                uint32_t loopHeight = mipStartHeight;

                for (uint32_t i = 1; i < hizMipChainCloest->getImage().getInfo().mipLevels; i++)
                {
                    VkImageSubresourceRange rangeMipN_1{ .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = i - 1, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1 };
                    VkImageSubresourceRange rangeMipN{ .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = i, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1 };

                    hizMipChainCloest->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, rangeMipN_1);
                    hizMipChainFurthest->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, rangeMipN_1);
                    hizMipChainCloest->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, rangeMipN);
                    hizMipChainFurthest->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, rangeMipN);

                    PushSetBuilder(cmd)
                        .addUAV(hizMipChainCloest, rangeMipN)
                        .addUAV(hizMipChainFurthest, rangeMipN)
                        .addSRV(hizMipChainCloest, rangeMipN_1)
                        .addSRV(hizMipChainCloest, rangeMipN_1)
                        .addSRV(hizMipChainFurthest, rangeMipN_1)
                        .push(pass->pipe.get());

                    loopWidth = math::max(1u, loopWidth / 2);
                    loopHeight = math::max(1u, loopHeight / 2);

                    vkCmdDispatch(cmd, getGroupCount(loopWidth, 8), getGroupCount(loopHeight, 8), 1);
                }
            }

            hizMipChainCloest->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            hizMipChainFurthest->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        }

        outClosed = hizMipChainCloest;
        outFurthest = hizMipChainFurthest;

        inGBuffers->hzbClosest = hizMipChainCloest;
        inGBuffers->hzbFurthest = hizMipChainFurthest;

        {
            auto chessboardHalfDepth = rtPool->createPoolImage(
                "chessboardHalfDepth",
                depthTex.getExtent().width / 2,
                depthTex.getExtent().height / 2,
                VK_FORMAT_R32_SFLOAT,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);

            chessboardHalfDepth->getImage().transitionGeneral(cmd);

            pass->chessboardPipe->bind(cmd);

            PushSetBuilder(cmd)
                .addUAV(chessboardHalfDepth)
                .addSRV(depthTex, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT))
                .push(pass->chessboardPipe.get());

            vkCmdDispatch(cmd, 
                getGroupCount(chessboardHalfDepth->getImage().getExtent().width, 8), 
                getGroupCount(chessboardHalfDepth->getImage().getExtent().height, 8), 1);

            chessboardHalfDepth->getImage().transitionShaderReadOnly(cmd);


            inGBuffers->chessboardHalfDepth = chessboardHalfDepth;
        }


    }
}