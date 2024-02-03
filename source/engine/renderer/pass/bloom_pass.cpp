#include "../deferred_renderer.h"
#include "../render_scene.h"
#include "../renderer.h"
#include "../scene_textures.h"
#include "../../scene/component/postprocess_component.h"

namespace engine
{
    constexpr uint32_t kMaxDownsampleCount = 6;

    struct BloomDownsample
    {
        glm::vec4 prefilterFactor;
        uint32_t mipLevel;
    };

    struct BloomPushUpscale
    {
        uint32_t bBlurX;
        uint32_t bFinalBlur = 0u;
        uint32_t upscaleTime;
        float    blurRadius;
    };

    class BloomPass : public PassInterface
    {
    public:
        std::unique_ptr<ComputePipeResources> downsamplePipe;
        std::unique_ptr<ComputePipeResources> upscalePipe;

    protected:
        virtual void onInit() override
        {
            VkDescriptorSetLayout setLayoutDownSample = VK_NULL_HANDLE;
            getContext()->descriptorFactoryBegin()
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 0) // in
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1) // out
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 2) // lum
                .bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3)// frame data
                .buildNoInfoPush(setLayoutDownSample);

            VkDescriptorSetLayout setLayoutUpscale = VK_NULL_HANDLE;
            getContext()->descriptorFactoryBegin()
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 0) // inHdr
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1) // inCurHdr
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2) // out
                .buildNoInfoPush(setLayoutUpscale);

            std::vector<VkDescriptorSetLayout> setLayoutsDown = { setLayoutDownSample, m_context->getSamplerCache().getCommonDescriptorSetLayout() };
            std::vector<VkDescriptorSetLayout> setLayoutsUp = { setLayoutUpscale, m_context->getSamplerCache().getCommonDescriptorSetLayout() };

            ShaderVariant shaderVariant("shader/bloom.glsl");
            shaderVariant.setStage(EShaderStage::eComputeShader);

            {
                auto copyVariant = shaderVariant;
                copyVariant.setMacro(L"BLOOM_DOWNSAMPLE_PASS");
                downsamplePipe = std::make_unique<ComputePipeResources>(copyVariant, sizeof(BloomDownsample), setLayoutsDown);
            }

            {
                auto copyVariant = shaderVariant;
                copyVariant.setMacro(L"BLOOM_UPSCALE_PASS");
                upscalePipe = std::make_unique<ComputePipeResources>(copyVariant, sizeof(BloomPushUpscale), setLayoutsUp);
            }
        }

        virtual void release() override
        {
            downsamplePipe.reset();
            upscalePipe.reset();
        }
    };

    PoolImageSharedRef engine::renderBloom(
        VkCommandBuffer cmd,
        GBufferTextures* inGBuffers,
        RenderScene* scene,
        BufferParameterHandle perFrameGPU,
        const PostprocessVolumeSetting& setting,
        GPUTimestamps* timer,
        PoolImageSharedRef exposureImage)
    {
        auto* pass = getContext()->getPasses().get<BloomPass>();
        auto* rtPool = &getContext()->getRenderTargetPools();

        auto& hdrSceneColor = inGBuffers->hdrSceneColorUpscale->getImage();
        hdrSceneColor.transitionShaderReadOnly(cmd);

        const uint32_t srcHdrColorWidth = hdrSceneColor.getExtent().width;
        const uint32_t srcHdrColorHeight = hdrSceneColor.getExtent().height;

        // Min size is 64x64
        const uint32_t mipStartWidth = srcHdrColorWidth >> 1;
        const uint32_t mipStartHeight = srcHdrColorHeight >> 1;

        const uint32_t downsampleMipCount = glm::min(kMaxDownsampleCount, std::bit_width(glm::min(mipStartWidth, mipStartHeight)) - 1U);


        std::vector<PoolImageSharedRef> downsampleBlurs;
        downsampleBlurs.resize(downsampleMipCount);

        std::vector<VkDescriptorSet> additionalSets =
        {
            getContext()->getSamplerCache().getCommonDescriptorSet()
        };

        for (uint32_t i = 0; i < downsampleMipCount; i++)
        {
            downsampleBlurs[i] = rtPool->createPoolImage(
                "SceneColorBlurChain",
                mipStartWidth >> i,
                mipStartHeight >> i,
                hdrSceneColor.getFormat(),
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
        }

        PoolImageSharedRef result;
        {
            ScopePerframeMarker marker(cmd, "Bloom Basic", { 1.0f, 1.0f, 0.0f, 1.0f }, timer);

            pass->downsamplePipe->bind(cmd);
            pass->downsamplePipe->bindSet(cmd, additionalSets, 1);

            BloomDownsample downsamplePush{};


            downsamplePush.prefilterFactor = getBloomPrefilter(
                setting.bloomThreshold,
                setting.bloomThresholdSoft);

            auto frameBufferInfo = perFrameGPU->getBufferInfo();

            VkDescriptorImageInfo inImageInfo{};
            VkDescriptorImageInfo outImageInfo{};
            for (uint32_t i = 0; i < downsampleMipCount; i++)
            {
                const bool bFirstLevel = (i == 0);
                downsamplePush.mipLevel = i;

                inImageInfo = RHIDescriptorImageInfoSample((bFirstLevel ? hdrSceneColor : downsampleBlurs[i - 1]->getImage()).getOrCreateView(buildBasicImageSubresource()).view);

                downsampleBlurs[i]->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

                outImageInfo = RHIDescriptorImageInfoStorage(downsampleBlurs[i]->getImage().getOrCreateView(buildBasicImageSubresource()).view);

                VkDescriptorImageInfo lumImgInfo = RHIDescriptorImageInfoSample(
                    exposureImage ? 
                    exposureImage->getImage().getOrCreateView(buildBasicImageSubresource()).view :
                    getContext()->getBuiltinTextureWhite()->getSelfImage().getOrCreateView(buildBasicImageSubresource()).view);

                std::vector<VkWriteDescriptorSet> writes
                {
                    RHIPushWriteDescriptorSetImage(0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &inImageInfo),
                    RHIPushWriteDescriptorSetImage(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outImageInfo),
                    RHIPushWriteDescriptorSetImage(2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &lumImgInfo),
                    RHIPushWriteDescriptorSetBuffer(3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &frameBufferInfo)
                };

                writes[3].pImageInfo = &lumImgInfo;

                getContext()->pushDescriptorSet(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->downsamplePipe->pipelineLayout, 0, uint32_t(writes.size()), writes.data());

                pass->downsamplePipe->pushConst(cmd, &downsamplePush);

                vkCmdDispatch(cmd, getGroupCount(downsampleBlurs[i]->getImage().getExtent().width, 8), getGroupCount(downsampleBlurs[i]->getImage().getExtent().height, 8), 1);

                downsampleBlurs[i]->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            }

            pass->upscalePipe->bind(cmd);
            pass->upscalePipe->bindSet(cmd, additionalSets, 1);

            // Upscale.
            VkDescriptorImageInfo inImageCurInfo{};
            PoolImageSharedRef prevLevelUpscaleResult = nullptr;
            for (uint32_t i = 0; i < downsampleMipCount; i++)
            {
                uint32_t workMip = downsampleMipCount - i;

                uint32_t workWidth = srcHdrColorWidth >> workMip;
                uint32_t workHeight = srcHdrColorHeight >> workMip;

                const bool bLowestUpscale = (i == 0);
                const bool bHighestUpscale = (i == (downsampleMipCount - 1));

                // Prev blur result.
                inImageInfo = RHIDescriptorImageInfoSample((
                    bLowestUpscale ?
                    downsampleBlurs[downsampleMipCount - 1] : // Input from last downsample texture.
                    prevLevelUpscaleResult // Input from prev upscale result.
                    )->getImage().getOrCreateView(buildBasicImageSubresource()).view);

                inImageCurInfo = RHIDescriptorImageInfoSample((
                    bHighestUpscale ?
                    hdrSceneColor :
                    downsampleBlurs[workMip - 1]->getImage()
                    ).getOrCreateView(buildBasicImageSubresource()).view);

                auto blurX = rtPool->createPoolImage("blurX", workWidth, workHeight, hdrSceneColor.getFormat(), VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);

                outImageInfo = RHIDescriptorImageInfoStorage(blurX->getImage().getOrCreateView(buildBasicImageSubresource()).view);

                std::vector<VkWriteDescriptorSet> writes
                {
                    RHIPushWriteDescriptorSetImage(0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &inImageInfo),
                    RHIPushWriteDescriptorSetImage(1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &inImageCurInfo),
                    RHIPushWriteDescriptorSetImage(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outImageInfo),
                };

                BloomPushUpscale upscalePush{ .bBlurX = 1u, .blurRadius = setting.bloomRadius, };

                pass->upscalePipe->pushConst(cmd, &upscalePush);

                getContext()->pushDescriptorSet(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->upscalePipe->pipelineLayout, 0, uint32_t(writes.size()), writes.data());

                blurX->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                vkCmdDispatch(cmd, getGroupCount(workWidth, 8), getGroupCount(workHeight, 8), 1);
                blurX->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

                inImageInfo = RHIDescriptorImageInfoSample(blurX->getImage().getOrCreateView(buildBasicImageSubresource()).view);

                upscalePush = 
                { 
                    .bBlurX = 0u, 
                    .bFinalBlur = (bHighestUpscale ? 1u : 0u), 
                    .upscaleTime = workMip - 1,
                    .blurRadius = setting.bloomRadius,
                };
                pass->upscalePipe->pushConst(cmd, &upscalePush);

                auto blurY = rtPool->createPoolImage("blurY", workWidth, workHeight, hdrSceneColor.getFormat(), VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);

                outImageInfo = RHIDescriptorImageInfoStorage(blurY->getImage().getOrCreateView(buildBasicImageSubresource()).view);

                writes =
                {
                    RHIPushWriteDescriptorSetImage(0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &inImageInfo),
                    RHIPushWriteDescriptorSetImage(1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &inImageCurInfo),
                    RHIPushWriteDescriptorSetImage(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outImageInfo),
                };
                getContext()->pushDescriptorSet(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->upscalePipe->pipelineLayout, 0, uint32_t(writes.size()), writes.data());

                blurY->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                vkCmdDispatch(cmd, getGroupCount(workWidth, 8), getGroupCount(workHeight, 8), 1);
                blurY->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

                // Update prevUpscale result.
                prevLevelUpscaleResult = blurY;
            }
            result = prevLevelUpscaleResult;
        }

        return result;
    }
}