#include "../renderer_interface.h"
#include "../render_scene.h"
#include "../renderer.h"
#include "../scene_textures.h"

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
		float blurRadius;

	};

	class BloomPass : public PassInterface
	{
	public:
		VkDescriptorSetLayout setLayoutDownSample = VK_NULL_HANDLE;
		VkDescriptorSetLayout setLayoutUpscale = VK_NULL_HANDLE;

		std::unique_ptr<ComputePipeResources> downsamplePipe;
		std::unique_ptr<ComputePipeResources> upscalePipe;

	protected:
		virtual void onInit() override
		{
			getContext()->descriptorFactoryBegin()
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0) // in
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1) // out
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 2) // lum
				.bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 3)// frame data
				.buildNoInfoPush(setLayoutDownSample);

			getContext()->descriptorFactoryBegin()
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0) // inHdr
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1) // inCurHdr
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 2) // out
				.buildNoInfoPush(setLayoutUpscale);

			std::vector<VkDescriptorSetLayout> setLayoutsDown = { setLayoutDownSample, m_context->getSamplerCache().getCommonDescriptorSetLayout() };
			std::vector<VkDescriptorSetLayout> setLayoutsUp = { setLayoutUpscale, m_context->getSamplerCache().getCommonDescriptorSetLayout() };

			downsamplePipe = std::make_unique<ComputePipeResources>("shader/bloom_downsample.comp.spv", sizeof(BloomDownsample), setLayoutsDown);
			upscalePipe = std::make_unique<ComputePipeResources>("shader/bloom_upscale.comp.spv", sizeof(BloomPushUpscale), setLayoutsUp);
		}

		virtual void release() override
		{
			downsamplePipe.reset();
			upscalePipe.reset();
		}
	};

    PoolImageSharedRef RendererInterface::renderBloom(
		VkCommandBuffer cmd,
		GBufferTextures* inGBuffers,
		RenderScene* scene,
		BufferParameterHandle perFrameGPU)
	{
		auto* pass = getContext()->getPasses().get<BloomPass>();
		auto* rtPool = &m_context->getRenderTargetPools();

        auto& hdrSceneColor = inGBuffers->hdrSceneColorUpscale->getImage();
        hdrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

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
            m_context->getSamplerCache().getCommonDescriptorSet()
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
            ScopePerframeMarker marker(cmd, "Bloom Basic", { 1.0f, 1.0f, 0.0f, 1.0f });

            pass->downsamplePipe->bind(cmd);
            pass->downsamplePipe->bindSet(cmd, additionalSets, 1);

            BloomDownsample downsamplePush{};

            const auto& postProcessVolumeSetting = scene->getPostprocessVolumeSetting();

            downsamplePush.prefilterFactor = getBloomPrefilter(postProcessVolumeSetting.bloomThreshold, postProcessVolumeSetting.bloomThresholdSoft);

            auto frameBufferInfo = perFrameGPU->getBufferInfo();

            VkDescriptorImageInfo inImageInfo{};
            VkDescriptorImageInfo outImageInfo{};
            for (uint32_t i = 0; i < downsampleMipCount; i++)
            {
                const bool bFirstLevel = (i == 0);
                downsamplePush.mipLevel = i;

                inImageInfo = RHIDescriptorImageInfoSample((bFirstLevel ? hdrSceneColor : downsampleBlurs[i - 1]->getImage()).getOrCreateView(buildBasicImageSubresource()));

                downsampleBlurs[i]->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

                outImageInfo = RHIDescriptorImageInfoStorage(downsampleBlurs[i]->getImage().getOrCreateView(buildBasicImageSubresource()));

                VkDescriptorImageInfo lumImgInfo = inImageInfo;
                if (m_averageLum)
                {
                    lumImgInfo = RHIDescriptorImageInfoSample(m_averageLum->getImage().getOrCreateView(buildBasicImageSubresource()));
                }


                std::vector<VkWriteDescriptorSet> writes
                {
                    RHIPushWriteDescriptorSetImage(0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &inImageInfo),
                    RHIPushWriteDescriptorSetImage(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outImageInfo),
                    RHIPushWriteDescriptorSetImage(2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &lumImgInfo),
                    RHIPushWriteDescriptorSetBuffer(3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &frameBufferInfo)
                    
                };
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
                    )->getImage().getOrCreateView(buildBasicImageSubresource()));

                inImageCurInfo = RHIDescriptorImageInfoSample((
                    bHighestUpscale ?
                    hdrSceneColor :
                    downsampleBlurs[workMip - 1]->getImage()
                    ).getOrCreateView(buildBasicImageSubresource()));

                auto blurX = rtPool->createPoolImage("blurX", workWidth, workHeight, hdrSceneColor.getFormat(), VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);

                outImageInfo = RHIDescriptorImageInfoStorage(blurX->getImage().getOrCreateView(buildBasicImageSubresource()));

                std::vector<VkWriteDescriptorSet> writes
                {
                    RHIPushWriteDescriptorSetImage(0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &inImageInfo),
                    RHIPushWriteDescriptorSetImage(1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &inImageCurInfo),
                    RHIPushWriteDescriptorSetImage(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outImageInfo),
                };

                const auto& postProcessVolumeSetting = scene->getPostprocessVolumeSetting();

                BloomPushUpscale upscalePush{ .bBlurX = 1u, .blurRadius = postProcessVolumeSetting.bloomRadius, };

                pass->upscalePipe->pushConst(cmd, &upscalePush);

                getContext()->pushDescriptorSet(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->upscalePipe->pipelineLayout, 0, uint32_t(writes.size()), writes.data());

                blurX->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                vkCmdDispatch(cmd, getGroupCount(workWidth, 8), getGroupCount(workHeight, 8), 1);
                blurX->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

                inImageInfo = RHIDescriptorImageInfoSample(blurX->getImage().getOrCreateView(buildBasicImageSubresource()));

                upscalePush = { .bBlurX = 0u, .bFinalBlur = (bHighestUpscale ? 1u : 0u), .upscaleTime = workMip - 1,.blurRadius = postProcessVolumeSetting.bloomRadius, };
                pass->upscalePipe->pushConst(cmd, &upscalePush);

                auto blurY = rtPool->createPoolImage("blurY", workWidth, workHeight, hdrSceneColor.getFormat(), VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);

                outImageInfo = RHIDescriptorImageInfoStorage(blurY->getImage().getOrCreateView(buildBasicImageSubresource()));

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

        m_gpuTimer.getTimeStamp(cmd, "Bloom");

        return result;
	}
}