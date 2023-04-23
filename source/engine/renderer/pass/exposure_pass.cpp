#include "../renderer_interface.h"
#include "../render_scene.h"
#include "../renderer.h"
#include "../scene_textures.h"

namespace engine
{
	const uint32_t kHistogramBin = 128;
	const uint32_t kHistogramThreadDim = 16;

	struct AdaptiveExposurePush
	{
		float scale;
		float offset;
		float lowPercent;
		float highPercent;
		float minBrightness;
		float maxBrightness;
		float speedDown;
		float speedUp;
		float exposureCompensation;
		float deltaTime;
	};

	class ExposurePass : public PassInterface
	{
	public:
		VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;

		std::unique_ptr<ComputePipeResources> averagePipe;
		std::unique_ptr<ComputePipeResources> histogramPipe;

	protected:
		virtual void onInit() override
		{
			getContext()->descriptorFactoryBegin()
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0) // in
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1) // out
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 2) // out
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 3) // in
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 4) // in
				.bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 5)// frame data
				.buildNoInfoPush(setLayout);

			std::vector<VkDescriptorSetLayout> setLayouts = { setLayout, m_context->getSamplerCache().getCommonDescriptorSetLayout() };

			averagePipe = std::make_unique<ComputePipeResources>("shader/exposure_average.comp.spv", sizeof(AdaptiveExposurePush), setLayouts);
			histogramPipe = std::make_unique<ComputePipeResources>("shader/exposure_histogram.comp.spv", sizeof(AdaptiveExposurePush), setLayouts);
		}

		virtual void release() override
		{
			averagePipe.reset();
			histogramPipe.reset();
		}
	};

	void RendererInterface::adaptiveExposure(
		VkCommandBuffer cmd,
		GBufferTextures* inGBuffers,
		RenderScene* scene,
		BufferParameterHandle perFrameGPU,
		const RuntimeModuleTickData& tickData)
	{
		auto* pass = getContext()->getPasses().get<ExposurePass>();
		auto* rtPool = &m_context->getRenderTargetPools();

		if (!m_averageLum)
		{
			m_averageLum = rtPool->createPoolImage(
				"AverageLum",
				1,
				1,
				VK_FORMAT_R16_SFLOAT,
				VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

			VkClearColorValue exposureClear =
			{
				.float32 = { 10.0f, 10.0f, 10.0f, 10.0f}
			};
			auto rangeClear = buildBasicImageSubresource();
			m_averageLum->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
			vkCmdClearColorImage(cmd, m_averageLum->getImage().getImage(), VK_IMAGE_LAYOUT_GENERAL, &exposureClear, 1, &rangeClear);
			m_averageLum->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
		}
		auto averageLumCurrent = rtPool->createPoolImage(
			"AverageLumCurrent",
			1,
			1,
			VK_FORMAT_R16_SFLOAT,
			VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);

		auto histogram = rtPool->createPoolImage(
			"Histogram",
			kHistogramBin,
			1,
			VK_FORMAT_R32_UINT,
			VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);


		histogram->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

		auto& hdrSceneColor = inGBuffers->hdrSceneColorUpscale->getImage();
		hdrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

		PushSetBuilder setBuilder(cmd);
		setBuilder
			.addSRV(hdrSceneColor)
			.addUAV(averageLumCurrent)
			.addUAV(histogram)
			.addSRV(histogram)
			.addSRV(m_averageLum)
			.addBuffer(perFrameGPU)
			.push(pass->averagePipe.get());

		std::vector<VkDescriptorSet> passSets =
		{
			m_context->getSamplerCache().getCommonDescriptorSet()
		};
		pass->averagePipe->bindSet(cmd, passSets, 1);

		const float maxEv = 9.0f;
		const float minEv = -9.0f;
		const float diff = maxEv - minEv;
		const float scale = 1.0f / diff;
		const float offset = -minEv * scale;

		const auto& postProcessVolumeSetting = scene->getPostprocessVolumeSetting();

		AdaptiveExposurePush pushConst
		{
			.scale = scale,
			.offset = offset,

			.lowPercent = math::clamp(postProcessVolumeSetting.autoExposureLowPercent, 0.01f, 0.99f),
			.highPercent = math::clamp(postProcessVolumeSetting.autoExposureHighPercent, 0.01f, 0.99f),
			.minBrightness = math::exp2(postProcessVolumeSetting.autoExposureMinBrightness),
			.maxBrightness = math::exp2(postProcessVolumeSetting.autoExposureMaxBrightness),
			.speedDown = postProcessVolumeSetting.autoExposureSpeedDown,
			.speedUp = postProcessVolumeSetting.autoExposureSpeedUp,
			.exposureCompensation = postProcessVolumeSetting.autoExposureExposureCompensation,
			.deltaTime = tickData.smoothDeltaTime,
		};
		vkCmdPushConstants(cmd, pass->histogramPipe->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConst), &pushConst);

		{
			ScopePerframeMarker marker(cmd, "AdaptiveExposure Histogram", { 1.0f, 1.0f, 0.0f, 1.0f });

			VkClearColorValue zeroClear = { .uint32 = {0,0,0,0} };

			auto rangeClear = buildBasicImageSubresource();
			vkCmdClearColorImage(cmd, histogram->getImage().getImage(), VK_IMAGE_LAYOUT_GENERAL, &zeroClear, 1, &rangeClear);
			VkImageMemoryBarrier2 clearBarrier =
			{
				.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
				.srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT,
				.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
				.dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
				.oldLayout = VK_IMAGE_LAYOUT_GENERAL,
				.newLayout = VK_IMAGE_LAYOUT_GENERAL,
				.image = histogram->getImage().getImage(),
				.subresourceRange = rangeClear
			};
			RHIPipelineBarrier(cmd, 0, 0, nullptr, 1, &clearBarrier);

			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->histogramPipe->pipeline);

			vkCmdDispatch(cmd,
				getGroupCount(hdrSceneColor.getExtent().width / 3 + 1, kHistogramThreadDim),
				getGroupCount(hdrSceneColor.getExtent().height / 3 + 1, kHistogramThreadDim), 1);

			histogram->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
		}

		averageLumCurrent->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
		{
			ScopePerframeMarker marker(cmd, "AdaptiveExposure Average", { 1.0f, 1.0f, 0.0f, 1.0f });

			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->averagePipe->pipeline);
			vkCmdDispatch(cmd, 1, 1, 1);
		}
		averageLumCurrent->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

		m_gpuTimer.getTimeStamp(cmd, "AdaptiveExposure");

		// Update history buffer.
		m_averageLum = averageLumCurrent;
	}
}