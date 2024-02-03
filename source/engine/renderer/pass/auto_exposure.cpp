#include "../deferred_renderer.h"
#include "../render_scene.h"
#include "../renderer.h"
#include "../scene_textures.h"

namespace engine
{
	const uint32_t kHistogramBin = 128;
	const uint32_t kHistogramThreadDim = 16;

	class ExposurePass : public PassInterface
	{
	public:


		std::unique_ptr<ComputePipeResources> averagePipe;
		std::unique_ptr<ComputePipeResources> histogramPipe;
		std::unique_ptr<ComputePipeResources> applyPipe;
	protected:
		virtual void onInit() override
		{
			{
				VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;

				getContext()->descriptorFactoryBegin()
					.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 0) // in
					.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1) // out
					.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2) // out
					.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 3) // in
					.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 4) // in
					.bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 5)// frame data
					.buildNoInfoPush(setLayout);

				std::vector<VkDescriptorSetLayout> setLayouts =
				{
					setLayout,
					m_context->getSamplerCache().getCommonDescriptorSetLayout()
				};

				ShaderVariant shaderVariant("shader/auto_exposure.glsl");
				shaderVariant.setStage(EShaderStage::eComputeShader);

				{
					auto copyVariant = shaderVariant;
					copyVariant.setMacro(L"EXPOSURE_HISTOGRAM_PASS");
					histogramPipe = std::make_unique<ComputePipeResources>(copyVariant, 0, setLayouts);
				}

				{
					auto copyVariant = shaderVariant;
					copyVariant.setMacro(L"EXPOSURE_AVERAGE_PASS");
					averagePipe = std::make_unique<ComputePipeResources>(copyVariant, 0, setLayouts);
				}
			}


			{
				VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;

				getContext()->descriptorFactoryBegin()
					.bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0)// frame data
					.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1) // in
					.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  2) // out
					.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  3) // in
					.buildNoInfoPush(setLayout);

				std::vector<VkDescriptorSetLayout> setLayouts =
				{
					setLayout,
					m_context->getSamplerCache().getCommonDescriptorSetLayout()
				};

				applyPipe = std::make_unique<ComputePipeResources>("shader/apply_exposure.glsl", 0, setLayouts);
			}
		}

		virtual void release() override
		{
			averagePipe.reset();
			histogramPipe.reset();
			applyPipe.reset();
		}
	};

	void DeferredRenderer::adaptiveExposure(
		VkCommandBuffer cmd,
		GBufferTextures* inGBuffers,
		RenderScene* scene,
		BufferParameterHandle perFrameGPU,
		const RuntimeModuleTickData& tickData)
	{
		auto* pass = getContext()->getPasses().get<ExposurePass>();
		auto* rtPool = &getContext()->getRenderTargetPools();
		auto& hdrSceneColor = inGBuffers->hdrSceneColor->getImage();

		if (!m_history.averageLum)
		{
			m_history.averageLum = rtPool->createPoolImage(
				"AverageLum",
				1,
				1,
				VK_FORMAT_R16_SFLOAT,
				VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

			VkClearColorValue exposureClear =
			{
				.float32 = {10.0f,10.0f,10.0f,10.0f}
			};
			auto rangeClear = buildBasicImageSubresource();
			m_history.averageLum->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
			vkCmdClearColorImage(cmd, m_history.averageLum->getImage().getImage(), VK_IMAGE_LAYOUT_GENERAL, &exposureClear, 1, &rangeClear);

			m_history.averageLum->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
		}
		auto averageLumCurrent = rtPool->createPoolImage(
			"AverageLumCurrent",
			1,
			1,
			VK_FORMAT_R32_SFLOAT,
			VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);

		auto histogram = rtPool->createPoolImage(
			"Histogram",
			kHistogramBin,
			1,
			VK_FORMAT_R32_UINT,
			VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

		histogram->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());


		hdrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

		PushSetBuilder setBuilder(cmd);
		setBuilder
			.addSRV(hdrSceneColor)
			.addUAV(averageLumCurrent)
			.addUAV(histogram)
			.addSRV(histogram)
			.addSRV(m_history.averageLum)
			.addBuffer(perFrameGPU)
			.push(pass->averagePipe.get());

		std::vector<VkDescriptorSet> passSets =
		{
			getContext()->getSamplerCache().getCommonDescriptorSet()
		};
		pass->averagePipe->bindSet(cmd, passSets, 1);

		{
			ScopePerframeMarker marker(cmd, "AdaptiveExposure Histogram", { 1.0f, 1.0f, 0.0f, 1.0f }, &m_gpuTimer);

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
			ScopePerframeMarker marker(cmd, "AdaptiveExposure Average", { 1.0f, 1.0f, 0.0f, 1.0f }, &m_gpuTimer);

			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->averagePipe->pipeline);
			vkCmdDispatch(cmd, 1, 1, 1);
		}
		averageLumCurrent->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

		// Update history buffer.
		m_history.averageLum = averageLumCurrent;
	}

	void engine::applyAdaptiveExposure(
		VkCommandBuffer cmd,
		GBufferTextures* inGBuffers,
		RenderScene* scene,
		BufferParameterHandle perFrameGPU,
		const RuntimeModuleTickData& tickData,
		PoolImageSharedRef eyeAdapt)
	{
		auto* pass = getContext()->getPasses().get<ExposurePass>();
		auto* rtPool = &getContext()->getRenderTargetPools();
		auto& hdrSceneColor = inGBuffers->hdrSceneColorUpscale->getImage();

		auto applyExposureImage = rtPool->createPoolImage("ApplyExposure", 
			hdrSceneColor.getExtent().width,
			hdrSceneColor.getExtent().height,
			hdrSceneColor.getFormat(),
			VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);

		applyExposureImage->getImage().transitionGeneral(cmd);
		hdrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
		if (eyeAdapt)
		{
			eyeAdapt->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
		}

		pass->applyPipe->bind(cmd);

		PushSetBuilder setBuilder(cmd);
		setBuilder
			.addBuffer(perFrameGPU)
			.addSRV(hdrSceneColor)
			.addUAV(applyExposureImage)
			.addSRV(eyeAdapt ?
				eyeAdapt->getImage():
				getContext()->getBuiltinTextureWhite()->getSelfImage())
			.push(pass->applyPipe.get());
		
		pass->applyPipe->bindSet(cmd, std::vector<VkDescriptorSet>{
			getContext()->getSamplerCache().getCommonDescriptorSet()
		}, 1);

		vkCmdDispatch(cmd, getGroupCount(hdrSceneColor.getExtent().width, 8), getGroupCount(hdrSceneColor.getExtent().height, 8), 1);

		applyExposureImage->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());


		// Just swap gbuffer.
		inGBuffers->hdrSceneColorUpscale = applyExposureImage;
	}
}