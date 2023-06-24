#include "scene_textures.h"
#include "renderer_interface.h"
#include "../../editor/editor.h"

namespace engine
{
	GBufferTextures GBufferTextures::build(RendererInterface* renderer, VulkanContext* context)
	{
		auto& pool = context->getRenderTargetPools();

		uint32_t renderWidth = renderer->getRenderWidth();
		uint32_t renderHeight = renderer->getRenderHeight();

		const auto kGBufferUsage = 
			VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | 
			VK_IMAGE_USAGE_STORAGE_BIT |
			VK_IMAGE_USAGE_SAMPLED_BIT |
			VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		const auto kDepthUsage =
			VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
			VK_IMAGE_USAGE_SAMPLED_BIT |
			VK_IMAGE_USAGE_TRANSFER_DST_BIT;

		GBufferTextures result { };

		result.hdrSceneColor = pool.createPoolImage("HdrSceneColor", renderWidth, renderHeight, hdrSceneColorFormat(), kGBufferUsage);
		result.hdrSceneColorUpscale = pool.createPoolImage("HdrSceneColorUpscale", renderer->getDisplayWidth(), renderer->getDisplayHeight(), hdrSceneColorFormat(), kGBufferUsage);
		result.depthTexture = pool.createPoolImage("DepthTexture", renderWidth, renderHeight, depthTextureFormat(), kDepthUsage);

		result.gbufferA = pool.createPoolImage("GBufferA", renderWidth, renderHeight, gbufferAFormat(), kGBufferUsage);
		result.gbufferB = pool.createPoolImage("GBufferB", renderWidth, renderHeight, gbufferBFormat(), kGBufferUsage);
		result.gbufferS = pool.createPoolImage("GBufferS", renderWidth, renderHeight, gbufferSFormat(), kGBufferUsage);
		result.gbufferV = pool.createPoolImage("GBufferV", renderWidth, renderHeight, gbufferVFormat(), kGBufferUsage);
		result.gbufferUpscaleTranslucencyAndComposition = pool.createPoolImage("gbufferUpscaleTranslucencyAndComposition", renderWidth, renderHeight, gbufferUpscaleTranslucencyAndCompositionFormat(), kGBufferUsage);
		result.gbufferUpscaleReactive = pool.createPoolImage("gbufferUpscaleReactive", renderWidth, renderHeight, gbufferUpscaleReactiveFormat(), kGBufferUsage);
		result.idTexture = pool.createPoolImage("IdTexture", renderWidth, renderHeight, getIdTextureFormat(), kGBufferUsage);

		uint32_t selectMaskWidth = renderWidth;
		uint32_t selectMaskHeight = renderHeight;
		if (Editor::get()->getSceneNodeSelected().empty())
		{
			selectMaskWidth = 1;
			selectMaskHeight = 1;
		}
		result.selectionOutlineMask = pool.createPoolImage("SelectionOutlineMask", renderWidth, renderHeight, gbufferSelectionOutlineMaskFormat(), kGBufferUsage);

		return result;
	}

	void GBufferTextures::clearValue(VkCommandBuffer graphicsCmd)
	{
		auto& hdrSceneColor = this->hdrSceneColor->getImage();
		auto& gbufferA = this->gbufferA->getImage();
		auto& gbufferB = this->gbufferB->getImage();
		auto& gbufferS = this->gbufferS->getImage();
		auto& gbufferV = this->gbufferV->getImage();
		auto& gbufferComposition = this->gbufferUpscaleTranslucencyAndComposition->getImage();
		auto& gbufferUpscaleMask = this->gbufferUpscaleReactive->getImage();
		auto& idTexture = this->idTexture->getImage();
		auto& selectMask = this->selectionOutlineMask->getImage();

		// Depth clear in mesh draw pass.
		auto& depthZ = this->depthTexture->getImage();

		ScopePerframeMarker marker(graphicsCmd, "GBuffer Clear", { 1.0f, 1.0f, 0.0f, 1.0f });
		VkClearColorValue zeroClear =
		{
			.uint32 = {0,0,0,0}
		};

		hdrSceneColor.transitionLayout(graphicsCmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
		gbufferA.transitionLayout(graphicsCmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
		gbufferB.transitionLayout(graphicsCmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
		gbufferS.transitionLayout(graphicsCmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
		gbufferV.transitionLayout(graphicsCmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
		gbufferComposition.transitionLayout(graphicsCmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
		gbufferUpscaleMask.transitionLayout(graphicsCmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
		idTexture.transitionLayout(graphicsCmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
		selectMask.transitionLayout(graphicsCmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
		depthZ.transitionLayout(graphicsCmd, VK_IMAGE_LAYOUT_GENERAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));

		auto rangeClear = buildBasicImageSubresource();
		vkCmdClearColorImage(graphicsCmd, hdrSceneColor.getImage(), VK_IMAGE_LAYOUT_GENERAL, &zeroClear, 1, &rangeClear);
		vkCmdClearColorImage(graphicsCmd, gbufferA.getImage(), VK_IMAGE_LAYOUT_GENERAL, &zeroClear, 1, &rangeClear);
		vkCmdClearColorImage(graphicsCmd, gbufferB.getImage(), VK_IMAGE_LAYOUT_GENERAL, &zeroClear, 1, &rangeClear);
		vkCmdClearColorImage(graphicsCmd, gbufferS.getImage(), VK_IMAGE_LAYOUT_GENERAL, &zeroClear, 1, &rangeClear);
		vkCmdClearColorImage(graphicsCmd, gbufferV.getImage(), VK_IMAGE_LAYOUT_GENERAL, &zeroClear, 1, &rangeClear);
		vkCmdClearColorImage(graphicsCmd, gbufferComposition.getImage(), VK_IMAGE_LAYOUT_GENERAL, &zeroClear, 1, &rangeClear);
		vkCmdClearColorImage(graphicsCmd, gbufferUpscaleMask.getImage(), VK_IMAGE_LAYOUT_GENERAL, &zeroClear, 1, &rangeClear);
		vkCmdClearColorImage(graphicsCmd, idTexture.getImage(), VK_IMAGE_LAYOUT_GENERAL, &zeroClear, 1, &rangeClear);
		vkCmdClearColorImage(graphicsCmd, selectMask.getImage(), VK_IMAGE_LAYOUT_GENERAL, &zeroClear, 1, &rangeClear);

		auto depthClear = VkClearDepthStencilValue{ 0.0f, 1 };
		auto rangeClearDepth = RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT);
		vkCmdClearDepthStencilImage(graphicsCmd, depthZ.getImage(), VK_IMAGE_LAYOUT_GENERAL, &depthClear, 1, &rangeClearDepth);

		hdrSceneColor.transitionLayout(graphicsCmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
		gbufferA.transitionLayout(graphicsCmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
		gbufferB.transitionLayout(graphicsCmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
		gbufferS.transitionLayout(graphicsCmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
		gbufferV.transitionLayout(graphicsCmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
		gbufferComposition.transitionLayout(graphicsCmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
		gbufferUpscaleMask.transitionLayout(graphicsCmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
		depthZ.transitionLayout(graphicsCmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, rangeClearDepth);
		idTexture.transitionLayout(graphicsCmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
		selectMask.transitionLayout(graphicsCmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
	}

	SharedTextures::SharedTextures()
	{
		auto cmd = getContext()->createMajorGraphicsCommandBuffer();
		RHICheck(vkResetCommandBuffer(cmd, 0));
		VkCommandBufferBeginInfo cmdBeginInfo = RHICommandbufferBeginInfo(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
		RHICheck(vkBeginCommandBuffer(cmd, &cmdBeginInfo));
		{
			compute(cmd);
		}
		RHICheck(vkEndCommandBuffer(cmd));
		VkPipelineStageFlags waitFlags = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
		RHISubmitInfo cmdSubmitInfo{};
		cmdSubmitInfo.setWaitStage(&waitFlags).setCommandBuffer(&cmd, 1);
		std::vector<VkSubmitInfo> infosRawSubmit{ cmdSubmitInfo };
		getContext()->submitNoFence((uint32_t)infosRawSubmit.size(), infosRawSubmit.data());

		getContext()->waitDeviceIdle();
	}

	void TemporalBlueNoise::BufferMisc::buildSet()
	{
		VkDescriptorBufferInfo sobolInfo{};
		sobolInfo.buffer = sobolBuffer->getVkBuffer();
		sobolInfo.offset = 0;
		sobolInfo.range = sobolBuffer->getSize();

		VkDescriptorBufferInfo rankingTileInfo{};
		rankingTileInfo.buffer = rankingTileBuffer->getVkBuffer();
		rankingTileInfo.offset = 0;
		rankingTileInfo.range = rankingTileBuffer->getSize();

		VkDescriptorBufferInfo scramblingTileInfo{};
		scramblingTileInfo.buffer = scramblingTileBuffer->getVkBuffer();
		scramblingTileInfo.offset = 0;
		scramblingTileInfo.range = scramblingTileBuffer->getSize();

		getContext()->descriptorFactoryBegin()
			.bindBuffers(0, 1, &sobolInfo, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCommonShaderStage)
			.bindBuffers(1, 1, &rankingTileInfo, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCommonShaderStage)
			.bindBuffers(2, 1, &scramblingTileInfo, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCommonShaderStage)
			.build(set, setLayouts);
	}

	namespace blueNoise_256_Spp
	{
		// blue noise sampler 256spp.
		#include <util/samplerCPP/samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_256spp.cpp>
	}

	namespace blueNoise_128_Spp
	{
		// blue noise sampler 128spp.
		#include <util/samplerCPP/samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_128spp.cpp>
	}
	namespace blueNoise_64_Spp
	{
		// blue noise sampler 64spp.
		#include <util/samplerCPP/samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_64spp.cpp>
	}
	namespace blueNoise_32_Spp
	{
		// blue noise sampler 32spp.
		#include <util/samplerCPP/samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_32spp.cpp>
	}

	namespace blueNoise_16_Spp
	{
		// blue noise sampler 16spp.
		#include <util/samplerCPP/samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_16spp.cpp>
	}

	namespace blueNoise_8_Spp
	{
		// blue noise sampler 8spp.
		#include <util/samplerCPP/samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_8spp.cpp>
	}

	namespace blueNoise_4_Spp
	{
		// blue noise sampler 4spp.
		#include <util/samplerCPP/samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_4spp.cpp>
	}

	namespace blueNoise_2_Spp
	{
		// blue noise sampler 2spp.
		#include <util/samplerCPP/samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_2spp.cpp>
	}

	namespace blueNoise_1_Spp
	{
		// blue noise sampler 1spp.
		#include <util/samplerCPP/samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_1spp.cpp>
	}

	TemporalBlueNoise::TemporalBlueNoise()
	{
		auto buildBuffer = [](const char* name, void* ptr, VkDeviceSize size)
		{
			return std::make_unique<VulkanBuffer>(
				getContext(),
				name,
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
				VulkanBuffer::getStageCopyForUploadBufferFlags(),
				size,
				ptr
			);
		};

		#define WORK_CODE \
		{\
			BlueNoiseWorkingBuffer.sobolBuffer = buildBuffer(BlueNoiseWorkingName, (void*)BlueNoiseWorkingSpace::sobol_256spp_256d, sizeof(BlueNoiseWorkingSpace::sobol_256spp_256d));\
			BlueNoiseWorkingBuffer.rankingTileBuffer = buildBuffer(BlueNoiseWorkingName, (void*)BlueNoiseWorkingSpace::rankingTile, sizeof(BlueNoiseWorkingSpace::rankingTile));\
			BlueNoiseWorkingBuffer.scramblingTileBuffer = buildBuffer(BlueNoiseWorkingName, (void*)BlueNoiseWorkingSpace::scramblingTile, sizeof(BlueNoiseWorkingSpace::scramblingTile));\
			BlueNoiseWorkingBuffer.buildSet();\
		}

		#define BlueNoiseWorkingBuffer spp_1_buffer
		#define BlueNoiseWorkingSpace blueNoise_1_Spp
		#define BlueNoiseWorkingName "Sobel_1_spp_buffer"
			WORK_CODE
		#undef BlueNoiseWorkingBuffer
		#undef BlueNoiseWorkingSpace 
		#undef BlueNoiseWorkingName

		#define BlueNoiseWorkingBuffer spp_2_buffer
		#define BlueNoiseWorkingSpace blueNoise_2_Spp
		#define BlueNoiseWorkingName "Sobel_2_spp_buffer"
			WORK_CODE
		#undef BlueNoiseWorkingBuffer
		#undef BlueNoiseWorkingSpace 
		#undef BlueNoiseWorkingName

		#define BlueNoiseWorkingBuffer spp_4_buffer
		#define BlueNoiseWorkingSpace blueNoise_4_Spp
		#define BlueNoiseWorkingName "Sobel_4_spp_buffer"
			WORK_CODE
		#undef BlueNoiseWorkingBuffer
		#undef BlueNoiseWorkingSpace 
		#undef BlueNoiseWorkingName

		#define BlueNoiseWorkingBuffer spp_8_buffer
		#define BlueNoiseWorkingSpace blueNoise_8_Spp
		#define BlueNoiseWorkingName "Sobel_8_spp_buffer"
			WORK_CODE
		#undef BlueNoiseWorkingBuffer
		#undef BlueNoiseWorkingSpace 
		#undef BlueNoiseWorkingName

		#define BlueNoiseWorkingBuffer spp_16_buffer
		#define BlueNoiseWorkingSpace blueNoise_16_Spp
		#define BlueNoiseWorkingName "Sobel_16_spp_buffer"
			WORK_CODE
		#undef BlueNoiseWorkingBuffer
		#undef BlueNoiseWorkingSpace 
		#undef BlueNoiseWorkingName

		#define BlueNoiseWorkingBuffer spp_32_buffer
		#define BlueNoiseWorkingSpace blueNoise_32_Spp
		#define BlueNoiseWorkingName "Sobel_32_spp_buffer"
			WORK_CODE
		#undef BlueNoiseWorkingBuffer
		#undef BlueNoiseWorkingSpace 
		#undef BlueNoiseWorkingName


		#define BlueNoiseWorkingBuffer spp_64_buffer
		#define BlueNoiseWorkingSpace blueNoise_64_Spp
		#define BlueNoiseWorkingName "Sobel_64_spp_buffer"
			WORK_CODE
		#undef BlueNoiseWorkingBuffer
		#undef BlueNoiseWorkingSpace 
		#undef BlueNoiseWorkingName

		#define BlueNoiseWorkingBuffer spp_128_buffer
		#define BlueNoiseWorkingSpace blueNoise_128_Spp
		#define BlueNoiseWorkingName "Sobel_128_spp_buffer"
			WORK_CODE
		#undef BlueNoiseWorkingBuffer
		#undef BlueNoiseWorkingSpace 
		#undef BlueNoiseWorkingName

		#define BlueNoiseWorkingBuffer spp_256_buffer
		#define BlueNoiseWorkingSpace blueNoise_256_Spp
		#define BlueNoiseWorkingName "Sobel_256_spp_buffer"
			WORK_CODE
		#undef BlueNoiseWorkingBuffer
		#undef BlueNoiseWorkingSpace 
		#undef BlueNoiseWorkingName

		#undef WORK_CODE
	}
}