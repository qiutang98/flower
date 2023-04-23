#include <fsr2/src/ffx-fsr2-api/vk/ffx_fsr2_vk.h>
#include "fsr2.h"
#include "renderer_interface.h"
#include "renderer.h"
#include "render_scene.h"
#include "scene_textures.h"

namespace engine
{
	static AutoCVarInt32 cVarEnableFSR2RCAS("r.FSR2.RCAS", "Enable RCAS", "FSR2", 1, CVarFlags::ReadAndWrite);
	static AutoCVarFloat cVarFSR2RCASSharp("r.FSR2.RCASSharpe", "RCAS shapern", "FSR2", 0.5f, CVarFlags::ReadAndWrite);

	static AutoCVarCmd cVarFSRReset("cmd.fsr.reset", "Reset fsr.");

	static VkDeviceSize getMemoryUsageSnapshot(VkPhysicalDevice physicalDevice)
	{
		// check if VK_EXT_memory_budget is enabled
		std::vector<VkExtensionProperties> extensionProperties;

		// enumerate all the device extensions 
		uint32_t deviceExtensionCount = 0;
		vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &deviceExtensionCount, nullptr);
		extensionProperties.resize(deviceExtensionCount);

		vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &deviceExtensionCount, extensionProperties.data());

		bool extensionFound = false;

		for (uint32_t i = 0; i < deviceExtensionCount; i++)
		{
			if (strcmp(extensionProperties[i].extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0)
				extensionFound = true;
		}

		if (!extensionFound)
			return 0;

		VkDeviceSize memoryUsage = 0;

		VkPhysicalDeviceMemoryBudgetPropertiesEXT memoryBudgetProperties = {};
		memoryBudgetProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;

		VkPhysicalDeviceMemoryProperties2 memoryProperties = {};
		memoryProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
		memoryProperties.pNext = &memoryBudgetProperties;

		vkGetPhysicalDeviceMemoryProperties2(physicalDevice, &memoryProperties);

		for (uint32_t i = 0; i < memoryProperties.memoryProperties.memoryTypeCount; i++)
			memoryUsage += memoryBudgetProperties.heapUsage[i];

		return memoryUsage;
	}

	void FSR2Context::onCreateWindowSizeDependentResources(
		VkImageView input,
		VkImageView output,
		uint32_t renderWidth,
		uint32_t renderHeight,
		uint32_t displayWidth,
		uint32_t displayHeight,
		bool hdr)
	{
		// Try release first.
		onDestroyWindowSizeDependentResources();

		// Setup VK interface.
		const size_t scratchBufferSize = ffxFsr2GetScratchMemorySizeVK(getContext()->getGPU());
		void* scratchBuffer = malloc(scratchBufferSize);
		FfxErrorCode errorCode = ffxFsr2GetInterfaceVK(
			&m_initializationParameters.callbacks,
			scratchBuffer,
			scratchBufferSize,
			getContext()->getGPU(),
			vkGetDeviceProcAddr);
		FFX_ASSERT(errorCode == FFX_OK);

		m_initializationParameters.device = ffxGetDeviceVK(getContext()->getDevice());
		m_initializationParameters.maxRenderSize.width = renderWidth;
		m_initializationParameters.maxRenderSize.height = renderHeight;
		m_initializationParameters.displaySize.width = displayWidth;
		m_initializationParameters.displaySize.height = displayHeight;

		// We enable fsr2 auto exposure.
		m_initializationParameters.flags = FFX_FSR2_ENABLE_AUTO_EXPOSURE;

		// Engine always reverse z.
		const bool bReverseZ = true;
		if (bReverseZ)
		{
			m_initializationParameters.flags |= FFX_FSR2_ENABLE_DEPTH_INVERTED;
		}

		if (hdr)
		{
			m_initializationParameters.flags |= FFX_FSR2_ENABLE_HIGH_DYNAMIC_RANGE;
		}

		const uint64_t memoryUsageBefore = getMemoryUsageSnapshot(getContext()->getGPU());
		ffxFsr2ContextCreate(&m_context, &m_initializationParameters);
		const uint64_t memoryUsageAfter = getMemoryUsageSnapshot(getContext()->getGPU());
		m_memoryUsageInMegabytes = (memoryUsageAfter - memoryUsageBefore) * 0.000001f;
	}

	void FSR2Context::onDestroyWindowSizeDependentResources()
	{
		// FSR free resource require device idle.
		vkDeviceWaitIdle(getContext()->getDevice());

		// Only destroy contexts which are live
		if (m_initializationParameters.callbacks.scratchBuffer != nullptr)
		{
			ffxFsr2ContextDestroy(&m_context);
			free(m_initializationParameters.callbacks.scratchBuffer);
			m_initializationParameters.callbacks.scratchBuffer = nullptr;
		}
	}

	void FSR2Context::draw(VkCommandBuffer commandBuffer, const FfxUpscaleSetup& cameraSetup)
	{
		FfxFsr2DispatchDescription dispatchParameters = {};

		static wchar_t inputColorName[] = L"FSR2_InputColor";

		dispatchParameters.commandList = ffxGetCommandListVK(commandBuffer);
		dispatchParameters.color = ffxGetTextureResourceVK(
			&m_context,
			cameraSetup.unresolvedColorResource->getImage().getImage(),
			cameraSetup.unresolvedColorResourceView,
			cameraSetup.unresolvedColorResource->getImage().getExtent().width,
			cameraSetup.unresolvedColorResource->getImage().getExtent().height,
			cameraSetup.unresolvedColorResource->getImage().getFormat(),
			inputColorName);

		static wchar_t inputDepthName[] = L"FSR2_InputDepth";
		dispatchParameters.depth = ffxGetTextureResourceVK(
			&m_context,
			cameraSetup.depthbufferResource->getImage().getImage(),
			cameraSetup.depthbufferResourceView,
			cameraSetup.depthbufferResource->getImage().getExtent().width,
			cameraSetup.depthbufferResource->getImage().getExtent().height,
			cameraSetup.depthbufferResource->getImage().getFormat(),
			inputDepthName);

		static wchar_t inputMotionName[] = L"FSR2_InputMotionVectors";
		dispatchParameters.motionVectors = ffxGetTextureResourceVK(
			&m_context,
			cameraSetup.motionvectorResource->getImage().getImage(),
			cameraSetup.motionvectorResourceView,
			cameraSetup.motionvectorResource->getImage().getExtent().width,
			cameraSetup.motionvectorResource->getImage().getExtent().height,
			cameraSetup.motionvectorResource->getImage().getFormat(),
			inputMotionName);

		// Ref:Exposure: a value which is multiplied against the result of the pre - exposed color value.
		static wchar_t inputExposureName[] = L"FSR2_InputExposure";
		dispatchParameters.exposure = ffxGetTextureResourceVK(
			&m_context,
			nullptr,
			nullptr,
			1,
			1,
			VK_FORMAT_UNDEFINED,
			inputExposureName);

		if ((config.reactiveMaskMode == EReactiveMaskMode::On) || (config.reactiveMaskMode == EReactiveMaskMode::AutoGen))
		{
			static wchar_t inputReactiveMapName[] = L"FSR2_InputExposure";
			dispatchParameters.reactive = ffxGetTextureResourceVK(
				&m_context,
				cameraSetup.reactiveMapResource->getImage().getImage(),
				cameraSetup.reactiveMapResourceView,
				cameraSetup.reactiveMapResource->getImage().getExtent().width,
				cameraSetup.reactiveMapResource->getImage().getExtent().height,
				cameraSetup.reactiveMapResource->getImage().getFormat(),
				inputReactiveMapName);
		}
		else
		{
			static wchar_t inputReactiveMapEmptyName[] = L"FSR2_EmptyInputReactiveMap";
			dispatchParameters.reactive = ffxGetTextureResourceVK(
				&m_context,
				nullptr,
				nullptr,
				1,
				1,
				VK_FORMAT_UNDEFINED,
				inputReactiveMapEmptyName);
		}

		if (config.bCompositionMask == true)
		{
			CHECK(cameraSetup.transparencyAndCompositionResource != nullptr);
			CHECK(cameraSetup.transparencyAndCompositionResourceView != VK_NULL_HANDLE);

			static wchar_t inputTransparencyAndCompositionName[] = L"FSR2_TransparencyAndCompositionMap";
			dispatchParameters.transparencyAndComposition = ffxGetTextureResourceVK(
				&m_context,
				cameraSetup.transparencyAndCompositionResource->getImage().getImage(),
				cameraSetup.transparencyAndCompositionResourceView,
				cameraSetup.transparencyAndCompositionResource->getImage().getExtent().width,
				cameraSetup.transparencyAndCompositionResource->getImage().getExtent().height,
				cameraSetup.transparencyAndCompositionResource->getImage().getFormat(),
				inputTransparencyAndCompositionName);
		}
		else
		{
			static wchar_t inputEmptyTransparencyAndCompositionName[] = L"FSR2_EmptyTransparencyAndCompositionMap";
			dispatchParameters.transparencyAndComposition = ffxGetTextureResourceVK(
				&m_context,
				nullptr,
				nullptr,
				1,
				1,
				VK_FORMAT_UNDEFINED,
				inputEmptyTransparencyAndCompositionName);
		}

		static wchar_t inputOutputUpscaledColorName[] = L"FSR2_OutputUpscaledColor";
		dispatchParameters.output = ffxGetTextureResourceVK(
			&m_context,
			cameraSetup.resolvedColorResource->getImage().getImage(),
			cameraSetup.resolvedColorResourceView,
			cameraSetup.resolvedColorResource->getImage().getExtent().width,
			cameraSetup.resolvedColorResource->getImage().getExtent().height,
			cameraSetup.resolvedColorResource->getImage().getFormat(),
			inputOutputUpscaledColorName,
			FFX_RESOURCE_STATE_UNORDERED_ACCESS);

		dispatchParameters.jitterOffset.x = config.jitterX;
		dispatchParameters.jitterOffset.y = config.jitterY;
		dispatchParameters.motionVectorScale.x = (float)config.renderWidth;
		dispatchParameters.motionVectorScale.y = (float)config.renderHeight;
		dispatchParameters.reset = config.bCameraReset;
		dispatchParameters.enableSharpening = config.bUseRcas;
		dispatchParameters.sharpness = config.sharpening;
		dispatchParameters.frameTimeDelta = (float)config.deltaTime;
		dispatchParameters.renderSize.width = config.renderWidth;
		dispatchParameters.renderSize.height = config.renderHeight;
		dispatchParameters.cameraFar = config.farPlane;
		dispatchParameters.cameraNear = config.nearPlane;
		dispatchParameters.cameraFovAngleVertical = config.fovV;

		// Pre exposure value used for FSR hdr input color mapping.
		// Ref: a value by which we divide the input signal to get back to the original signal produced by the game before any packing into lower precision render targets.
		dispatchParameters.preExposure = 1.0f;


		FfxErrorCode errorCode = ffxFsr2ContextDispatch(&m_context, &dispatchParameters);
		FFX_ASSERT(errorCode == FFX_OK);
	}

	void FSR2Context::generateReactiveMask(VkCommandBuffer pCommandList, const FfxUpscaleSetup& cameraSetup)
	{
		FfxFsr2GenerateReactiveDescription generateReactiveParameters;
		generateReactiveParameters.commandList = ffxGetCommandListVK(pCommandList);

		static wchar_t inputOpaqueOnlyColor[] = L"FSR2_OpaqueOnlyColorResource";
		generateReactiveParameters.colorOpaqueOnly = ffxGetTextureResourceVK(
			&m_context,
			cameraSetup.opaqueOnlyColorResource->getImage().getImage(),
			cameraSetup.opaqueOnlyColorResourceView,
			cameraSetup.opaqueOnlyColorResource->getImage().getExtent().width,
			cameraSetup.opaqueOnlyColorResource->getImage().getExtent().height,
			cameraSetup.opaqueOnlyColorResource->getImage().getFormat(),
			inputOpaqueOnlyColor
		);

		static wchar_t inputUnresolvedColorColor[] = L"FSR2_UnresolvedColorResource";
		generateReactiveParameters.colorPreUpscale = ffxGetTextureResourceVK(
			&m_context,
			cameraSetup.unresolvedColorResource->getImage().getImage(),
			cameraSetup.unresolvedColorResourceView,
			cameraSetup.unresolvedColorResource->getImage().getExtent().width,
			cameraSetup.unresolvedColorResource->getImage().getExtent().height,
			cameraSetup.unresolvedColorResource->getImage().getFormat(),
			inputUnresolvedColorColor);

		static wchar_t inputReactiveMapColor[] = L"FSR2_InputReactiveMap";
		generateReactiveParameters.outReactive = ffxGetTextureResourceVK(
			&m_context,
			cameraSetup.reactiveMapResource->getImage().getImage(),
			cameraSetup.reactiveMapResourceView,
			cameraSetup.reactiveMapResource->getImage().getExtent().width,
			cameraSetup.reactiveMapResource->getImage().getExtent().height,
			cameraSetup.reactiveMapResource->getImage().getFormat(),
			inputReactiveMapColor,
			FFX_RESOURCE_STATE_GENERIC_READ);

		generateReactiveParameters.renderSize.width = config.renderWidth;
		generateReactiveParameters.renderSize.height = config.renderHeight;
		generateReactiveParameters.scale = config.fFsr2AutoReactiveScale;
		generateReactiveParameters.cutoffThreshold = config.fFsr2AutoReactiveThreshold;
		generateReactiveParameters.binaryValue = config.fFsr2AutoReactiveBinaryValue;

		generateReactiveParameters.flags =
			(config.bFsr2AutoReactiveTonemap ? FFX_FSR2_AUTOREACTIVEFLAGS_APPLY_TONEMAP : 0)
			| (config.bFsr2AutoReactiveInverseTonemap ? FFX_FSR2_AUTOREACTIVEFLAGS_APPLY_INVERSETONEMAP : 0)
			| (config.bFsr2AutoReactiveThreshold ? FFX_FSR2_AUTOREACTIVEFLAGS_APPLY_THRESHOLD : 0)
			| (config.bFsr2AutoReactiveUseMax ? FFX_FSR2_AUTOREACTIVEFLAGS_USE_COMPONENTS_MAX : 0);

		ffxFsr2ContextGenerateReactiveMask(&m_context, &generateReactiveParameters);
	}

	void RendererInterface::renderFSR2(VkCommandBuffer cmd, GBufferTextures* inGBuffers, RenderScene* scene, BufferParameterHandle perFrameGPU, const RuntimeModuleTickData& tickData)
	{


		// Config setup.
		auto& config = m_fsr2->config;
		{
			// Basic config.
			config.jitterX = m_cacheGPUPerFrameData.jitterData.x;
			config.jitterY = m_cacheGPUPerFrameData.jitterData.y;
			config.renderWidth = m_renderWidth;
			config.renderHeight = m_renderHeight;
			config.nearPlane = m_cacheGPUPerFrameData.camInfo.z;
			config.farPlane = m_cacheGPUPerFrameData.camInfo.w;
			config.fovV = m_cacheGPUPerFrameData.camInfo.x;
			config.deltaTime = tickData.deltaTime;

			bool bForceCameraCut = false;
			CVarCmdHandle(cVarFSRReset, [&]() { bForceCameraCut = true; });

			// When camera cut, tick count reset 0.
			config.bCameraReset = bForceCameraCut || (m_tickCount == 0);

			// RCAS config.
			{
				config.bUseRcas = cVarEnableFSR2RCAS.get() > 0;
				config.sharpening = glm::clamp(cVarFSR2RCASSharp.get(), 0.0f, 1.0f);
			}

			//////////////////////////////////////////////////////////

			// ReactiveMask mode.
			{
				config.reactiveMaskMode = EReactiveMaskMode::On;
				config.fFsr2AutoReactiveScale = 1.f;
				config.fFsr2AutoReactiveThreshold = 0.2f;
				config.fFsr2AutoReactiveBinaryValue = 0.9f;
				config.bFsr2AutoReactiveTonemap = true;
				config.bFsr2AutoReactiveInverseTonemap = false;
				config.bFsr2AutoReactiveThreshold = true;
				config.bFsr2AutoReactiveUseMax = true;
			}

			// Translucency composition mask.
			config.bCompositionMask = true;



			// Test?
			config.lodTextureBasicBias = 0.0f;
		}

		ScopePerframeMarker marker(cmd, "FSR2.1", { 1.0f, 0.0f, 0.0f, 1.0f });
		{
			//////////////////////////////////////////////////////////////////
			auto& hdrSceneColor = inGBuffers->hdrSceneColor->getImage();
			auto& gbufferV = inGBuffers->gbufferV->getImage();
			auto& sceneDepthZ = inGBuffers->depthTexture->getImage();
			auto& displayOut = inGBuffers->hdrSceneColorUpscale->getImage();

			FfxUpscaleSetup upscaleSetup;
			upscaleSetup.unresolvedColorResource = inGBuffers->hdrSceneColor;
			upscaleSetup.unresolvedColorResourceView = hdrSceneColor.getOrCreateView(buildBasicImageSubresource());
			upscaleSetup.motionvectorResource = inGBuffers->gbufferV;
			upscaleSetup.motionvectorResourceView = gbufferV.getOrCreateView(buildBasicImageSubresource());
			upscaleSetup.depthbufferResource = inGBuffers->depthTexture;
			upscaleSetup.depthbufferResourceView = sceneDepthZ.getOrCreateView(RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));
			upscaleSetup.resolvedColorResource = inGBuffers->hdrSceneColorUpscale;
			upscaleSetup.resolvedColorResourceView = displayOut.getOrCreateView(buildBasicImageSubresource());

			upscaleSetup.reactiveMapResource = inGBuffers->gbufferUpscaleReactive;
			upscaleSetup.reactiveMapResourceView = inGBuffers->gbufferUpscaleReactive->getImage().getOrCreateView(buildBasicImageSubresource());

			upscaleSetup.transparencyAndCompositionResource = inGBuffers->gbufferUpscaleTranslucencyAndComposition;
			upscaleSetup.transparencyAndCompositionResourceView = inGBuffers->gbufferUpscaleTranslucencyAndComposition->getImage().getOrCreateView(buildBasicImageSubresource());

			// If FSR2 and auto reactive mask is enabled: generate reactive mask
			if (config.reactiveMaskMode == EReactiveMaskMode::AutoGen)
			{
				m_fsr2->generateReactiveMask(cmd, upscaleSetup);
			}

			// FSR2 upscale.
			{
				sceneDepthZ.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));
				gbufferV.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
				hdrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
				displayOut.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
				m_fsr2->draw(cmd, upscaleSetup);
			}
		}

		m_gpuTimer.getTimeStamp(cmd, "FSR2.1");
	}

}