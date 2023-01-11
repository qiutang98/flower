#include "Pch.h"
#include "RHI.h"

#include <vma/vk_mem_alloc.h>

namespace Flower
{
	static AutoCVarInt32 cVarVulkanOpenValidation(
		"r.RHI.OpenValidation",
		"Enable vulkan validation layer.0 is off,1 is on.",
		"RHI",
		1,
		CVarFlags::ReadOnly | CVarFlags::InitOnce
	);

	static AutoCVarInt32 cVarVulkanOpenRayTrace(
		"r.RHI.OpenRayTrace",
		"Enable vulkan raytrace.0 is off,1 is on.",
		"RHI",
		0, // NOTE: Current my desktop computer graphics still dont support RTX, temporal disable.
		   //       
		CVarFlags::ReadOnly | CVarFlags::InitOnce
	);

	size_t RHI::GMaxSwapchainCount = ~0;

	bool RHI::bSupportRayTrace = false;

	VkPhysicalDevice RHI::GPU    = VK_NULL_HANDLE;
	VkDevice         RHI::Device = VK_NULL_HANDLE;
	VmaAllocator     RHI::VMA    = VK_NULL_HANDLE;

	SamplerCache* RHI::SamplerManager = nullptr;
	ShaderCache*  RHI::ShaderManager = nullptr;

	inline bool openVulkanValiadation()
	{
		return cVarVulkanOpenValidation.get() != 0;
	}

	static PFN_vkSetDebugUtilsObjectNameEXT GVkSetDebugUtilsObjectName = nullptr;
	static PFN_vkCmdBeginDebugUtilsLabelEXT GVkCmdBeginDebugUtilsLabel = nullptr;
	static PFN_vkCmdEndDebugUtilsLabelEXT   GVkCmdEndDebugUtilsLabel = nullptr;

	void extDebugUtilsGetProcAddresses(VkDevice device)
	{
		GVkSetDebugUtilsObjectName = (PFN_vkSetDebugUtilsObjectNameEXT)vkGetDeviceProcAddr(device, "vkSetDebugUtilsObjectNameEXT");
		GVkCmdBeginDebugUtilsLabel = (PFN_vkCmdBeginDebugUtilsLabelEXT)vkGetDeviceProcAddr(device, "vkCmdBeginDebugUtilsLabelEXT");
		GVkCmdEndDebugUtilsLabel = (PFN_vkCmdEndDebugUtilsLabelEXT)vkGetDeviceProcAddr(device, "vkCmdEndDebugUtilsLabelEXT");
	}

	PFN_vkCmdPushDescriptorSetKHR RHI::PushDescriptorSetKHR = nullptr;
	PFN_vkCmdPushDescriptorSetWithTemplateKHR RHI::PushDescriptorSetWithTemplateKHR = nullptr;

	PFN_vkCreateAccelerationStructureKHR RHI::CreateAccelerationStructure = nullptr;
	PFN_vkDestroyAccelerationStructureKHR RHI::DestroyAccelerationStructure = nullptr;
	PFN_vkCmdBuildAccelerationStructuresKHR RHI::CmdBuildAccelerationStructures = nullptr;
	PFN_vkGetAccelerationStructureDeviceAddressKHR RHI::GetAccelerationStructureDeviceAddress = nullptr;
	PFN_vkGetAccelerationStructureBuildSizesKHR RHI::GetAccelerationStructureBuildSizes = nullptr;

	// Functions for regular HDR ex: HDR10
	RHI::DisplayMode RHI::eDisplayMode = RHI::DisplayMode::DISPLAYMODE_SDR;
	bool RHI::bSupportHDR = false;
	bool RHI::bSupportHDR10_2084 = false;
	bool RHI::bSupportHDR10_SCRGB = false;
	PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR RHI::GetPhysicalDeviceSurfaceCapabilities2KHR = nullptr;
	PFN_vkGetPhysicalDeviceSurfaceFormats2KHR      RHI::GetPhysicalDeviceSurfaceFormats2KHR = nullptr;
	PFN_vkSetHdrMetadataEXT                        RHI::SetHdrMetadataEXT = nullptr;

	
	void GetSurfaceFormats(uint32_t* pFormatCount, std::vector<VkSurfaceFormat2KHR>* surfFormats)
	{
		static VkPhysicalDeviceSurfaceInfo2KHR GPhysicalDeviceSurfaceInfo2KHR;

		GPhysicalDeviceSurfaceInfo2KHR.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SURFACE_INFO_2_KHR;
		GPhysicalDeviceSurfaceInfo2KHR.pNext = nullptr;
		GPhysicalDeviceSurfaceInfo2KHR.surface = RHI::get()->getSurface();

		RHICheck(RHI::GetPhysicalDeviceSurfaceFormats2KHR(RHI::GPU, &GPhysicalDeviceSurfaceInfo2KHR, pFormatCount, NULL));

		uint32_t formatCount = *pFormatCount;
		surfFormats->resize(formatCount);
		for (UINT i = 0; i < formatCount; ++i)
		{
			(*surfFormats)[i].sType = VK_STRUCTURE_TYPE_SURFACE_FORMAT_2_KHR;
		}

		RHICheck(RHI::GetPhysicalDeviceSurfaceFormats2KHR(RHI::GPU, &GPhysicalDeviceSurfaceInfo2KHR, &formatCount, (*surfFormats).data()));
	}

	void additionalGetProcAddresses(VkDevice device)
	{
		RHI::PushDescriptorSetKHR = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(device, "vkCmdPushDescriptorSetKHR");

		RHI::PushDescriptorSetWithTemplateKHR = (PFN_vkCmdPushDescriptorSetWithTemplateKHR)vkGetDeviceProcAddr(device, "vkCmdPushDescriptorSetWithTemplateKHR");

		RHI::CreateAccelerationStructure = (PFN_vkCreateAccelerationStructureKHR)vkGetDeviceProcAddr(device, "vkCreateAccelerationStructureKHR");
		RHI::DestroyAccelerationStructure = (PFN_vkDestroyAccelerationStructureKHR)vkGetDeviceProcAddr(device, "vkDestroyAccelerationStructureKHR");
		RHI::CmdBuildAccelerationStructures = (PFN_vkCmdBuildAccelerationStructuresKHR)vkGetDeviceProcAddr(device, "vkCmdBuildAccelerationStructuresKHR");
		RHI::GetAccelerationStructureDeviceAddress = (PFN_vkGetAccelerationStructureDeviceAddressKHR)vkGetDeviceProcAddr(device, "vkGetAccelerationStructureDeviceAddressKHR");
		RHI::GetAccelerationStructureBuildSizes = (PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetDeviceProcAddr(device, "vkGetAccelerationStructureBuildSizesKHR");
	}

	void hdrInit()
	{
		if (RHI::bSupportHDR)
		{
			RHI::GetPhysicalDeviceSurfaceCapabilities2KHR = (PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR)vkGetInstanceProcAddr(RHI::get()->getInstance(), "vkGetPhysicalDeviceSurfaceCapabilities2KHR");

			RHI::GetPhysicalDeviceSurfaceFormats2KHR = (PFN_vkGetPhysicalDeviceSurfaceFormats2KHR)vkGetInstanceProcAddr(RHI::get()->getInstance(), "vkGetPhysicalDeviceSurfaceFormats2KHR");

			RHI::SetHdrMetadataEXT = (PFN_vkSetHdrMetadataEXT)vkGetDeviceProcAddr(RHI::Device, "vkSetHdrMetadataEXT");
		}
		else
		{
			RHI::bSupportHDR10_2084 = false;
			RHI::bSupportHDR10_SCRGB = false;
		}

		uint32_t formatCount;
		std::vector<VkSurfaceFormat2KHR> surfFormats;
		GetSurfaceFormats(&formatCount, &surfFormats);
		for (uint32_t i = 0; i < formatCount; i++)
		{
			if ((surfFormats[i].surfaceFormat.format == VK_FORMAT_A2R10G10B10_UNORM_PACK32 && surfFormats[i].surfaceFormat.colorSpace == VK_COLOR_SPACE_HDR10_ST2084_EXT)
			||  (surfFormats[i].surfaceFormat.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32 && surfFormats[i].surfaceFormat.colorSpace == VK_COLOR_SPACE_HDR10_ST2084_EXT))
			{
				// If surface formats have HDR10 format even before fullscreen surface is attached, it can only mean windows hdr toggle is on
				RHI::bSupportHDR10_2084 = true; // enable window hdr toggle.
				// RHI::eDisplayMode = RHI::DisplayMode::DISPLAYMODE_HDR10_2084;
				break;
			}
		}

		for (uint32_t i = 0; i < formatCount; i++)
		{
			if (surfFormats[i].surfaceFormat.format == VK_FORMAT_R16G16B16A16_SFLOAT && surfFormats[i].surfaceFormat.colorSpace == VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT)
			{
				RHI::bSupportHDR10_SCRGB = true;
			}
		}
	}

	VkHdrMetadataEXT RHI::HdrMetadataEXT = { .sType = VK_STRUCTURE_TYPE_HDR_METADATA_EXT, .pNext = nullptr };
	bool HdrSetDisplayMode(RHI::DisplayMode displayMode, VkSwapchainKHR swapChain)
	{
		// Don't set hdr display mode, use monitor default setting, user change tonemapper parameter to get better config.
		return false;

		using namespace RHI;
		if (!RHI::bSupportHDR)
			return false;

		switch (displayMode)
		{
		case DISPLAYMODE_SDR:
			// rec 709 primaries
			HdrMetadataEXT.displayPrimaryRed.x = 0.64f;
			HdrMetadataEXT.displayPrimaryRed.y = 0.33f;
			HdrMetadataEXT.displayPrimaryGreen.x = 0.30f;
			HdrMetadataEXT.displayPrimaryGreen.y = 0.60f;
			HdrMetadataEXT.displayPrimaryBlue.x = 0.15f;
			HdrMetadataEXT.displayPrimaryBlue.y = 0.06f;
			HdrMetadataEXT.whitePoint.x = 0.3127f;
			HdrMetadataEXT.whitePoint.y = 0.3290f;
			HdrMetadataEXT.minLuminance = 0.0f;
			HdrMetadataEXT.maxLuminance = 300.0f;
			break;
		case DISPLAYMODE_HDR10_2084:
			// rec 2020 primaries
			HdrMetadataEXT.displayPrimaryRed.x = 0.708f;
			HdrMetadataEXT.displayPrimaryRed.y = 0.292f;
			HdrMetadataEXT.displayPrimaryGreen.x = 0.170f;
			HdrMetadataEXT.displayPrimaryGreen.y = 0.797f;
			HdrMetadataEXT.displayPrimaryBlue.x = 0.131f;
			HdrMetadataEXT.displayPrimaryBlue.y = 0.046f;
			HdrMetadataEXT.whitePoint.x = 0.3127f;
			HdrMetadataEXT.whitePoint.y = 0.3290f;
			HdrMetadataEXT.minLuminance = 0.0f;
			HdrMetadataEXT.maxLuminance = 1000.0f; // This will cause tonemapping to happen on display end as long as it's greater than display's actual queried max luminance. The look will change and it will be display dependent!
			HdrMetadataEXT.maxContentLightLevel = 1000.0f;
			HdrMetadataEXT.maxFrameAverageLightLevel = 400.0f; // max and average content light level data will be used to do tonemapping on display
			break;
		case DISPLAYMODE_HDR10_SCRGB:
			// rec 709 primaries
			HdrMetadataEXT.displayPrimaryRed.x = 0.64f;
			HdrMetadataEXT.displayPrimaryRed.y = 0.33f;
			HdrMetadataEXT.displayPrimaryGreen.x = 0.30f;
			HdrMetadataEXT.displayPrimaryGreen.y = 0.60f;
			HdrMetadataEXT.displayPrimaryBlue.x = 0.15f;
			HdrMetadataEXT.displayPrimaryBlue.y = 0.06f;
			HdrMetadataEXT.whitePoint.x = 0.3127f;
			HdrMetadataEXT.whitePoint.y = 0.3290f;
			HdrMetadataEXT.minLuminance = 0.0f;
			HdrMetadataEXT.maxLuminance = 1000.0f; // This will cause tonemapping to happen on display end as long as it's greater than display's actual queried max luminance. The look will change and it will be display dependent!
			HdrMetadataEXT.maxContentLightLevel = 1000.0f;
			HdrMetadataEXT.maxFrameAverageLightLevel = 400.0f; // max and average content light level data will be used to do tonemapping on display
			break;
		}

		SetHdrMetadataEXT(RHI::Device, 1, &swapChain, &HdrMetadataEXT);

		return true;
	}

	constexpr uint32_t cMaxDebugPrintObjectCount = 3;
	VKAPI_ATTR VkBool32 VKAPI_CALL debugUtilsMessengerCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, 
		VkDebugUtilsMessageTypeFlagsEXT messageType,
		const VkDebugUtilsMessengerCallbackDataEXT* callbackData, 
		void* userData)
	{
		bool bWarn = messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;
		bool bError = messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;

		if (bWarn)
		{
			LOG_RHI_WARN(callbackData->pMessage);
		}
		else if (bError)
		{
			LOG_RHI_ERROR(callbackData->pMessage);
		}
		return VK_FALSE;
	}

	static VKAPI_ATTR VkBool32 VKAPI_CALL debugReportCallback(
		VkDebugReportFlagsEXT flags,
		VkDebugReportObjectTypeEXT type,
		uint64_t object,
		size_t location,
		int32_t messageCode,
		const char* layerPrefix,
		const char* message,
		void* userData)
	{
		if (flags & VK_DEBUG_REPORT_ERROR_BIT_EXT)
		{
			LOG_RHI_ERROR("{}: {}",layerPrefix,message);
		}
		else if (flags & VK_DEBUG_REPORT_WARNING_BIT_EXT)
		{
			LOG_RHI_WARN("{}: {}",layerPrefix, message);
		}
		else if (flags & VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT)
		{
			LOG_RHI_INFO("{}: {}", layerPrefix, message);
		}
		else
		{
			LOG_RHI_TRACE("{}: {}", layerPrefix, message);
		}
		return VK_FALSE;
	}

	VkResult createDebugUtilsMessengerEXT(
		VkInstance instance,
		const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
		const VkAllocationCallbacks* pAllocator,
		VkDebugUtilsMessengerEXT* pDebugMessenger)
	{
		const char* funcName = "vkCreateDebugUtilsMessengerEXT";
		auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, funcName);
		if (func != nullptr)
		{
			return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
		}
		else
		{
			LOG_RHI_ERROR("No vulkan function: {} find, maybe exist some driver problem on the machine.", funcName);
			return VK_ERROR_EXTENSION_NOT_PRESENT;
		}
	}

	void destroyDebugUtilsMessengerEXT(
		VkInstance instance,
		VkDebugUtilsMessengerEXT debugMessenger,
		const VkAllocationCallbacks* pAllocator)
	{
		const char* funcName = "vkDestroyDebugUtilsMessengerEXT";
		auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, funcName);
		if (func != nullptr)
		{
			func(instance, debugMessenger, pAllocator);
		}
		else
		{
			LOG_RHI_ERROR("No vulkan function: {} find, maybe exist some driver problem on the machine.", funcName);
		}
	}

	VkResult createDebugReportCallbackEXT(
		VkInstance instance,
		const VkDebugReportCallbackCreateInfoEXT* pCreateInfo,
		const VkAllocationCallbacks* pAllocator,
		VkDebugReportCallbackEXT* pDebugReporter)
	{
		const char* funcName = "vkCreateDebugReportCallbackEXT";
		auto func = (PFN_vkCreateDebugReportCallbackEXT) vkGetInstanceProcAddr(instance, funcName);
		if (func != nullptr)
		{
			return func(instance, pCreateInfo, pAllocator, pDebugReporter);
		}
		else
		{
			LOG_RHI_ERROR("No vulkan function: {} find, maybe exist some driver problem on the machine.", funcName);
			return VK_ERROR_EXTENSION_NOT_PRESENT;
		}
	}

	void destroyDebugReportCallbackEXT(
		VkInstance instance,
		VkDebugReportCallbackEXT DebugReporter,
		const VkAllocationCallbacks* pAllocator)
	{
		const char* funcName = "vkDestroyDebugReportCallbackEXT";
		auto func = (PFN_vkDestroyDebugReportCallbackEXT) vkGetInstanceProcAddr(instance, funcName);
		if (func != nullptr)
		{
			func(instance, DebugReporter, pAllocator);
		}
		else
		{
			LOG_RHI_ERROR("No vulkan function: {} find, maybe exist some driver problem on the machine.", funcName);
		}
	}

	bool requestLayersAvailable(const std::vector<const char*>& required, const std::vector<VkLayerProperties>& available)
	{
		for (auto layer : required)
		{
			bool found = false;
			for (auto& available_layer : available)
			{
				if (strcmp(available_layer.layerName, layer) == 0)
				{
					found = true;
					break;
				}
			}

			if (!found)
			{
				LOG_RHI_ERROR("No validate layer find: {}.", layer);
				return false;
			}
		}
		return true;
	}

	std::vector<const char*> getOptimalValidationLayers(const std::vector<VkLayerProperties>& supportedInstanceLayers)
	{
		// KHRONOS recommend validation layer select priorities.
		std::vector<std::vector<const char*>> validationLayerPriorityList =
		{
			{"VK_LAYER_KHRONOS_validation"},
			{"VK_LAYER_LUNARG_standard_validation"},
			{
				"VK_LAYER_GOOGLE_threading",
				"VK_LAYER_LUNARG_parameter_validation",
				"VK_LAYER_LUNARG_object_tracker",
				"VK_LAYER_LUNARG_core_validation",
				"VK_LAYER_GOOGLE_unique_objects",
			},
			{"VK_LAYER_LUNARG_core_validation"}
		};

		for (auto& validationLayers : validationLayerPriorityList)
		{
			if (requestLayersAvailable(validationLayers, supportedInstanceLayers))
			{
				return validationLayers;
			}

			LOG_RHI_WARN("Can't open validate lyaer! vulkan will run without debug layer.");
		}
		return {};
	}

	bool checkDeviceExtensionSupport(const std::vector<const char*>& requestDeviceExtens, VkPhysicalDevice GPU)
	{
		uint32_t extensionCount;
		vkEnumerateDeviceExtensionProperties(GPU, nullptr, &extensionCount, nullptr);
		std::vector<VkExtensionProperties> availableExtensions(extensionCount);
		vkEnumerateDeviceExtensionProperties(GPU, nullptr, &extensionCount, availableExtensions.data());

		std::set<std::string> requiredExtensions(requestDeviceExtens.begin(), requestDeviceExtens.end());
		for (const auto& extension : availableExtensions)
		{
			requiredExtensions.erase(extension.extensionName);
		}
		return requiredExtensions.empty();
	}

	void VulkanContext::pickupSuitableGpu(const std::vector<const char*>& requestExtens)
	{
		uint32_t physicalDeviceCount;
		RHICheck(vkEnumeratePhysicalDevices(m_instance, &physicalDeviceCount, nullptr));
		if (physicalDeviceCount < 1)
		{
			LOG_RHI_FATAL("No gpu support vulkan on your computer.");
		}
		std::vector<VkPhysicalDevice> physicalDevices;
		physicalDevices.resize(physicalDeviceCount);
		RHICheck(vkEnumeratePhysicalDevices(m_instance, &physicalDeviceCount, physicalDevices.data()));

		ASSERT(!physicalDevices.empty(), "No gpu on your computer.");

		for (auto& GPU : physicalDevices)
		{
			VkPhysicalDeviceProperties deviceProperties;
			vkGetPhysicalDeviceProperties(GPU, &deviceProperties);
			if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
			{
				m_physicalDevice = GPU;
				if (isPhysicalDeviceSuitable(requestExtens))
				{
					LOG_RHI_INFO("Using discrete gpu: {0}.", toString(deviceProperties.deviceName));
					m_physicalDeviceProperties = deviceProperties;
					return;
				}
			}
		}

		LOG_RHI_WARN("No discrete gpu found, using default gpu.");

		for (auto& GPU : physicalDevices)
		{
			m_physicalDevice = GPU;
			VkPhysicalDeviceProperties deviceProperties;
			vkGetPhysicalDeviceProperties(GPU, &deviceProperties);
			if (isPhysicalDeviceSuitable(requestExtens))
			{
				LOG_RHI_INFO("Choose default gpu: {0}.", toString(deviceProperties.deviceName));
				m_physicalDeviceProperties = deviceProperties;
				return;
			}
		}

		LOG_RHI_FATAL("No suitable gpu can use.");
	}

	bool VulkanContext::isPhysicalDeviceSuitable(const std::vector<const char*>& requestExtens)
	{
		bool bAllQueueExist = false;
		{
			uint32_t queueFamilyCount = 0;
			vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &queueFamilyCount, nullptr);
			std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
			vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &queueFamilyCount, queueFamilies.data());

			bool bSupportGraphics = false;
			bool bSupportCompute = false;
			bool bSupportCopy = false;
			for (const auto& queueFamily : queueFamilies)
			{
				if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
				{
					bSupportGraphics = true;
				}
				if (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT)
				{
					bSupportCompute = true;
				}
				if (queueFamily.queueFlags & VK_QUEUE_TRANSFER_BIT)
				{
					bSupportCopy = true;
				}
			}
			bAllQueueExist = bSupportGraphics && bSupportCompute && bSupportCopy;
		}

		bool extensionsSupported = checkDeviceExtensionSupport(requestExtens, m_physicalDevice);

		bool swapChainAdequate = false;
		if (extensionsSupported)
		{
			auto swapChainSupport = querySwapchainSupportDetail();
			swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
		}

		return bAllQueueExist && extensionsSupported && swapChainAdequate;
	}

	SwapchainSupportDetails VulkanContext::querySwapchainSupportDetail()
	{
		SwapchainSupportDetails details;
		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_physicalDevice, m_surface, &details.capabilities);

		uint32_t formatCount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(m_physicalDevice, m_surface, &formatCount, nullptr);
		if (formatCount != 0)
		{
			details.formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(m_physicalDevice, m_surface, &formatCount, details.formats.data());
		}

		uint32_t presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(m_physicalDevice, m_surface, &presentModeCount, nullptr);
		if (presentModeCount != 0)
		{
			details.presentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(m_physicalDevice, m_surface, &presentModeCount, details.presentModes.data());
		}

		return details;
	}

	void VulkanContext::createLogicDevice(VkPhysicalDeviceFeatures features, void* nextChain, const std::vector<const char*>& requestExtens)
	{
		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &queueFamilyCount, nullptr);
		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &queueFamilyCount, queueFamilies.data());

		uint32_t graphicsQueueCounts = 0;
		uint32_t computeQueueCounts = 0;
		uint32_t copyQueueCounts = 0;

		uint32_t queueIndex = 0;

		bool bGraphicsQueueSet = false;
		bool bCopyQueueSet = false;
		bool bComputeQueueSet = false;

		// NOTE: queueFamilies sort by VkQueueFlagBits in my graphics card, no ensure the order is same with AMD graphics card.
		for (const auto& queueFamily : queueFamilies)
		{
			const bool bSupportSparseBinding = queueFamily.queueFlags & VK_QUEUE_SPARSE_BINDING_BIT;
			const bool bSupportGraphics = queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT;
			const bool bSupportCompute = queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT;
			const bool bSupportCopy = queueFamily.queueFlags & VK_QUEUE_TRANSFER_BIT;

			if (bSupportGraphics && (!bGraphicsQueueSet))
			{
				m_queues.graphicsFamily = queueIndex;
				graphicsQueueCounts = queueFamily.queueCount;

				bGraphicsQueueSet = true;
			}
			else if (bSupportCompute && (!bComputeQueueSet))
			{
				m_queues.computeFamily = queueIndex;
				computeQueueCounts = queueFamily.queueCount;

				bComputeQueueSet = true;
			}
			else if (bSupportCopy && (!bCopyQueueSet))
			{
				m_queues.copyFamily = queueIndex;
				copyQueueCounts = queueFamily.queueCount;

				bCopyQueueSet = true;
			}

			queueIndex++;
		}

		CHECK(graphicsQueueCounts > 2 && "We need more than one queue to do some async dispatch.");
		CHECK(copyQueueCounts > 1 && "We need more than one queue to do some async dispatch.");

		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;

		// Prepare queue priority. all 0.5f.
		std::vector<float> graphicsQueuePriority(graphicsQueueCounts, 0.5f); 
		std::vector<float>   computeQueuePriority(computeQueueCounts, 0.5f); 
		std::vector<float> copyQueuePriority(copyQueueCounts, 0.5f);

		// Major queue use for present and render UI.
		graphicsQueuePriority[0] = 1.0f;

		// Major compute queue. 
		computeQueuePriority[0] = 0.8f;
		graphicsQueuePriority[1] = 0.8f;

		VkDeviceQueueCreateInfo queueCreateInfo{};
		queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queueCreateInfo.queueFamilyIndex = m_queues.graphicsFamily;
		queueCreateInfo.queueCount = graphicsQueueCounts;
		queueCreateInfo.pQueuePriorities = graphicsQueuePriority.data();
		queueCreateInfos.push_back(queueCreateInfo);

		if (computeQueueCounts > 0)
		{
			queueCreateInfo.queueFamilyIndex = m_queues.computeFamily;
			queueCreateInfo.queueCount = computeQueueCounts;
			queueCreateInfo.pQueuePriorities = computeQueuePriority.data();
			queueCreateInfos.push_back(queueCreateInfo);
		}

		if (copyQueueCounts > 0)
		{
			queueCreateInfo.queueFamilyIndex = m_queues.copyFamily;
			queueCreateInfo.queueCount = copyQueueCounts;
			queueCreateInfo.pQueuePriorities = copyQueuePriority.data();
			queueCreateInfos.push_back(queueCreateInfo);
		}

		VkDeviceCreateInfo createInfo{};
		VkPhysicalDeviceFeatures2 physicalDeviceFeatures2{};

		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		createInfo.pQueueCreateInfos = queueCreateInfos.data();
		createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());

		// we use physical device features2.
		createInfo.pEnabledFeatures = nullptr;
		createInfo.pNext = &physicalDeviceFeatures2;
		{
			physicalDeviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
			physicalDeviceFeatures2.features = features;
			physicalDeviceFeatures2.pNext = nextChain;
		}

		// device extension.
		createInfo.enabledExtensionCount = static_cast<uint32_t>(requestExtens.size());
		createInfo.ppEnabledExtensionNames = requestExtens.data();

		// no special device layer, all control by instance layer.
		createInfo.ppEnabledLayerNames = NULL;
		createInfo.enabledLayerCount = 0;

		RHICheck(vkCreateDevice(m_physicalDevice, &createInfo, nullptr, &m_device));

		// get major queues.
		m_queues.graphcisQueues.resize(graphicsQueueCounts);
		for (uint32_t id = 0; id < graphicsQueueCounts; id++)
		{
			vkGetDeviceQueue(m_device, m_queues.graphicsFamily, id, &m_queues.graphcisQueues[id]);
		}

		m_queues.computeQueues.resize(computeQueueCounts);
		for (uint32_t id = 0; id < computeQueueCounts; id++)
		{
			vkGetDeviceQueue(m_device, m_queues.computeFamily, id, &m_queues.computeQueues[id]);
		}

		m_queues.copyQueues.resize(copyQueueCounts);
		for (uint32_t id = 0; id < copyQueueCounts; id++)
		{
			vkGetDeviceQueue(m_device, m_queues.copyFamily, id, &m_queues.copyQueues[id]);
		}
	}

	VkFormat VulkanContext::findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features)
	{
		for (VkFormat format : candidates)
		{
			VkFormatProperties props;
			vkGetPhysicalDeviceFormatProperties(m_physicalDevice, format, &props);

			if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features)
			{
				return format;
			}
			else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features)
			{
				return format;
			}
		}
		LOG_RHI_FATAL("Can't find supported format.");
	}

	int32_t VulkanContext::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
	{
		for (uint32_t i = 0; i < m_physicalDeviceMemoryProperties.memoryTypeCount; i++)
		{
			if ((typeFilter & (1 << i)) && (m_physicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
			{
				return i;
			}
		}

		LOG_RHI_FATAL("No suitable memory type can find.");
		return ~0;
	}

	void VulkanContext::initInstance(const std::vector<const char*>& requiredExtensions, const std::vector<const char*>& requiredLayers)
	{
		bool bUtilsDebug = false;
		std::vector<const char*> enableExtensions{ };

		// Query all useful instance extensions.
		uint32_t instanceExtensionCount;
		RHICheck(vkEnumerateInstanceExtensionProperties(nullptr, &instanceExtensionCount, nullptr));
		std::vector<VkExtensionProperties> availableInstanceExtensions(instanceExtensionCount);
		RHICheck(vkEnumerateInstanceExtensionProperties(nullptr, &instanceExtensionCount, availableInstanceExtensions.data()));
		
		// Prepare debug extensions.
		if (openVulkanValiadation())
		{
			for (auto& availableExtension : availableInstanceExtensions)
			{
				if (strcmp(availableExtension.extensionName, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0)
				{
					bUtilsDebug = true;
					LOG_RHI_INFO("Extension {} is useful. opening...", VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
					enableExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
				}
			}
			if (!bUtilsDebug)
			{
				// When debug util unused, we open debug report extension.
				enableExtensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
			}
		}

		// Surface extensions.
		enableExtensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);

		// GLFW extensions.
		{
			uint32_t glfwExtensionCount = 0;
			const char** glfwExtensions;
			glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
			std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
			for (uint32_t i = 0; i < glfwExtensionCount; i++)
			{
				enableExtensions.push_back(extensions[i]);
			}
		}
		
		// Other extension need open.
		for (auto& availableExtension : availableInstanceExtensions)
		{
			if (strcmp(availableExtension.extensionName, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME) == 0)
			{
				LOG_RHI_INFO("Extension {} is useful. opening...", VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
				enableExtensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
			}
		}

		// Add input request instance extension.
		for (auto extension : requiredExtensions)
		{
			auto extensionName = extension;
			if (std::find_if(availableInstanceExtensions.begin(), availableInstanceExtensions.end(),
				[&extensionName](VkExtensionProperties availableExtension)
				{
					return strcmp(availableExtension.extensionName, extensionName) == 0;
				}) == availableInstanceExtensions.end())
			{
				LOG_RHI_FATAL("Extension {} is no useful. closing...", extensionName);
			}
			else
			{
				enableExtensions.push_back(extensionName);
			}
		}


		// Query all support instance layer.
		uint32_t instanceLayerCount;
		RHICheck(vkEnumerateInstanceLayerProperties(&instanceLayerCount, nullptr));
		std::vector<VkLayerProperties> supportedInstanceLayers(instanceLayerCount);
		RHICheck(vkEnumerateInstanceLayerProperties(&instanceLayerCount, supportedInstanceLayers.data()));

		// Instance layer.
		std::vector<const char*> requestedInstanceLayers(requiredLayers);

		// Validation layer.
		if (openVulkanValiadation())
		{
			auto validationLayers = getOptimalValidationLayers(supportedInstanceLayers);
			requestedInstanceLayers.insert(requestedInstanceLayers.end(), validationLayers.begin(), validationLayers.end());

			if (requestLayersAvailable(requestedInstanceLayers, supportedInstanceLayers))
			{
				LOG_RHI_INFO("Request layers opened:");
				for (const auto& layer : requestedInstanceLayers)
				{
					LOG_RHI_INFO("	\t{}", layer);
				}
			}
			else
			{
				LOG_RHI_FATAL("No instance layer found.");
			}
		}

		// Vulkan info.
		VkApplicationInfo appInfo{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
		appInfo.pApplicationName = "FlowerRHI";
		appInfo.applicationVersion = 0;
		appInfo.pEngineName = "FlowerEngine";
		appInfo.engineVersion = 0;
		appInfo.apiVersion = VK_MAKE_VERSION(1, 3, 0);

		// Instance info.
		VkInstanceCreateInfo instanceInfo = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
		instanceInfo.pApplicationInfo = &appInfo;
		instanceInfo.enabledExtensionCount = static_cast<uint32_t>(enableExtensions.size());
		instanceInfo.ppEnabledExtensionNames = enableExtensions.data();
		instanceInfo.enabledLayerCount = static_cast<uint32_t>(requestedInstanceLayers.size());
		instanceInfo.ppEnabledLayerNames = requestedInstanceLayers.data();

		VkDebugUtilsMessengerCreateInfoEXT debugUtilsCreateInfo = { VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
		VkDebugReportCallbackCreateInfoEXT debugReportCreateInfo = { VK_STRUCTURE_TYPE_DEBUG_REPORT_CREATE_INFO_EXT };
		if (openVulkanValiadation())
		{
			if (bUtilsDebug)
			{
				debugUtilsCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;
				debugUtilsCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
				debugUtilsCreateInfo.pfnUserCallback = debugUtilsMessengerCallback;
				instanceInfo.pNext = &debugUtilsCreateInfo;
			}
			else
			{
				debugReportCreateInfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT;
				debugReportCreateInfo.pfnCallback = debugReportCallback;
				instanceInfo.pNext = &debugReportCreateInfo;
			}
		}

		// create vulkan instance.
		RHICheck(vkCreateInstance(&instanceInfo, nullptr, &m_instance));

		if (openVulkanValiadation())
		{
			if (bUtilsDebug)
			{
				RHICheck(createDebugUtilsMessengerEXT(m_instance, &debugUtilsCreateInfo, nullptr, &m_debugUtilsHandle));
			}
			else
			{
				RHICheck(createDebugReportCallbackEXT(m_instance, &debugReportCreateInfo, nullptr, &m_debugReportHandle));
			}
		}
	}

	void VulkanContext::releaseInstance()
	{
		if (m_debugUtilsHandle != VK_NULL_HANDLE)
		{
			destroyDebugUtilsMessengerEXT(m_instance, m_debugUtilsHandle, nullptr);
			m_debugUtilsHandle = VK_NULL_HANDLE;
		}

		if (m_debugReportHandle != VK_NULL_HANDLE)
		{
			destroyDebugReportCallbackEXT(m_instance, m_debugReportHandle, nullptr);
			m_debugReportHandle = VK_NULL_HANDLE;
		}

		if (m_instance != VK_NULL_HANDLE)
		{
			vkDestroyInstance(m_instance, nullptr);
			m_instance = VK_NULL_HANDLE;
		}
	}


	void VulkanContext::initDevice(VkPhysicalDeviceFeatures features, const std::vector<const char*>& requestExtens, void* nextChain)
	{
		// Pick best GPU on machine.
		pickupSuitableGpu(requestExtens);

		// Store GPU memory properties.
		vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &m_physicalDeviceMemoryProperties);

		// Store GPU properties.
		vkGetPhysicalDeviceProperties(m_physicalDevice, &m_physicalDeviceProperties);
		LOG_RHI_INFO("GPU min align memory size: {0}.", m_physicalDeviceProperties.limits.minUniformBufferOffsetAlignment);

		// Cache useful GPU properties2.
		auto vkGetPhysicalDeviceProperties2KHR = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2KHR>(vkGetInstanceProcAddr(m_instance, "vkGetPhysicalDeviceProperties2KHR"));
		CHECK(vkGetPhysicalDeviceProperties2KHR);
		{
			VkPhysicalDeviceProperties2KHR deviceProperties{};

			m_descriptorIndexingProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES_EXT;
			deviceProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2_KHR;
			deviceProperties.pNext = &m_descriptorIndexingProperties;

			vkGetPhysicalDeviceProperties2KHR(m_physicalDevice, &deviceProperties);
		}

		// Build logic device.
		createLogicDevice(features, nextChain, requestExtens);

		// Cache some support format.
		m_cacheSupportDepthOnlyFormat = findSupportedFormat(
			{ VK_FORMAT_D32_SFLOAT }, 
			VK_IMAGE_TILING_OPTIMAL, 
			VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
		);
		m_cacheSupportDepthStencilFormat = findSupportedFormat(
			{ VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
			VK_IMAGE_TILING_OPTIMAL, 
			VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
		);
	}

	void VulkanContext::releaseDevice()
	{
		vkDestroyDevice(m_device, nullptr);
	}

	void VulkanContext::PresentContext::init()
	{
		CHECK(RHI::GMaxSwapchainCount != ~0);

		semaphoresImageAvailable.resize(RHI::GMaxSwapchainCount);
		semaphoresRenderFinished.resize(RHI::GMaxSwapchainCount);

		inFlightFences.resize(RHI::GMaxSwapchainCount);
		imagesInFlight.resize(RHI::GMaxSwapchainCount);
		for (auto& fence : imagesInFlight)
		{
			fence = VK_NULL_HANDLE;
		}

		VkSemaphoreCreateInfo semaphoreInfo{};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		for (size_t i = 0; i < RHI::GMaxSwapchainCount; i++)
		{
			RHICheck(vkCreateSemaphore(RHI::Device, &semaphoreInfo, nullptr, &semaphoresImageAvailable[i]));
			RHICheck(vkCreateSemaphore(RHI::Device, &semaphoreInfo, nullptr, &semaphoresRenderFinished[i]));
			RHICheck(vkCreateFence(RHI::Device, &fenceInfo, nullptr, &inFlightFences[i]));
		}
	}

	void VulkanContext::PresentContext::release()
	{
		for (size_t i = 0; i < RHI::GMaxSwapchainCount; i++)
		{
			vkDestroySemaphore(RHI::Device, semaphoresImageAvailable[i], nullptr);
			vkDestroySemaphore(RHI::Device, semaphoresRenderFinished[i], nullptr);
			    vkDestroyFence(RHI::Device,           inFlightFences[i], nullptr);
		}
	}

	void VulkanContext::init(GLFWwindow* window)
	{
		m_window = window;

		// Init instance.
		{
			std::vector<const char*> instanceExtensionNames{ };
			instanceExtensionNames.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
			instanceExtensionNames.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
			instanceExtensionNames.push_back(VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME); // push once here, query later. bad design. :(

			std::vector<const char*> instanceLayerNames{ };

			initInstance(instanceExtensionNames, instanceLayerNames);
		}

		// Init surface.
		if (glfwCreateWindowSurface(m_instance, window, nullptr, &m_surface) != VK_SUCCESS)
		{
			LOG_RHI_FATAL("Window surface create error.");
		}
		
		// Init device.
		{
			std::vector<const char*> deviceExtensionNames{ };
			deviceExtensionNames.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
			deviceExtensionNames.push_back(VK_KHR_MAINTENANCE1_EXTENSION_NAME);
			deviceExtensionNames.push_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
			deviceExtensionNames.push_back(VK_KHR_MAINTENANCE3_EXTENSION_NAME);
			deviceExtensionNames.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
			deviceExtensionNames.push_back(VK_KHR_16BIT_STORAGE_EXTENSION_NAME);
			deviceExtensionNames.push_back(VK_KHR_8BIT_STORAGE_EXTENSION_NAME);
			deviceExtensionNames.push_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
			deviceExtensionNames.push_back(VK_EXT_HDR_METADATA_EXTENSION_NAME); // push once here, query later. bad design. :(
#if 0
			deviceExtensionNames.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME); // RTX.
			deviceExtensionNames.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
			deviceExtensionNames.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
			deviceExtensionNames.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
#endif		


			// Current only nvidia support Meshshader, so we don't use it, we simulate by compute shader.
			// deviceExtensionNames.push_back(VK_NV_MESH_SHADER_EXTENSION_NAME);

			VkPhysicalDeviceFeatures         enable10GpuFeatures = {}; // vulkan 1.0
			VkPhysicalDeviceVulkan11Features enable11GpuFeatures = {}; // vulkan 1.1
			VkPhysicalDeviceVulkan12Features enable12GpuFeatures = {}; // vulkan 1.2
			VkPhysicalDeviceVulkan13Features enable13GpuFeatures = {}; // vulkan 1.3

			// Enable gpu features 1.0 here.
			enable10GpuFeatures.samplerAnisotropy = true;
			enable10GpuFeatures.depthClamp = true;
			enable10GpuFeatures.shaderSampledImageArrayDynamicIndexing = true;
			enable10GpuFeatures.multiDrawIndirect = VK_TRUE;
			enable10GpuFeatures.drawIndirectFirstInstance = VK_TRUE;
			enable10GpuFeatures.independentBlend = VK_TRUE;
			enable10GpuFeatures.multiViewport = VK_TRUE;
			enable10GpuFeatures.fragmentStoresAndAtomics = VK_TRUE;
			enable10GpuFeatures.shaderInt16 = VK_TRUE;

			// Enable gpu features 1.1 here.
			enable11GpuFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
			enable11GpuFeatures.pNext = &enable12GpuFeatures;
			enable11GpuFeatures.shaderDrawParameters = VK_TRUE;

			// Enable gpu features 1.2 here.
			enable12GpuFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
			enable12GpuFeatures.pNext = &enable13GpuFeatures;
			enable12GpuFeatures.drawIndirectCount = VK_TRUE;
			enable12GpuFeatures.drawIndirectCount = VK_TRUE;
			enable12GpuFeatures.imagelessFramebuffer = VK_TRUE;
			enable12GpuFeatures.separateDepthStencilLayouts = VK_TRUE;
			enable12GpuFeatures.descriptorIndexing = VK_TRUE;
			enable12GpuFeatures.runtimeDescriptorArray = VK_TRUE;
			enable12GpuFeatures.descriptorBindingPartiallyBound = VK_TRUE;
			enable12GpuFeatures.descriptorBindingVariableDescriptorCount = VK_TRUE;
			enable12GpuFeatures.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
			enable12GpuFeatures.descriptorBindingUpdateUnusedWhilePending = VK_TRUE;
			enable12GpuFeatures.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
			enable12GpuFeatures.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
			enable12GpuFeatures.shaderStorageBufferArrayNonUniformIndexing = VK_TRUE;
			enable12GpuFeatures.timelineSemaphore = VK_TRUE;
			enable12GpuFeatures.bufferDeviceAddress = VK_TRUE;

			// Enable gpu features 1.3 here.
			enable13GpuFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
			enable13GpuFeatures.pNext = nullptr;
			enable13GpuFeatures.dynamicRendering = VK_TRUE;
			enable13GpuFeatures.synchronization2 = VK_TRUE;
			enable13GpuFeatures.maintenance4 = VK_TRUE;
			initDevice(enable10GpuFeatures, deviceExtensionNames, &enable11GpuFeatures);
		}

		RHI::GPU = m_physicalDevice;
		RHI::Device = m_device;

		{
			// Query all useful instance extensions.
			uint32_t instanceExtensionCount;
			RHICheck(vkEnumerateInstanceExtensionProperties(nullptr, &instanceExtensionCount, nullptr));
			std::vector<VkExtensionProperties> availableInstanceExtensions(instanceExtensionCount);
			RHICheck(vkEnumerateInstanceExtensionProperties(nullptr, &instanceExtensionCount, availableInstanceExtensions.data()));

			// Query all useful device extensions.
			uint32_t deviceExtensionCount;
			vkEnumerateDeviceExtensionProperties(RHI::GPU, nullptr, &deviceExtensionCount, nullptr);
			std::vector<VkExtensionProperties> availableDeviceExtensions(deviceExtensionCount);
			vkEnumerateDeviceExtensionProperties(RHI::GPU, nullptr, &deviceExtensionCount, availableDeviceExtensions.data());

			auto existInstanceExtension = [&](const char* name)
			{
				for (auto& availableExtension : availableInstanceExtensions)
				{
					if (strcmp(availableExtension.extensionName, name) == 0)
					{
						return true;
					}
				}
				return false;
			};

			auto existDeviceExtension = [&](const char* name)
			{
				for (auto& availableExtension : availableDeviceExtensions)
				{
					if (strcmp(availableExtension.extensionName, name) == 0)
					{
						return true;
					}
				}
				return false;
			};

			//
			RHI::bSupportHDR =
				existInstanceExtension(VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME) &&
				existDeviceExtension(VK_EXT_HDR_METADATA_EXTENSION_NAME);

			RHI::bSupportRayTrace =
				existDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME) &&
				existDeviceExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) &&
				existDeviceExtension(VK_KHR_RAY_QUERY_EXTENSION_NAME) &&
				existDeviceExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);

			RHI::bSupportRayTrace &= (cVarVulkanOpenRayTrace.get() != 0);
		}

		if(RHI::bSupportRayTrace)
		{
			// Get the acceleration structure features, which we'll need later on in the sample
			m_accelerationStructure.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;

			VkPhysicalDeviceFeatures2 deviceFeatures{};
			deviceFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
			deviceFeatures.pNext = &m_accelerationStructure;

			vkGetPhysicalDeviceFeatures2(RHI::GPU, &deviceFeatures);
		}

		// Find some proce address.
		extDebugUtilsGetProcAddresses(m_device);
		additionalGetProcAddresses(m_device);

		initVMA();
		RHI::VMA = m_vmaAllocator;

		m_shaderCache.init();
		RHI::ShaderManager = &m_shaderCache;

		// Init sampler cache.
		m_samplerCache.init();
		RHI::SamplerManager = &m_samplerCache;

		// Init descriptor cache.
		m_descriptorAllocator.init();
		m_descriptorLayoutCache.init();

		// Init hdr info before swapchin build.
		hdrInit();

		// Init swapchain.
		m_swapchain.init();
		RHI::GMaxSwapchainCount = m_swapchain.getImageViews().size();

		m_presentContext.init();

		initCommandPool();

		CHECK(m_copyPools.size() > 0 && m_computePools.size() > 0 && m_graphicsPools.size() > 0 &&
			"Your graphics card is too old and even no support async vulkan queue. exiting...");
	
		Bindless::Sampler->init();
		Bindless::Texture->init();


	}

	void VulkanContext::release()
	{
		Bindless::Sampler->release();
		Bindless::Texture->release();
		releaseCommandPool();

		// release present context.
		m_presentContext.release();

		// release swapchain.
		m_swapchain.release();

		// release descriptor cache.
		m_descriptorAllocator.release();
		m_descriptorLayoutCache.release();

		// release sampler cache.
		m_samplerCache.release();

		m_shaderCache.release();

		releaseVMA();

		// release device.
		releaseDevice();

		// release surface.
		vkDestroySurfaceKHR(m_instance, m_surface, nullptr);

		// release instance.
		releaseInstance();
	}

	void VulkanContext::initVMA()
	{
		VmaAllocatorCreateInfo allocatorInfo = {};
		allocatorInfo.physicalDevice = m_physicalDevice;
		allocatorInfo.device = m_device;
		allocatorInfo.instance = m_instance;
		if (RHI::bSupportRayTrace)
		{
			allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
		}
		
		vmaCreateAllocator(&allocatorInfo, &m_vmaAllocator);
	}

	void VulkanContext::releaseVMA()
	{
		vmaDestroyAllocator(m_vmaAllocator); 
	}

	void VulkanContext::initCommandPool()
	{
		VkCommandPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

		// Graphics command pools.
		CHECK(m_queues.graphcisQueues.size() > 2 && "Your device too old and even don't support more than one graphics queue.");
		poolInfo.queueFamilyIndex = m_queues.graphicsFamily;

		m_majorGraphicsPool.queue = m_queues.graphcisQueues[0];
		m_secondMajorGraphicsPool.queue = m_queues.graphcisQueues[1];

		RHICheck(vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_majorGraphicsPool.pool));
		RHICheck(vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_secondMajorGraphicsPool.pool));

		if (m_queues.graphcisQueues.size() > 2)
		{
			m_graphicsPools.resize(m_queues.graphcisQueues.size() - 2);
			uint32_t index = 2;
			for (auto& pool : m_graphicsPools)
			{
				pool.queue = m_queues.graphcisQueues[index];
				RHICheck(vkCreateCommandPool(m_device, &poolInfo, nullptr, &pool.pool));

				index++;
			}
		}

		// Compute command pools.
		CHECK(m_queues.computeQueues.size() > 1 && "Your device too old and even don't support more than one compute queue.");
		poolInfo.queueFamilyIndex = m_queues.computeFamily;
		m_majorComputePool.queue = m_queues.computeQueues[0];
		RHICheck(vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_majorComputePool.pool));
		if (m_queues.computeQueues.size() > 1)
		{
			m_computePools.resize(m_queues.computeQueues.size() - 1);
			uint32_t index = 1;
			for (auto& pool : m_computePools)
			{
				pool.queue = m_queues.computeQueues[index];
				RHICheck(vkCreateCommandPool(m_device, &poolInfo, nullptr, &pool.pool));
				index++;
			}
		}

		// Copy command pools.
		poolInfo.queueFamilyIndex = m_queues.copyFamily;
		if (m_queues.copyQueues.size() > 0)
		{
			m_copyPools.resize(m_queues.copyQueues.size());
			uint32_t index = 0;
			for (auto& pool : m_copyPools)
			{
				pool.queue = m_queues.copyQueues[index];
				RHICheck(vkCreateCommandPool(m_device, &poolInfo, nullptr, &pool.pool));
				index++;
			}
		}
	}

	void VulkanContext::releaseCommandPool()
	{
		// Destroy major command pool.
		vkDestroyCommandPool(m_device, m_majorGraphicsPool.pool, nullptr);
		
		// Async queue.
		vkDestroyCommandPool(m_device, m_majorComputePool.pool, nullptr);
		vkDestroyCommandPool(m_device, m_secondMajorGraphicsPool.pool, nullptr);

		// Destroy other command pool.
		for (auto& pool : m_graphicsPools)
		{
			vkDestroyCommandPool(m_device, pool.pool, nullptr);
		}
		for (auto& pool : m_computePools)
		{
			vkDestroyCommandPool(m_device, pool.pool, nullptr);
		}
		for (auto& pool : m_copyPools)
		{
			vkDestroyCommandPool(m_device, pool.pool, nullptr);
		}
	}

	bool VulkanContext::swapchainRebuild()
	{
		glfwGetWindowSize(m_window, &currentWidth, &currentHeight);

		if (currentWidth != lastWidth || currentHeight != lastHeight)
		{
			lastWidth = currentWidth;
			lastHeight = currentHeight;
			return true;
		}

		return false;
	}

	void VulkanContext::recreateSwapChain()
	{
		vkDeviceWaitIdle(m_device);

		static int width = 0, height = 0;
		glfwGetFramebufferSize(m_window, &width, &height);

		// just return if swapchain width or height is 0.
		if (width == 0 || height == 0)
		{
			m_presentContext.bSwapchainChange = true;
			return;
		}

		// Clean special
		onBeforeSwapchainRecreate.broadcast();

		m_presentContext.release();
		m_swapchain.release();
		m_swapchain.init();
		HdrSetDisplayMode(RHI::eDisplayMode, m_swapchain.get());

		m_presentContext.init();

		m_presentContext.imagesInFlight.resize(m_swapchain.getImageViews().size(), VK_NULL_HANDLE);

		// Recreate special
		onAfterSwapchainRecreate.broadcast();
	}

	uint32_t VulkanContext::acquireNextPresentImage()
	{
		m_presentContext.bSwapchainChange |= swapchainRebuild();

		vkWaitForFences(m_device, 1, &m_presentContext.inFlightFences[m_presentContext.currentFrame], VK_TRUE, UINT64_MAX);

		VkResult result = vkAcquireNextImageKHR(
			m_device, 
			m_swapchain.get(),
			UINT64_MAX, 
			m_presentContext.semaphoresImageAvailable[m_presentContext.currentFrame],
			VK_NULL_HANDLE, 
			&m_presentContext.imageIndex
		);

		if (result == VK_ERROR_OUT_OF_DATE_KHR)
		{
			recreateSwapChain();
		}
		else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
		{
			LOG_RHI_FATAL("Fail to requeset present image.");
		}

		if (m_presentContext.imagesInFlight[m_presentContext.imageIndex] != VK_NULL_HANDLE)
		{
			vkWaitForFences(m_device, 1, &m_presentContext.imagesInFlight[m_presentContext.imageIndex], VK_TRUE, UINT64_MAX);
		}

		m_presentContext.imagesInFlight[m_presentContext.imageIndex] = m_presentContext.inFlightFences[m_presentContext.currentFrame];

		return m_presentContext.imageIndex;
	}

	void VulkanContext::present()
	{
		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

		VkSemaphore signalSemaphores[] = { m_presentContext.semaphoresRenderFinished[m_presentContext.currentFrame] };
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = signalSemaphores;

		VkSwapchainKHR swapchains[] = { m_swapchain.get()};
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapchains;
		presentInfo.pImageIndices = &m_presentContext.imageIndex;

		auto result = vkQueuePresentKHR(m_majorGraphicsPool.queue, &presentInfo);
		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || m_presentContext.bSwapchainChange)
		{
			m_presentContext.bSwapchainChange = false;
			recreateSwapChain();
		}
		else if (result != VK_SUCCESS)
		{
			LOG_RHI_FATAL("Fail to present image.");
		}

		// if swapchain rebuild and on minimized, still add frame.
		m_presentContext.currentFrame = (m_presentContext.currentFrame + 1) % RHI::GMaxSwapchainCount;
	}

	void VulkanContext::submit(uint32_t count, VkSubmitInfo* infos)
	{
		RHICheck(vkQueueSubmit(m_majorGraphicsPool.queue, count, infos, m_presentContext.inFlightFences[m_presentContext.currentFrame]));
	}

	void VulkanContext::submitNoFence(uint32_t count, VkSubmitInfo* infos)
	{
		RHICheck(vkQueueSubmit(m_majorGraphicsPool.queue, count, infos, nullptr));
	}

	void VulkanContext::resetFence()
	{
		RHICheck(vkResetFences(m_device, 1, &m_presentContext.inFlightFences[m_presentContext.currentFrame]));
	}

	VkCommandBuffer VulkanContext::createMajorGraphicsCommandBuffer()
	{
		VkCommandBufferAllocateInfo info{};
		info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		info.commandBufferCount = 1;
		info.commandPool = getMajorGraphicsCommandPool();
		info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

		VkCommandBuffer newBuffer;
		RHICheck(vkAllocateCommandBuffers(RHI::Device, &info, &newBuffer));

		return newBuffer;
	}

	DescriptorFactory VulkanContext::descriptorFactoryBegin()
	{
		return DescriptorFactory::begin(&m_descriptorLayoutCache, &m_descriptorAllocator);
	}

	VkPipelineLayout VulkanContext::createPipelineLayout(const VkPipelineLayoutCreateInfo& info)
	{
		VkPipelineLayout layout;
		RHICheck(vkCreatePipelineLayout(m_device, &info, nullptr, &layout));
		return layout;
	}

	void RHI::setResourceName(VkObjectType objectType, uint64_t handle, const char* name)
	{
		VkDevice device = Device;

		static std::mutex gMutexForSetResource;
		if (GVkSetDebugUtilsObjectName && handle && name)
		{
			std::unique_lock<std::mutex> lock(gMutexForSetResource);

			VkDebugUtilsObjectNameInfoEXT nameInfo = {};
			nameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
			nameInfo.objectType = objectType;
			nameInfo.objectHandle = handle;
			nameInfo.pObjectName = name;

			GVkSetDebugUtilsObjectName(device, &nameInfo);
		}

	}

	void RHI::setPerfMarkerBegin(VkCommandBuffer cmdBuf, const char* name, const glm::vec4& color)
	{
		if (GVkCmdBeginDebugUtilsLabel)
		{
			VkDebugUtilsLabelEXT label = {};
			label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
			label.pLabelName = name;
			label.color[0] = color.r;
			label.color[1] = color.g;
			label.color[2] = color.b;
			label.color[3] = color.a;
			GVkCmdBeginDebugUtilsLabel(cmdBuf, &label);
		}
	}

	void RHI::setPerfMarkerEnd(VkCommandBuffer cmdBuf)
	{
		if (GVkCmdEndDebugUtilsLabel)
		{
			GVkCmdEndDebugUtilsLabel(cmdBuf);
		}
	}

	void RHI::executeImmediately(VkCommandPool commandPool, VkQueue queue, std::function<void(VkCommandBuffer cb)>&& func)
	{
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = commandPool;
		allocInfo.commandBufferCount = 1;

		VkCommandBuffer commandBuffer;
		vkAllocateCommandBuffers(RHI::Device, &allocInfo, &commandBuffer);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(commandBuffer, &beginInfo);
		func(commandBuffer);

		vkEndCommandBuffer(commandBuffer);
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;
		vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(queue);
		vkFreeCommandBuffers(RHI::Device, commandPool, 1, &commandBuffer);
	}

	void RHI::executeImmediatelyMajorGraphics(std::function<void(VkCommandBuffer cb)>&& func)
	{
		executeImmediately(RHI::get()->getMajorGraphicsCommandPool(), RHI::get()->getMajorGraphicsQueue(), std::move(func));
	}

	namespace RHI
	{
		std::atomic<size_t> GpuResourceMemoryUsed = 0;
	}

	void RHI::addGpuResourceMemoryUsed(size_t in)
	{
		GpuResourceMemoryUsed.fetch_add(in);
	}

	void RHI::minusGpuResourceMemoryUsed(size_t in)
	{
		GpuResourceMemoryUsed.fetch_sub(in);
	}

	size_t RHI::getGpuResourceMemoryUsed()
	{
		return GpuResourceMemoryUsed.load();
	}
}