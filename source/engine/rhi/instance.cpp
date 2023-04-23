#include "rhi.h"

namespace engine
{
    static AutoCVarBool cVarRHIOpenValidation(
        "r.RHI.OpenValidation",
        "Enable rhi validation info output or not.",
        "RHI",
        true,
        CVarFlags::ReadOnly
    );

    static AutoCVarBool cVarRHIDebugUtilsEnable(
        "r.RHI.DebugUtilsEnable",
        "Debug utils enable or not.",
        "RHI",
        true,
        CVarFlags::ReadOnly
    );

    static AutoCVarInt32 cVarRHIDebugUtilsLevel(
        "r.RHI.DebugUtilsLevel",
        "Debug utils output info levels, 0 is off, 1 is only error, 2 is warning|error, 3 is info|warning|error, 4 is all message capture from validation.",
        "RHI",
        1,
        CVarFlags::ReadAndWrite
    );

    static inline bool enableDebugUtilsCallback()
    {
        return cVarRHIDebugUtilsEnable.get();
    }

    static inline bool openValidation()
    {
        return cVarRHIOpenValidation.get() != 0;
    }

    static inline bool requestLayersAvailable(const std::vector<const char*>& required, const std::vector<VkLayerProperties>& available)
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
                return false;
            }
        }
        return true;
    }

    std::vector<const char*> getOptimalValidationLayers(const std::vector<VkLayerProperties>& supportedInstanceLayers)
    {
        // Khronos recommend validation layer select priorities.
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

        // Select one validation layer which can use.
        for (auto& validationLayers : validationLayerPriorityList)
        {
            if (requestLayersAvailable(validationLayers, supportedInstanceLayers))
            {
                return validationLayers;
            }
        }

        LOG_RHI_WARN("Can't open suitable validate lyaer! vulkan will run without debug layer.");
        return {};
    }

    static inline VkDebugUtilsMessengerEXT createDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT& ci)
    {
        static PFN_vkCreateDebugUtilsMessengerEXT pfnCreateDebugUtilsMessengerEXT = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
        
        VkDebugUtilsMessengerEXT cb;
        RHICheck(pfnCreateDebugUtilsMessengerEXT(instance, &ci, nullptr, &cb));

        return cb;
    }

    static inline void destroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT cb)
    {
        static PFN_vkDestroyDebugUtilsMessengerEXT pfnDestroyDebugUtilsMessengerEXT = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
        if (cb != VK_NULL_HANDLE)
        {
            pfnDestroyDebugUtilsMessengerEXT(instance, cb, nullptr);
        }
    }

    VKAPI_ATTR VkBool32 VKAPI_CALL debugUtilsMessengerCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
        void* userData)
    {
        static int32_t* varPtr = cVarRHIDebugUtilsLevel.getPtr();
        if(*varPtr == 0)
        {
            return VK_FALSE;
        }

        const bool bVerse = (*varPtr >= 4) && (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT);
        const bool bInfo  = (*varPtr >= 3) && (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT);
        const bool bWarn  = (*varPtr >= 2) && (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT);
        const bool bError = (*varPtr >= 1) && (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT);

        if(bVerse)
        {
            LOG_RHI_TRACE(callbackData->pMessage);
        }
        else if(bInfo)
        {
            LOG_RHI_INFO(callbackData->pMessage);
        }
        else if (bWarn)
        {
            LOG_RHI_WARN(callbackData->pMessage);
        }
        else if (bError)
        {
            LOG_RHI_ERROR(callbackData->pMessage);
            return VK_TRUE;
        }
        return VK_FALSE;
    }

    void VulkanContext::initInstance()
    {
        // Query all useful instance extensions.
        uint32_t instanceExtensionCount;
        RHICheck(vkEnumerateInstanceExtensionProperties(nullptr, &instanceExtensionCount, nullptr));
        std::vector<VkExtensionProperties> availableInstanceExtensions(instanceExtensionCount);
        RHICheck(vkEnumerateInstanceExtensionProperties(nullptr, &instanceExtensionCount, availableInstanceExtensions.data()));

        // Fill instance extensions.
        std::vector<const char*> enableExtensions{ };
        {
            // Debug utils extension enable.
            if(enableDebugUtilsCallback()) enableExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            
            // Windows surface enable and glfw enable.
            if(!m_engine->isConsoleApp())
            {
                // Surface.
                enableExtensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);

                // Glfw require extensions.
                uint32_t glfwExtensionCount = 0;
                const char** glfwExtensions;
                glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
                std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
                for (uint32_t i = 0; i < glfwExtensionCount; i++)
                {
                    enableExtensions.push_back(extensions[i]);
                }

                // Surfaces extension additional.
                enableExtensions.push_back(VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME);
            } 

            // Also need these extension for query some additional info.
            enableExtensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
            
            // Check all extension available.
            for (const auto& extensionName : enableExtensions)
            {
                if (std::find_if(
                    availableInstanceExtensions.begin(), 
                    availableInstanceExtensions.end(),
                    [&extensionName](const VkExtensionProperties& availableExtension)
                    {
                        return strcmp(availableExtension.extensionName, extensionName) == 0;
                    }) == availableInstanceExtensions.end())
                {
                    LOG_RHI_FATAL("Require instance extension {} is no useful, please update your driver or install newest vulkan sdk. closing...", extensionName);
                }
            }
        }

        // Query all support instance layer.
		uint32_t instanceLayerCount;
		RHICheck(vkEnumerateInstanceLayerProperties(&instanceLayerCount, nullptr));
		std::vector<VkLayerProperties> supportedInstanceLayers(instanceLayerCount);
		RHICheck(vkEnumerateInstanceLayerProperties(&instanceLayerCount, supportedInstanceLayers.data()));

        // Instance layer.
		std::vector<const char*> enableInstanceLayers{};
        {
            if(openValidation())
            {
                auto validationLayers = getOptimalValidationLayers(supportedInstanceLayers);
                enableInstanceLayers.insert(enableInstanceLayers.end(), validationLayers.begin(), validationLayers.end());
            }

            if (requestLayersAvailable(enableInstanceLayers, supportedInstanceLayers))
			{
				LOG_RHI_INFO("Request layers opened:");
				for (const auto& layer : enableInstanceLayers)
				{
					LOG_RHI_INFO("	\t{}", layer);
				}
			}
			else
			{
				LOG_RHI_FATAL("No all instance layer found, update your PC driver or install newest Vulkan SDK.");
			}
        }

        // Vulkan info.
		VkApplicationInfo appInfo{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
		appInfo.pApplicationName = "rhi";
		appInfo.applicationVersion = 0;
		appInfo.pEngineName   = "engine";
		appInfo.engineVersion = 0;
		appInfo.apiVersion    = VK_MAKE_VERSION(1, 3, 0);

        // Instance info.
		VkInstanceCreateInfo instanceInfo = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
		instanceInfo.pApplicationInfo = &appInfo;
		instanceInfo.enabledExtensionCount = static_cast<uint32_t>(enableExtensions.size());
		instanceInfo.ppEnabledExtensionNames = enableExtensions.data();
		instanceInfo.enabledLayerCount = static_cast<uint32_t>(enableInstanceLayers.size());
		instanceInfo.ppEnabledLayerNames = enableInstanceLayers.data();

        // Create debug util callback.
        VkDebugUtilsMessengerCreateInfoEXT debugUtilsCreateInfo = { VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
        debugUtilsCreateInfo.pNext = NULL;
        if(enableDebugUtilsCallback())
        {
            debugUtilsCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;
			debugUtilsCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
			debugUtilsCreateInfo.pfnUserCallback = debugUtilsMessengerCallback;
			instanceInfo.pNext = &debugUtilsCreateInfo;
        }
        
        // create vulkan instance.
		RHICheck(vkCreateInstance(&instanceInfo, nullptr, &m_instance));

        if(enableDebugUtilsCallback())
        {
            m_debugUtilsHandle = createDebugUtilsMessengerEXT(m_instance, debugUtilsCreateInfo);
        }
    }

    void VulkanContext::destroyInstance()
    {
        if (m_debugUtilsHandle != VK_NULL_HANDLE)
		{
			destroyDebugUtilsMessengerEXT(m_instance, m_debugUtilsHandle);
			m_debugUtilsHandle = VK_NULL_HANDLE;
		}

		if (m_instance != VK_NULL_HANDLE)
		{
			vkDestroyInstance(m_instance, nullptr);
			m_instance = VK_NULL_HANDLE;
		}
    }

}