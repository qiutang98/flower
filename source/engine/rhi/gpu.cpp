#include "rhi.h"

namespace engine
{
    void VulkanContext::selectGPU()
    {
        uint32_t physicalDeviceCount;
        RHICheck(vkEnumeratePhysicalDevices(m_instance, &physicalDeviceCount, nullptr));
        ASSERT(physicalDeviceCount > 0, "No gpu support vulkan on your computer.");

        std::vector<VkPhysicalDevice> physicalDevices;
        physicalDevices.resize(physicalDeviceCount);
        RHICheck(vkEnumeratePhysicalDevices(m_instance, &physicalDeviceCount, physicalDevices.data()));
        ASSERT(!physicalDevices.empty(), "No gpu on your computer.");

        // Find discrete gpu first.
        for (auto& gpu : physicalDevices)
        {
            VkPhysicalDeviceProperties deviceProperties;
            vkGetPhysicalDeviceProperties(gpu, &deviceProperties);
            if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
            {
                m_gpu = gpu;
                return;
            }
        }

        LOG_RHI_WARN("No discrete gpu found, using default gpu.");

        m_gpu = physicalDevices[0];
        return;
    }

    void VulkanContext::queryGPUInfo()
    {
        ASSERT(m_gpu != VK_NULL_HANDLE, "You must finish gpu select before query gpu infos.");

        vkGetPhysicalDeviceProperties(m_gpu, &m_deviceProperties);
        LOG_RHI_INFO("Select gpu {0}.", m_deviceProperties.deviceName);

        vkGetPhysicalDeviceMemoryProperties(m_gpu, &m_memoryProperties);

        static auto getPhysicalDeviceProperties2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2KHR>(vkGetInstanceProcAddr(m_instance, "vkGetPhysicalDeviceProperties2KHR"));
        CHECK(getPhysicalDeviceProperties2);
        {
            VkPhysicalDeviceProperties2KHR deviceProperties{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2_KHR };
            m_descriptorIndexingProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES_EXT };

            


            m_accelerationStructureProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR };

            deviceProperties.pNext = &m_descriptorIndexingProperties;
            m_descriptorIndexingProperties.pNext = &m_accelerationStructureProperties;


            getPhysicalDeviceProperties2(m_gpu, &deviceProperties);
        }

        
    }
}