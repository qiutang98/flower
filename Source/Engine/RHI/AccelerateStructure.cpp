#include "Pch.h"
#include "AccelerateStructure.h"
#include "RHI.h"

namespace Flower
{
	AccelerateStructure::~AccelerateStructure()
	{
		release();
	}

	void AccelerateStructure::release()
	{
		if (m_handle != VK_NULL_HANDLE)
		{
			RHI::DestroyAccelerationStructure(RHI::Device, m_handle, nullptr);
		}
		cleanScratchBuffer();
		m_buffer = nullptr;
		m_handle = VK_NULL_HANDLE;
	}

	void AccelerateStructure::cleanScratchBuffer()
	{
		m_scratchBuffer = nullptr;
	}

	void AccelerateStructure::create(
		const std::string& name,
		VkAccelerationStructureTypeKHR type, 
		VkAccelerationStructureBuildSizesInfoKHR buildSizeInfo)
	{
		CHECK(m_buffer == nullptr && "You should only init and create once.");

		m_buffer = VulkanBuffer::create2(
			name.c_str(),
			VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			{},
			buildSizeInfo.accelerationStructureSize
		);
		

		// Acceleration structure
		VkAccelerationStructureCreateInfoKHR asCreateInfo{};
		asCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
		asCreateInfo.buffer = m_buffer->getVkBuffer();
		asCreateInfo.size = buildSizeInfo.accelerationStructureSize;
		asCreateInfo.type = type;
		RHICheck(RHI::CreateAccelerationStructure(RHI::Device, &asCreateInfo, nullptr, &m_handle));

		// AS device address
		VkAccelerationStructureDeviceAddressInfoKHR accelerationDeviceAddressInfo{};
		accelerationDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
		accelerationDeviceAddressInfo.accelerationStructure = m_handle;
		m_deviceAddress = RHI::GetAccelerationStructureDeviceAddress(RHI::Device, &accelerationDeviceAddressInfo);


		// Also prepare scratch buffer.
		m_scratchBuffer = VulkanBuffer::create2(
			(name + "_scratch").c_str(),
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			{},
			buildSizeInfo.buildScratchSize
		);
	}

}