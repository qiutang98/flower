#pragma once

#include "RHICommon.h"

namespace Flower
{
	class VulkanBuffer;



	class AccelerateStructure
	{
	private:
		VkAccelerationStructureKHR m_handle = VK_NULL_HANDLE;
		uint64_t m_deviceAddress = 0;

		std::shared_ptr<VulkanBuffer> m_buffer = nullptr;
		std::shared_ptr<VulkanBuffer> m_scratchBuffer;

	private: 
		void release();


	public:
		VulkanBuffer& getBuffer() { return *m_buffer.get(); }
		const VulkanBuffer& getBuffer() const { return *m_buffer.get(); }
		uint64_t getDeviceAddress() const { return m_deviceAddress; }

		VulkanBuffer& getScratchBuffer() { return *m_scratchBuffer.get(); }
		const VulkanBuffer& getScratchBuffer() const { return *m_scratchBuffer.get(); }

		VkAccelerationStructureKHR getHandle() const { return m_handle; }
		const VkAccelerationStructureKHR* get() const { return &m_handle; }

		~AccelerateStructure();

		void cleanScratchBuffer();

		void create(const std::string& name, VkAccelerationStructureTypeKHR type, VkAccelerationStructureBuildSizesInfoKHR buildSizeInfo);
	};
}