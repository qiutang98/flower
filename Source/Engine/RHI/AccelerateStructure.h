#pragma once

#include "RHICommon.h"

namespace Flower
{
	class VulkanBuffer;

	class AccelerateStructure
	{
	private:
		struct Geometry
		{
			VkAccelerationStructureGeometryKHR geometry{};
			uint32_t primitiveCount{};
			uint32_t transformOffset{};
			bool updated = false;
		};
		std::map<uint64_t, Geometry> m_geometries { };

		VkAccelerationStructureKHR m_handle = VK_NULL_HANDLE;
		uint64_t m_deviceAddress = 0;
		VkAccelerationStructureTypeKHR m_type;

		VkAccelerationStructureBuildSizesInfoKHR m_size { };

		std::shared_ptr<VulkanBuffer> m_buffer;
		std::shared_ptr<VulkanBuffer> m_scratchBuffer;

	public:
		void resetGeometries() { m_geometries.clear(); }
		VulkanBuffer* getBuffer() const { return m_buffer.get(); }

		uint64_t getDeviceAddress() const { return m_deviceAddress; }
		VkAccelerationStructureKHR getHandle() const { return m_handle; }
		const VkAccelerationStructureKHR* get() const { return &m_handle; }

		AccelerateStructure(VkAccelerationStructureTypeKHR type) :
			m_type(type)
		{

		}

		~AccelerateStructure();

		void cleanScratchBuffer();

		uint64_t addTriangleGeometry(
			VulkanBuffer* vertexBuffer, 
			VulkanBuffer* indexBuffer, 
			VulkanBuffer* transformBuffer,
			uint32_t triangleCount, // indices count.
			uint32_t maxVertex, // vertex count.
			VkDeviceSize vertexStride,
			uint32_t transformOffset = 0,
			VkFormat vertexFormat = VK_FORMAT_R32G32B32_SFLOAT, // Position only.
			VkGeometryFlagsKHR flags = VK_GEOMETRY_OPAQUE_BIT_KHR,
			uint64_t vertexBufferDataAddress = 0,
			uint64_t indexBufferDataAddress = 0,
			uint64_t transformBufferDataAddress = 0);

		void updateTriangleGeometry(
			uint64_t triangleUUID, 
			VulkanBuffer* vertexBuffer,
			VulkanBuffer* indexBuffer,
			VulkanBuffer* transformBuffer,
			uint32_t triangleCount,
			uint32_t maxVertex,
			VkDeviceSize vertexStride,
			uint32_t transformOffset = 0,
			VkFormat vertexFormat = VK_FORMAT_R32G32B32_SFLOAT, // Position only.
			VkGeometryFlagsKHR flags = VK_GEOMETRY_OPAQUE_BIT_KHR,
			uint64_t vertexBufferDataAddress = 0,
			uint64_t indexBufferDataAddress = 0,
			uint64_t transformBufferDataAddress = 0);

		uint64_t addInstanceGeometry(
			VulkanBuffer* instanceBuffer,
			uint32_t instanceCount,
			uint32_t transformOffset = 0,
			VkGeometryFlagsKHR flags = VK_GEOMETRY_OPAQUE_BIT_KHR);

		void updateInstanceGeometry(
			uint64_t instanceUID, 
			VulkanBuffer* instanceBuffer,
			uint32_t instanceCount,
			uint32_t transformOffset = 0,
			VkGeometryFlagsKHR flags = VK_GEOMETRY_OPAQUE_BIT_KHR);

		void build(
			VkCommandBuffer cmd,
			VkBuildAccelerationStructureFlagsKHR flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
			VkBuildAccelerationStructureModeKHR  mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR);

		void buildAndFlush(
			VkBuildAccelerationStructureFlagsKHR flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
			VkBuildAccelerationStructureModeKHR  mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR);
	};
}