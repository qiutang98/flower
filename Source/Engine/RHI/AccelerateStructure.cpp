#include "Pch.h"
#include "AccelerateStructure.h"
#include "RHI.h"

namespace Flower
{
	AccelerateStructure::~AccelerateStructure()
	{
		if (m_handle != VK_NULL_HANDLE)
		{
			RHI::DestroyAccelerationStructure(RHI::Device, m_handle, nullptr);
		}
	}
	uint64_t AccelerateStructure::addTriangleGeometry(
		VulkanBuffer* vertexBuffer, 
		VulkanBuffer* indexBuffer, 
		VulkanBuffer* transformBuffer, 
		uint32_t triangleCount, 
		uint32_t maxVertex, 
		VkDeviceSize vertexStride, 
		uint32_t transformOffset, 
		VkFormat vertexFormat, 
		VkGeometryFlagsKHR flags, 
		uint64_t vertexBufferDataAddress, 
		uint64_t indexBufferDataAddress, 
		uint64_t transformBufferDataAddress)
	{
		VkAccelerationStructureGeometryKHR geometry{};
		geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
		geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
		geometry.flags = flags;
		geometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
		geometry.geometry.triangles.vertexFormat = vertexFormat;
		geometry.geometry.triangles.maxVertex = maxVertex;
		geometry.geometry.triangles.vertexStride = vertexStride;
		geometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
		geometry.geometry.triangles.vertexData.deviceAddress = vertexBufferDataAddress == 0 ? vertexBuffer->getDeviceAddress() : vertexBufferDataAddress;
		geometry.geometry.triangles.indexData.deviceAddress = indexBufferDataAddress == 0 ? indexBuffer->getDeviceAddress() : indexBufferDataAddress;
		geometry.geometry.triangles.transformData.deviceAddress = transformBufferDataAddress == 0 ? transformBuffer->getDeviceAddress() : transformBufferDataAddress;

		uint64_t index = m_geometries.size();
		m_geometries.insert({ index, { geometry, triangleCount, transformOffset} });
		return index;
	}

	void AccelerateStructure::updateTriangleGeometry(
		uint64_t triangleUUID, 
		VulkanBuffer* vertexBuffer, 
		VulkanBuffer* indexBuffer, 
		VulkanBuffer* transformBuffer, 
		uint32_t triangleCount, 
		uint32_t maxVertex, 
		VkDeviceSize vertexStride, 
		uint32_t transformOffset, 
		VkFormat vertexFormat, 
		VkGeometryFlagsKHR flags, 
		uint64_t vertexBufferDataAddress, 
		uint64_t indexBufferDataAddress, 
		uint64_t transformBufferDataAddress)
	{
		VkAccelerationStructureGeometryKHR* geometry = &m_geometries[triangleUUID].geometry;
		geometry->sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
		geometry->geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
		geometry->flags = flags;
		geometry->geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
		geometry->geometry.triangles.vertexFormat = vertexFormat;
		geometry->geometry.triangles.maxVertex = maxVertex;
		geometry->geometry.triangles.vertexStride = vertexStride;
		geometry->geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
		geometry->geometry.triangles.vertexData.deviceAddress = vertexBufferDataAddress == 0 ? vertexBuffer->getDeviceAddress() : vertexBufferDataAddress;
		geometry->geometry.triangles.indexData.deviceAddress = indexBufferDataAddress == 0 ? indexBuffer->getDeviceAddress() : indexBufferDataAddress;
		geometry->geometry.triangles.transformData.deviceAddress = transformBufferDataAddress == 0 ? transformBuffer->getDeviceAddress() : transformBufferDataAddress;

		m_geometries[triangleUUID].primitive_count = triangleCount;
		m_geometries[triangleUUID].transform_offset = transformOffset;
		m_geometries[triangleUUID].updated = true;
	}

	uint64_t AccelerateStructure::addInstanceGeometry(
		VulkanBuffer* instanceBuffer,
		uint32_t instanceCount,
		uint32_t transformOffset,
		VkGeometryFlagsKHR flags)
	{

		VkAccelerationStructureGeometryKHR geometry{};
		geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
		geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
		geometry.flags = flags;
		geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
		geometry.geometry.instances.arrayOfPointers = VK_FALSE;
		geometry.geometry.instances.data.deviceAddress = instanceBuffer->getDeviceAddress();

		uint64_t index = m_geometries.size();
		m_geometries.insert({ index, {geometry, instanceCount, transformOffset} });
		return index;
	}

	void AccelerateStructure::updateInstanceGeometry(
		uint64_t instanceUID, 
		VulkanBuffer* instanceBuffer,
		uint32_t instanceCount,
		uint32_t transformOffset,
		VkGeometryFlagsKHR flags)
	{
		VkAccelerationStructureGeometryKHR* geometry = &m_geometries[instanceUID].geometry;
		geometry->sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
		geometry->geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
		geometry->flags = flags;
		geometry->geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
		geometry->geometry.instances.arrayOfPointers = VK_FALSE;
		geometry->geometry.instances.data.deviceAddress = instanceBuffer->getDeviceAddress();

		m_geometries[instanceUID].primitive_count = instanceCount;
		m_geometries[instanceUID].transform_offset = transformOffset;
		m_geometries[instanceUID].updated = true;
	}

	void AccelerateStructure::build(VkCommandBuffer cmd, VkBuildAccelerationStructureFlagsKHR flags, VkBuildAccelerationStructureModeKHR mode)
	{
		CHECK(!m_geometries.empty() && "Geometry should no empty.");

		std::vector<VkAccelerationStructureGeometryKHR>       acceleration_structure_geometries;
		std::vector<VkAccelerationStructureBuildRangeInfoKHR> acceleration_structure_build_range_infos;
		std::vector<uint32_t>                                 primitive_counts;
		for (auto& geometry : m_geometries)
		{
			if (mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR && !geometry.second.updated)
			{
				continue;
			}
			acceleration_structure_geometries.push_back(geometry.second.geometry);
			// Infer build range info from geometry
			VkAccelerationStructureBuildRangeInfoKHR build_range_info;
			build_range_info.primitiveCount = geometry.second.primitive_count;
			build_range_info.primitiveOffset = 0;
			build_range_info.firstVertex = 0;
			build_range_info.transformOffset = geometry.second.transform_offset;
			acceleration_structure_build_range_infos.push_back(build_range_info);
			primitive_counts.push_back(geometry.second.primitive_count);
			geometry.second.updated = false;
		}

		VkAccelerationStructureBuildGeometryInfoKHR build_geometry_info{};
		build_geometry_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
		build_geometry_info.type = m_type;
		build_geometry_info.flags = flags;
		build_geometry_info.mode = mode;
		if (mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR && m_handle != VK_NULL_HANDLE)
		{
			build_geometry_info.srcAccelerationStructure = m_handle;
			build_geometry_info.dstAccelerationStructure = m_handle;
		}
		build_geometry_info.geometryCount = static_cast<uint32_t>(acceleration_structure_geometries.size());
		build_geometry_info.pGeometries = acceleration_structure_geometries.data();

		// Get required build sizes
		m_size.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
		RHI::GetAccelerationStructureBuildSizes(
			RHI::Device,
			VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
			&build_geometry_info,
			primitive_counts.data(),
			&m_size);

		// Create a buffer for the acceleration structure
		if (!m_buffer || m_buffer->getSize() != m_size.accelerationStructureSize)
		{
			m_buffer = VulkanBuffer::create(
				"RT accelerate structure",
				VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				EVMAUsageFlags::GPUOnly,
				m_size.accelerationStructureSize);

			VkAccelerationStructureCreateInfoKHR acceleration_structure_create_info{};
			acceleration_structure_create_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
			acceleration_structure_create_info.buffer = m_buffer->getVkBuffer();
			acceleration_structure_create_info.size = m_size.accelerationStructureSize;
			acceleration_structure_create_info.type = m_type;
			RHICheck(RHI::CreateAccelerationStructure(RHI::Device, &acceleration_structure_create_info, nullptr, &m_handle));
		}

		// Get the acceleration structure's handle
		VkAccelerationStructureDeviceAddressInfoKHR acceleration_device_address_info{};
		acceleration_device_address_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
		acceleration_device_address_info.accelerationStructure = m_handle;
		m_deviceAddress = RHI::GetAccelerationStructureDeviceAddress(RHI::Device, &acceleration_device_address_info);

		// Create a scratch buffer as a temporary storage for the acceleration structure build
		m_scratchBuffer = VulkanBuffer::createRTScratchBuffer("ScatchBuffer", m_size.buildScratchSize);

		build_geometry_info.scratchData.deviceAddress = m_scratchBuffer->getDeviceAddress();
		build_geometry_info.dstAccelerationStructure = m_handle;

		auto  as_build_range_infos = &*acceleration_structure_build_range_infos.data();

		RHI::CmdBuildAccelerationStructures(
			cmd,
			1,
			&build_geometry_info,
			&as_build_range_infos);
	}
}