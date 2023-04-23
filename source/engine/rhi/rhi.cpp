#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include "rhi.h"


namespace engine
{
	VkResult createAccelerationStructure(
		const VkAccelerationStructureCreateInfoKHR* pCreateInfo, 
		const VkAllocationCallbacks* pAllocator, 
		VkAccelerationStructureKHR* pAccelerationStructure)
	{
		static auto ptr = (PFN_vkCreateAccelerationStructureKHR)vkGetDeviceProcAddr(getContext()->getDevice(), "vkCreateAccelerationStructureKHR");
		return ptr(getContext()->getDevice(), pCreateInfo, pAllocator, pAccelerationStructure);
	}

	void destroyAccelerationStructure(VkAccelerationStructureKHR accelerationStructure, const VkAllocationCallbacks* pAllocator)
	{
		static auto ptr = (PFN_vkDestroyAccelerationStructureKHR)vkGetDeviceProcAddr(getContext()->getDevice(), "vkDestroyAccelerationStructureKHR");
		ptr(getContext()->getDevice(), accelerationStructure, pAllocator);
	}

	void cmdBuildAccelerationStructures(VkCommandBuffer commandBuffer, uint32_t infoCount, const VkAccelerationStructureBuildGeometryInfoKHR* pInfos, const VkAccelerationStructureBuildRangeInfoKHR* const* ppBuildRangeInfos)
	{
		static auto ptr = (PFN_vkCmdBuildAccelerationStructuresKHR)vkGetDeviceProcAddr(getContext()->getDevice(), "vkCmdBuildAccelerationStructuresKHR");
		ptr(commandBuffer, infoCount, pInfos, ppBuildRangeInfos);
	}

	VkDeviceAddress getAccelerationStructureDeviceAddress(const VkAccelerationStructureDeviceAddressInfoKHR* pInfo)
	{
		static auto ptr = (PFN_vkGetAccelerationStructureDeviceAddressKHR)vkGetDeviceProcAddr(getContext()->getDevice(), "vkGetAccelerationStructureDeviceAddressKHR");

		return ptr(getContext()->getDevice(), pInfo);
	}

	void getAccelerationStructureBuildSizesKHR(VkAccelerationStructureBuildTypeKHR buildType, const VkAccelerationStructureBuildGeometryInfoKHR* pBuildInfo, const uint32_t* pMaxPrimitiveCounts, VkAccelerationStructureBuildSizesInfoKHR* pSizeInfo)
	{
		static auto ptr = (PFN_vkGetAccelerationStructureBuildSizesKHR)
			vkGetDeviceProcAddr(getContext()->getDevice(), "vkGetAccelerationStructureBuildSizesKHR");

		ptr(getContext()->getDevice(), buildType, pBuildInfo, pMaxPrimitiveCounts, pSizeInfo);
	}

	VkResult buildAccelerationStructuresKHR(VkDeferredOperationKHR deferredOperation, uint32_t infoCount, const VkAccelerationStructureBuildGeometryInfoKHR* pInfos, const VkAccelerationStructureBuildRangeInfoKHR* const* ppBuildRangeInfos)
	{
		static auto ptr = (PFN_vkBuildAccelerationStructuresKHR)
			vkGetDeviceProcAddr(getContext()->getDevice(), "vkBuildAccelerationStructuresKHR");
		return ptr(getContext()->getDevice(), deferredOperation, infoCount, pInfos, ppBuildRangeInfos);
	}

	void cmdTraceRaysKHR(VkCommandBuffer commandBuffer, const VkStridedDeviceAddressRegionKHR* pRaygenShaderBindingTable, const VkStridedDeviceAddressRegionKHR* pMissShaderBindingTable, const VkStridedDeviceAddressRegionKHR* pHitShaderBindingTable, const VkStridedDeviceAddressRegionKHR* pCallableShaderBindingTable, uint32_t width, uint32_t height, uint32_t depth)
	{
		static auto ptr = (PFN_vkCmdTraceRaysKHR)
			vkGetDeviceProcAddr(getContext()->getDevice(), "vkCmdTraceRaysKHR");

		ptr(commandBuffer, pRaygenShaderBindingTable, pMissShaderBindingTable, pHitShaderBindingTable, pCallableShaderBindingTable, width, height, depth);
	}

	VkResult createRayTracingPipelinesKHR(VkDeferredOperationKHR deferredOperation, VkPipelineCache pipelineCache, uint32_t createInfoCount, const VkRayTracingPipelineCreateInfoKHR* pCreateInfos, const VkAllocationCallbacks* pAllocator, VkPipeline* pPipelines)
	{
		static auto ptr = (PFN_vkCreateRayTracingPipelinesKHR)
			vkGetDeviceProcAddr(getContext()->getDevice(), "vkCreateRayTracingPipelinesKHR");

		return ptr(getContext()->getDevice(), deferredOperation, pipelineCache, createInfoCount, pCreateInfos, pAllocator, pPipelines);
	}

	VkResult getRayTracingShaderGroupHandlesKHR(VkPipeline pipeline, uint32_t firstGroup, uint32_t groupCount, size_t dataSize, void* pData)
	{
		static auto ptr = (PFN_vkGetRayTracingShaderGroupHandlesKHR)
			vkGetDeviceProcAddr(getContext()->getDevice(), "vkGetRayTracingShaderGroupHandlesKHR");
		return ptr(getContext()->getDevice(), pipeline, firstGroup, groupCount, dataSize, pData);
	}
	void cmdCopyAccelerationStructure(VkCommandBuffer commandBuffer, const VkCopyAccelerationStructureInfoKHR* pInfo)
	{
		static auto ptr = (PFN_vkCmdCopyAccelerationStructureKHR)
			vkGetDeviceProcAddr(getContext()->getDevice(), "vkCmdCopyAccelerationStructureKHR");
		return ptr(commandBuffer, pInfo);
	}

	void cmdWriteAccelerationStructuresProperties(VkCommandBuffer commandBuffer, uint32_t accelerationStructureCount, const VkAccelerationStructureKHR* pAccelerationStructures, VkQueryType queryType, VkQueryPool queryPool, uint32_t firstQuery)
	{
		static auto ptr = (PFN_vkCmdWriteAccelerationStructuresPropertiesKHR)
			vkGetDeviceProcAddr(getContext()->getDevice(), "vkCmdWriteAccelerationStructuresPropertiesKHR");
		return ptr(commandBuffer, accelerationStructureCount, pAccelerationStructures, queryType, queryPool, firstQuery);
	}
}

