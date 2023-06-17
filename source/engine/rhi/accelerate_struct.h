#pragma once
#include "rhi_misc.h"
#include "resource.h"

namespace engine
{
	struct AccelKHR
	{
		VkAccelerationStructureCreateInfoKHR createInfo{};
		VkAccelerationStructureKHR accel = VK_NULL_HANDLE;
		std::shared_ptr<VulkanBuffer> buffer = nullptr;

		void release();
		void create(VkAccelerationStructureCreateInfoKHR& accelInfo);
	};

	struct BuildAccelerationStructure
	{
		VkAccelerationStructureBuildGeometryInfoKHR buildInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
		VkAccelerationStructureBuildSizesInfoKHR sizeInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
		const VkAccelerationStructureBuildRangeInfoKHR* rangeInfo;
		AccelKHR as;
		AccelKHR cleanupAS;
	};

	class TLASBuilder : NonCopyable
	{
	public:
		~TLASBuilder() { destroy(); }

		void destroy();

		// TLAS
		bool isInit() const { return m_bInit; }
		const VkAccelerationStructureKHR& getAccelerationStructure() const { return m_tlas.accel; }

		void buildTlas(
			VkCommandBuffer cmdBuf, 
			const std::vector<VkAccelerationStructureInstanceKHR>& instances,
			bool update,
			VkBuildAccelerationStructureFlagsKHR flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);

	protected:
		// Creating the TLAS, called by buildTlas 
		void cmdCreateTlas(VkCommandBuffer cmdBuf,
			uint32_t countInstance,   
			VkDeviceAddress instBufferAddr, 
			VkBuildAccelerationStructureFlagsKHR flags, 
			bool update
		);

		bool m_bInit = false;
		AccelKHR m_tlas;
		std::unique_ptr<VulkanBuffer> m_scratchBuffer;
	};

	class BLASBuilder : NonCopyable
	{
	public:
		struct BlasInput
		{
			std::vector<VkAccelerationStructureGeometryKHR> asGeometry;
			std::vector<VkAccelerationStructureBuildRangeInfoKHR> asBuildOffsetInfo;
			VkBuildAccelerationStructureFlagsKHR flags{ 0 };
		};

		~BLASBuilder() { destroy(); }

		void destroy();

		bool isInit() const { return m_bInit; }
		VkDeviceAddress getBlasDeviceAddress(uint32_t blasId);

		void build(const std::vector<BlasInput>& input,
			VkBuildAccelerationStructureFlagsKHR flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
		void update(uint32_t blasIdx, BlasInput& blas, VkBuildAccelerationStructureFlagsKHR flags);

	protected:
		bool m_bInit = false;

		std::vector<AccelKHR> m_blas{ };

		void cmdCreateBlas(
			VkCommandBuffer cmdBuf,
			std::vector<uint32_t> indices,
			std::vector<BuildAccelerationStructure>& buildAs,
			VkDeviceAddress scratchAddress,
			VkQueryPool queryPool);

		void cmdCompactBlas(
			VkCommandBuffer cmdBuf, 
			std::vector<uint32_t> indices, 
			std::vector<BuildAccelerationStructure>& buildAs, 
			VkQueryPool queryPool);

		void destroyNonCompacted(
			std::vector<uint32_t> indices, 
			std::vector<BuildAccelerationStructure>& buildAs);
	};
}