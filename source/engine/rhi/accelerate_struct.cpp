#include "accelerate_struct.h"
#include "rhi.h"

#include <numeric>

namespace engine
{
    static std::string getRuntimeUniqueGPUASName(const std::string& in)
    {
        static size_t GRuntimeId = 0;
        GRuntimeId++;
        return std::format("AS: {}. {}.", GRuntimeId, in);
    }

    void AccelKHR::release()
    {
        if (accel != VK_NULL_HANDLE)
        {
            destroyAccelerationStructure(accel, nullptr);
            accel = VK_NULL_HANDLE;
        }
        buffer = nullptr;
    }

    void AccelKHR::create(VkAccelerationStructureCreateInfoKHR& accelInfo)
    {
        buffer = std::make_shared<VulkanBuffer>(
            getContext(),
            getRuntimeUniqueGPUASName("AccelBuffer"),
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            0,
            accelInfo.size
        );

        // Setting the buffer
        accelInfo.buffer = buffer->getVkBuffer();

        // Create the acceleration structure
        createAccelerationStructure(&accelInfo, nullptr, &accel);

        // Cache create info.
        createInfo = accelInfo;
    }

    void TLASBuilder::destroy()
    {
        if (m_bInit)
        {
            getContext()->waitDeviceIdle();
            m_bInit = false;

            m_tlas.release();
            m_scratchBuffer = nullptr;
        }
    }

    void TLASBuilder::buildTlas(
        VkCommandBuffer cmdBuf,
        const std::vector<VkAccelerationStructureInstanceKHR>& instances, 
        bool update,
        VkBuildAccelerationStructureFlagsKHR flags)
    {
        bool bUpdate = update && m_bInit;

        // Copy instance matrix to buffer.
        auto instanceGPU = getContext()->getBufferParameters().getParameter("TLAS_Instances", 
            sizeof(instances[0]) * instances.size(),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
            VK_DESCRIPTOR_TYPE_MAX_ENUM, VulkanBuffer::getStageCopyForUploadBufferFlags());
        instanceGPU->updateDataPtr((void*)instances.data());

        VkDeviceAddress instBufferAddr = instanceGPU->getBuffer()->getDeviceAddress();

        // Cannot call buildTlas twice except to update.
        uint32_t countInstance = static_cast<uint32_t>(instances.size());

        // Creating the TLAS
        cmdCreateTlas(cmdBuf, countInstance, instBufferAddr, flags, bUpdate);

        m_bInit = true;
    }

    void TLASBuilder::cmdCreateTlas(
        VkCommandBuffer cmdBuf, 
        uint32_t countInstance, 
        VkDeviceAddress instBufferAddr, 
        VkBuildAccelerationStructureFlagsKHR flags, 
        bool update)
    {
        // Wraps a device pointer to the above uploaded instances.
        VkAccelerationStructureGeometryInstancesDataKHR instancesVk{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR };
        instancesVk.data.deviceAddress = instBufferAddr;

        // Put the above into a VkAccelerationStructureGeometryKHR. We need to put the instances struct in a union and label it as instance data.
        VkAccelerationStructureGeometryKHR topASGeometry{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
        topASGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        topASGeometry.geometry.instances = instancesVk;

        // Find sizes
        VkAccelerationStructureBuildGeometryInfoKHR buildInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
        VkAccelerationStructureBuildSizesInfoKHR sizeInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };

        buildInfo.flags = flags | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
        buildInfo.geometryCount = 1;
        buildInfo.pGeometries = &topASGeometry;
        buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;

        {

            buildInfo.mode = update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            getAccelerationStructureBuildSizesKHR(VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &countInstance, &sizeInfo);
        }
        

        if (update)
        {
            CHECK(m_tlas.accel);
            CHECK(m_scratchBuffer);

            // Size not match, rebuild.
            if (m_scratchBuffer->getSize() != sizeInfo.buildScratchSize || 
                m_tlas.createInfo.size     != sizeInfo.accelerationStructureSize)
            {
                update = false;
                destroy();
            }

            // Need rebuild, re-compute size info and build info.
            if (!update)
            {
                buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
                buildInfo.flags = flags;
                getAccelerationStructureBuildSizesKHR(VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &countInstance, &sizeInfo);
            }
        }

        // Create TLAS
        if (m_tlas.accel == VK_NULL_HANDLE)
        {
            VkAccelerationStructureCreateInfoKHR createInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
            createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
            createInfo.size = sizeInfo.accelerationStructureSize;

            m_tlas.create(createInfo);
        }

        // Prepare tlas scratch buffer if no exist.
        if (m_scratchBuffer == nullptr)
        {
            // Allocate the scratch memory
            m_scratchBuffer = std::make_unique<VulkanBuffer>(
                getContext(),
                getRuntimeUniqueGPUASName("tlas_scratch"),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                0,
                sizeInfo.buildScratchSize
            );
        }

        VkDeviceAddress scratchAddress = m_scratchBuffer->getDeviceAddress();

        // Update build information
        buildInfo.srcAccelerationStructure = update ? m_tlas.accel : VK_NULL_HANDLE;
        buildInfo.dstAccelerationStructure = m_tlas.accel;
        buildInfo.scratchData.deviceAddress = scratchAddress;

        // Build Offsets info: n instances
        VkAccelerationStructureBuildRangeInfoKHR buildOffsetInfo{ countInstance, 0, 0, 0 };
        const VkAccelerationStructureBuildRangeInfoKHR* pBuildOffsetInfo = &buildOffsetInfo;

        // Build the TLAS
        cmdBuildAccelerationStructures(cmdBuf, 1, &buildInfo, &pBuildOffsetInfo);

        // Barrier.
        VkMemoryBarrier barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
        barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &barrier, 0, nullptr, 0, nullptr);

        if (!update)
        {
            vkQueueWaitIdle(getContext()->getMajorGraphicsQueue());
        }
    }


	void BLASBuilder::destroy()
	{
        getContext()->waitDeviceIdle();
        m_bInit = false;

		for (auto& b : m_blas)
		{
            b.release();
		}
		m_blas.clear();
	}

	VkDeviceAddress BLASBuilder::getBlasDeviceAddress(uint32_t inBlasId)
	{
        CHECK(m_bInit);
		CHECK(size_t(inBlasId) < m_blas.size());

        VkAccelerationStructureDeviceAddressInfoKHR addressInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR };
        addressInfo.accelerationStructure = m_blas[inBlasId].accel;
        return getAccelerationStructureDeviceAddress(&addressInfo);
	}

	void BLASBuilder::build(const std::vector<BlasInput>& input, VkBuildAccelerationStructureFlagsKHR flags)
	{
        m_bInit = true;

        auto nbBlas = static_cast<uint32_t>(input.size());
        VkDeviceSize asTotalSize{ 0 };     // Memory size of all allocated BLAS
        uint32_t nbCompactions{ 0 };   // Nb of BLAS requesting compaction
        VkDeviceSize maxScratchSize{ 0 };  // Largest scratch size

        // Preparing the information for the acceleration build commands.
        std::vector<BuildAccelerationStructure> buildAs(nbBlas);
        for (uint32_t idx = 0; idx < nbBlas; idx++)
        {
            // Filling partially the VkAccelerationStructureBuildGeometryInfoKHR for querying the build sizes.
            // Other information will be filled in the createBlas (see #2)
            buildAs[idx].buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            buildAs[idx].buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            buildAs[idx].buildInfo.flags = input[idx].flags | flags;
            buildAs[idx].buildInfo.geometryCount = static_cast<uint32_t>(input[idx].asGeometry.size());
            buildAs[idx].buildInfo.pGeometries = input[idx].asGeometry.data();

            // Build range information
            buildAs[idx].rangeInfo = input[idx].asBuildOffsetInfo.data();

            // Finding sizes to create acceleration structures and scratch
            std::vector<uint32_t> maxPrimCount(input[idx].asBuildOffsetInfo.size());
            for (auto tt = 0; tt < input[idx].asBuildOffsetInfo.size(); tt++)
                maxPrimCount[tt] = input[idx].asBuildOffsetInfo[tt].primitiveCount;  // Number of primitives/triangles
            getAccelerationStructureBuildSizesKHR(VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                &buildAs[idx].buildInfo, maxPrimCount.data(), &buildAs[idx].sizeInfo);

            // Extra info
            asTotalSize += buildAs[idx].sizeInfo.accelerationStructureSize;
            maxScratchSize = std::max(maxScratchSize, buildAs[idx].sizeInfo.buildScratchSize);
            nbCompactions += hasFlag(buildAs[idx].buildInfo.flags, VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR);
        }

        // Allocate the scratch buffers holding the temporary data of the acceleration structure builder
        std::unique_ptr<VulkanBuffer> scratchBuffer = std::make_unique<VulkanBuffer>(
            getContext(),
            getRuntimeUniqueGPUASName("scratch"),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            0,
            maxScratchSize
        );
        VkDeviceAddress  scratchAddress = scratchBuffer->getDeviceAddress();

        // Allocate a query pool for storing the needed size for every BLAS compaction.
        VkQueryPool queryPool{ VK_NULL_HANDLE };
        if (nbCompactions > 0)  // Is compaction requested?
        {
            CHECK(nbCompactions == nbBlas);  // Don't allow mix of on/off compaction
            VkQueryPoolCreateInfo qpci{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
            qpci.queryCount = nbBlas;
            qpci.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
            vkCreateQueryPool(getContext()->getDevice(), &qpci, nullptr, &queryPool);
        }

        // Batching creation/compaction of BLAS to allow staying in restricted amount of memory
        std::vector<uint32_t> indices;  // Indices of the BLAS to create
        VkDeviceSize          batchSize{ 0 };
        VkDeviceSize          batchLimit{ 256'000'000 };  // 256 MB
        for (uint32_t idx = 0; idx < nbBlas; idx++)
        {
            indices.push_back(idx);
            batchSize += buildAs[idx].sizeInfo.accelerationStructureSize;
            // Over the limit or last BLAS element
            if (batchSize >= batchLimit || idx == nbBlas - 1)
            {
                getContext()->executeImmediately(
                    getContext()->getMajorComputeCommandPool(),
                    getContext()->getMajorComputeQueue(), [&](VkCommandBuffer cmd)
                {
                    cmdCreateBlas(cmd, indices, buildAs, scratchAddress, queryPool);
                });

                if (queryPool)
                {
                    getContext()->executeImmediately(
                        getContext()->getMajorComputeCommandPool(),
                        getContext()->getMajorComputeQueue(), [&](VkCommandBuffer cmd)
                    {
                        cmdCompactBlas(cmd, indices, buildAs, queryPool);
                    });
                    // Destroy the non-compacted version
                    destroyNonCompacted(indices, buildAs);
                }
                // Reset

                batchSize = 0;
                indices.clear();
            }
        }

        // Logging reduction
        if (queryPool)
        {
            VkDeviceSize compactSize = std::accumulate(buildAs.begin(), buildAs.end(), 0ULL, [](const auto& a, const auto& b) {
                return a + b.sizeInfo.accelerationStructureSize;
                });
            LOG_TRACE(" RT BLAS: reducing from: {0} KB to: {1}KB, Save {2}KB({3}% smaller).", 
                asTotalSize / 1024.0f, 
                compactSize / 1024.0f,
                (asTotalSize - compactSize) / 1024.0f, 
                (asTotalSize - compactSize) / float(asTotalSize) * 100.f);
        }

        // Keeping all the created acceleration structures
        for (auto& b : buildAs)
        {
            m_blas.emplace_back(b.as);
        }

        // Clean up
        vkDestroyQueryPool(getContext()->getDevice(), queryPool, nullptr);
	}

	void BLASBuilder::update(uint32_t blasIdx, BlasInput& blas, VkBuildAccelerationStructureFlagsKHR flags)
	{
        CHECK(m_bInit);
        CHECK(size_t(blasIdx) < m_blas.size());

        // Preparing all build information, acceleration is filled later
        VkAccelerationStructureBuildGeometryInfoKHR buildInfos{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
        buildInfos.flags = flags;
        buildInfos.geometryCount = (uint32_t)blas.asGeometry.size();
        buildInfos.pGeometries = blas.asGeometry.data();
        buildInfos.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;  // UPDATE
        buildInfos.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        buildInfos.srcAccelerationStructure = m_blas[blasIdx].accel;  // UPDATE
        buildInfos.dstAccelerationStructure = m_blas[blasIdx].accel;

        // Find size to build on the device
        std::vector<uint32_t> maxPrimCount(blas.asBuildOffsetInfo.size());
        for (auto tt = 0; tt < blas.asBuildOffsetInfo.size(); tt++)
            maxPrimCount[tt] = blas.asBuildOffsetInfo[tt].primitiveCount;  // Number of primitives/triangles
        VkAccelerationStructureBuildSizesInfoKHR sizeInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
        getAccelerationStructureBuildSizesKHR(VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfos, maxPrimCount.data(), &sizeInfo);

        // Allocate the scratch buffer and setting the scratch info
        auto scratchBuffer = std::make_unique<VulkanBuffer>(
            getContext(),
            getRuntimeUniqueGPUASName("scratch_update"),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            0,
            sizeInfo.buildScratchSize
        );

        buildInfos.scratchData.deviceAddress = scratchBuffer->getDeviceAddress();

        std::vector<const VkAccelerationStructureBuildRangeInfoKHR*> pBuildOffset(blas.asBuildOffsetInfo.size());
        for (size_t i = 0; i < blas.asBuildOffsetInfo.size(); i++)
            pBuildOffset[i] = &blas.asBuildOffsetInfo[i];

        getContext()->executeImmediatelyMajorGraphics([&](VkCommandBuffer cmd) 
        {
            // Update the acceleration structure. Note the VK_TRUE parameter to trigger the update,
            // and the existing BLAS being passed and updated in place
            cmdBuildAccelerationStructures(cmd, 1, &buildInfos, pBuildOffset.data());
        });
	}


    void BLASBuilder::cmdCreateBlas(
        VkCommandBuffer cmdBuf, 
        std::vector<uint32_t> indices, 
        std::vector<BuildAccelerationStructure>& buildAs, 
        VkDeviceAddress scratchAddress, 
        VkQueryPool queryPool)
    {
        if (queryPool)
        {
            vkResetQueryPool(getContext()->getDevice(), queryPool, 0, static_cast<uint32_t>(indices.size()));
        }

        uint32_t queryCnt{ 0 };

        for (const auto& idx : indices)
        {
            // Actual allocation of buffer and acceleration structure.
            VkAccelerationStructureCreateInfoKHR createInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
            createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            createInfo.size = buildAs[idx].sizeInfo.accelerationStructureSize;  // Will be used to allocate memory.

            buildAs[idx].as.create(createInfo);

            // BuildInfo #2 part
            buildAs[idx].buildInfo.dstAccelerationStructure = buildAs[idx].as.accel;  // Setting where the build lands
            buildAs[idx].buildInfo.scratchData.deviceAddress = scratchAddress;  // All build are using the same scratch buffer

            // Building the bottom-level-acceleration-structure
            cmdBuildAccelerationStructures(cmdBuf, 1, &buildAs[idx].buildInfo, &buildAs[idx].rangeInfo);

            // Since the scratch buffer is reused across builds, we need a barrier to ensure one build
            // is finished before starting the next one.
            VkMemoryBarrier barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
            barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
            barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
            vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &barrier, 0, nullptr, 0, nullptr);

            if (queryPool)
            {
                // Add a query to find the 'real' amount of memory needed, use for compaction
                cmdWriteAccelerationStructuresProperties(cmdBuf, 1, &buildAs[idx].buildInfo.dstAccelerationStructure,
                    VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR, queryPool, queryCnt++);
            }
        }
    }

    void BLASBuilder::cmdCompactBlas(VkCommandBuffer cmdBuf, std::vector<uint32_t> indices, std::vector<BuildAccelerationStructure>& buildAs, VkQueryPool queryPool)
    {
        uint32_t queryCtn{ 0 };

        // Get the compacted size result back
        std::vector<VkDeviceSize> compactSizes(static_cast<uint32_t>(indices.size()));
        vkGetQueryPoolResults(getContext()->getDevice(), queryPool, 0, (uint32_t)compactSizes.size(), compactSizes.size() * sizeof(VkDeviceSize),
            compactSizes.data(), sizeof(VkDeviceSize), VK_QUERY_RESULT_WAIT_BIT);

        for (auto idx : indices)
        {
            buildAs[idx].cleanupAS = buildAs[idx].as;           // previous AS to destroy
            buildAs[idx].sizeInfo.accelerationStructureSize = compactSizes[queryCtn++];  // new reduced size

            // Creating a compact version of the AS
            VkAccelerationStructureCreateInfoKHR asCreateInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
            asCreateInfo.size = buildAs[idx].sizeInfo.accelerationStructureSize;
            asCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            buildAs[idx].as.create(asCreateInfo);

            // Copy the original BLAS to a compact version
            VkCopyAccelerationStructureInfoKHR copyInfo{ VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR };
            copyInfo.src = buildAs[idx].buildInfo.dstAccelerationStructure;
            copyInfo.dst = buildAs[idx].as.accel;
            copyInfo.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;
            cmdCopyAccelerationStructure(cmdBuf, &copyInfo);
        }
    }

    void BLASBuilder::destroyNonCompacted(std::vector<uint32_t> indices, std::vector<BuildAccelerationStructure>& buildAs)
    {
        for (auto& i : indices)
        {
            buildAs[i].cleanupAS.release();
        }
    }

}