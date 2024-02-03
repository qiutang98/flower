#include "resource.h"
#include "context.h"
#include "log.h"
#include "engine.h"

namespace engine
{
	static AutoCVarBool cVarRHIDebugMarkerEnable(
		"r.RHI.LogGpuResourceCreateAndDestroyInfo",
		"Enable debug gpu resource create and destroy infos.",
		"RHI",
		false,
		CVarFlags::ReadOnly
	);

	static VkDeviceSize sTotalGpuDeviceSize = 0;
	VkDeviceSize getAllocateGpuResourceSize()
	{
		return sTotalGpuDeviceSize;
	}

	GpuResource::GpuResource(const std::string name, VkDeviceSize size)
		: m_name(name), m_size(size), m_runtimeUUID(buildRuntimeUUID64u())
	{
		
	}

	GpuResource::~GpuResource()
	{

	}

	VulkanBuffer::VulkanBuffer(
		VmaAllocator vma,
		const std::string& name,
		VkBufferUsageFlags usageFlags,
		VmaAllocationCreateFlags vmaUsage,
		VkDeviceSize size,
		void* data) 
	: GpuResource(name, size), m_vma(vma)
	{
		VkBufferCreateInfo bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = m_size;
		bufferInfo.usage = usageFlags;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		VmaAllocationCreateInfo vmaallocInfo = {};
		vmaallocInfo.usage = VMA_MEMORY_USAGE_AUTO;
		vmaallocInfo.flags = vmaUsage;

		RHICheck(vmaCreateBuffer(vma, &bufferInfo, &vmaallocInfo, &m_buffer, &m_allocation, nullptr));
		postInit(usageFlags, data);
	}

	VulkanBuffer::VulkanBuffer(
		VkBufferUsageFlags usageFlags, 
		VkMemoryPropertyFlags memoryPropertyFlags, 
		const std::string& name, 
		VkDeviceSize size, 
		void* data,
		bool bBindAfterInit)
		: GpuResource(name, size)
	{
		VkBufferCreateInfo bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = m_size;
		bufferInfo.usage = usageFlags;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		RHICheck(vkCreateBuffer(getDevice(), &bufferInfo, nullptr, &m_buffer));

		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(getDevice(), m_buffer, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = getContext()->findMemoryType(memRequirements.memoryTypeBits, memoryPropertyFlags);

		VkMemoryAllocateFlagsInfo memoryAllocateFlagsInfo = {};
		if (usageFlags & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)
		{
			memoryAllocateFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
			memoryAllocateFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;
			allocInfo.pNext = &memoryAllocateFlagsInfo;
		}

		RHICheck(vkAllocateMemory(getDevice(), &allocInfo, nullptr, &m_memory));

		if (bBindAfterInit)
		{
			RHICheck(vkBindBufferMemory(getDevice(), m_buffer, m_memory, 0));
		}

		postInit(usageFlags, data);
	}

	void VulkanBuffer::postInit(VkBufferUsageFlags usageFlags, void* data)
	{
		m_bSupportDeviceAddress = (usageFlags & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
		if (m_bSupportDeviceAddress)
		{
			// Get buffer address.
			VkBufferDeviceAddressInfo bufferDeviceAddressInfo{};
			bufferDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
			bufferDeviceAddressInfo.buffer = m_buffer;
			m_deviceAddress = vkGetBufferDeviceAddress(getContext()->getDevice(), &bufferDeviceAddressInfo);
		}

		// Copy data if need upload.
		if (data != nullptr)
		{
			copyTo(data, m_size);
		}

		getContext()->setResourceName(VK_OBJECT_TYPE_BUFFER, (uint64_t)m_buffer, m_name.c_str());

		m_defaultInfo = {};
		{
			// Build default buffer descriptor infos.
			m_defaultInfo.offset = 0;
			m_defaultInfo.range = m_size;
			m_defaultInfo.buffer = m_buffer;
		}

		if (cVarRHIDebugMarkerEnable.get())
		{
			LOG_RHI_TRACE("Create gpu buffer {0} with size {1} KB.", m_name, float(m_size) / 1024.0f);
		}
		sTotalGpuDeviceSize += m_size;
	}


	void VulkanBuffer::rename(const std::string& newName)
	{
		m_name = newName;
		getContext()->setResourceName(VK_OBJECT_TYPE_BUFFER, (uint64_t)m_buffer, newName.c_str());
	}

	uint64_t VulkanBuffer::getDeviceAddress() const
	{
		CHECK(m_bSupportDeviceAddress);
		return m_deviceAddress;
	}


	void VulkanBuffer::map(VkDeviceSize size)
	{
		if (m_mapped == nullptr)
		{
			if (m_memory != VK_NULL_HANDLE)
			{
				RHICheck(vkMapMemory(getDevice(), m_memory, 0, size, 0, &m_mapped));
			}
			else
			{
				RHICheck(vmaMapMemory(m_vma, m_allocation, &m_mapped));
			}
		}
	}

	void VulkanBuffer::copyTo(const void* data, VkDeviceSize size)
	{
		if (m_mapped != nullptr)
		{
			LOG_RHI_FATAL("Buffer already mapped, don't use this function.");
		}

		map(size);
		memcpy(m_mapped, data, size);
		unmap();
	}

	void VulkanBuffer::copyAndUpload(VkCommandBuffer cmd, const void* data, VulkanBuffer* dstBuffer)
	{
		auto* stageBuffer = this;
		CHECK(dstBuffer->getSize() <= stageBuffer->getSize());

		// cpu copy buffer to map stage buffer. 
		copyTo(data, dstBuffer->getSize());

		// cpu copy to gpu
		VkBufferCopy copyRegion{};
		copyRegion.srcOffset = 0;
		copyRegion.dstOffset = 0;
		copyRegion.size = dstBuffer->getSize();
		vkCmdCopyBuffer(cmd, stageBuffer->getVkBuffer(), dstBuffer->getVkBuffer(), 1, &copyRegion);
	}

	void VulkanBuffer::unmap()
	{
		if (m_mapped != nullptr)
		{
			if (m_memory != VK_NULL_HANDLE)
			{
				vkUnmapMemory(getDevice(), m_memory);
			}
			else
			{
				vmaUnmapMemory(m_vma, m_allocation);
			}

			m_mapped = nullptr;
		}
	}

	VulkanBuffer::~VulkanBuffer()
	{
		sTotalGpuDeviceSize -= m_size;
		if (cVarRHIDebugMarkerEnable.get())
		{
			LOG_RHI_TRACE("Destroy gpu buffer {0} with size {1} KB.", m_name, float(m_size) / 1024.0f);
		}

		if (m_mapped)
		{
			unmap();
		}

		if (m_memory != VK_NULL_HANDLE)
		{
			CHECK(m_allocation == VK_NULL_HANDLE);

			vkDestroyBuffer(getDevice(), m_buffer, nullptr);
			vkFreeMemory(getDevice(), m_memory, nullptr);
		}
		else
		{
			vmaDestroyBuffer(m_vma, m_buffer, m_allocation);
		}
	}


	void VulkanBuffer::bind(VkDeviceSize offset)
	{
		if (m_memory != VK_NULL_HANDLE)
		{
			RHICheck(vkBindBufferMemory(getDevice(), m_buffer, m_memory, offset));
		}
		else
		{
			RHICheck(vmaBindBufferMemory2(m_vma, m_allocation, offset, m_buffer, nullptr));
		}
	}

	void VulkanBuffer::flush(VkDeviceSize size, VkDeviceSize offset)
	{
		if (m_memory != VK_NULL_HANDLE)
		{
			VkMappedMemoryRange mappedRange = {};
			mappedRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
			mappedRange.memory = m_memory;
			mappedRange.offset = offset;
			mappedRange.size = size;
			RHICheck(vkFlushMappedMemoryRanges(getDevice(), 1, &mappedRange));
		}
		else
		{
			RHICheck(vmaFlushAllocation(m_vma, m_allocation, offset, size));
		}
	}

	void VulkanBuffer::invalidate(VkDeviceSize size, VkDeviceSize offset)
	{
		if (m_memory != VK_NULL_HANDLE)
		{
			VkMappedMemoryRange mappedRange = {};
			mappedRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
			mappedRange.memory = m_memory;
			mappedRange.offset = offset;
			mappedRange.size = size;
			RHICheck(vkInvalidateMappedMemoryRanges(getDevice(), 1, &mappedRange));
		}
		else
		{
			RHICheck(vmaInvalidateAllocation(m_vma, m_allocation, offset, size));
		}
	}

	void VulkanBuffer::stageCopyFrom(VkBuffer inBuffer, VkDeviceSize size, VkDeviceSize srcOffset, VkDeviceSize destOffset)
	{
		getContext()->executeImmediatelyMajorGraphics([&](VkCommandBuffer cb) 
		{
			VkBufferCopy copyRegion{};

			copyRegion.srcOffset = srcOffset;
			copyRegion.dstOffset = destOffset;
			copyRegion.size = size;
			vkCmdCopyBuffer(cb, inBuffer, m_buffer, 1, &copyRegion);
		});
	}

	VulkanImage::VulkanImage(VmaAllocator vma, const std::string& name, VkImageCreateInfo createInfo)
		: GpuResource(name, 0), m_createInfo(createInfo), m_vma(vma)
	{
		// Create image to query memory size.
		VkMemoryRequirements memRequirements;
		{
			RHICheck(vkCreateImage(getContext()->getDevice(), &m_createInfo, nullptr, &m_image));
			vkGetImageMemoryRequirements(getContext()->getDevice(), m_image, &memRequirements);
			vkDestroyImage(getContext()->getDevice(), m_image, nullptr);
			m_image = VK_NULL_HANDLE;

			// Get final image memory size.
			this->m_size = memRequirements.size;
		}

		// Init some subresources.
		size_t subresourceNum = m_createInfo.arrayLayers * m_createInfo.mipLevels;
		m_subresourceStates.resize(subresourceNum);
		for (size_t i = 0; i < subresourceNum; i++)
		{
			m_subresourceStates[i].imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			m_subresourceStates[i].ownerQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		}

		// Finally we create image.
		{
			VmaAllocationCreateInfo imageAllocCreateInfo = {};
			imageAllocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
			imageAllocCreateInfo.flags = VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT;
			imageAllocCreateInfo.pUserData = (void*)m_name.c_str();
			VmaAllocationInfo gpuImageAllocInfo = {};

			RHICheck(vmaCreateImage(m_vma, &m_createInfo, &imageAllocCreateInfo, &m_image, &m_allocation, &gpuImageAllocInfo));
		}

		getContext()->setResourceName(VK_OBJECT_TYPE_IMAGE, (uint64_t)m_image, m_name.c_str());

		if (cVarRHIDebugMarkerEnable.get())
		{
			LOG_RHI_TRACE("Create gpu image {0} with size {1} KB.", m_name, float(m_size) / 1024.0f);
		}
		sTotalGpuDeviceSize += m_size;

	}

	VulkanImage::~VulkanImage()
	{
		const bool bReleasing = (Engine::get()->getModuleState() == Engine::EModuleState::Releasing);

		sTotalGpuDeviceSize -= m_size;
		if (cVarRHIDebugMarkerEnable.get())
		{
			LOG_RHI_TRACE("Destroy gpu image {0} with size {1} KB.", m_name, float(m_size) / 1024.0f);
		}

		for (auto& pair : m_cacheImageViews)
		{
			// Free bindless srv.
			getContext()->getBindlessTexture().freeBindlessImpl(
				pair.second.srvBindless,
				bReleasing ? nullptr : getContext()->getBuiltinTextureWhite()->getReadyImage());

			vkDestroyImageView(getContext()->getDevice(), pair.second.view, nullptr);
		}
		m_cacheImageViews.clear();

		if (m_image != VK_NULL_HANDLE)
		{
			vmaDestroyImage(m_vma, m_image, m_allocation);
			m_image = VK_NULL_HANDLE;
		}
	}

	uint32_t VulkanImage::getSubresourceIndex(uint32_t layerIndex, uint32_t mipLevel) const
	{
		CHECK((layerIndex < m_createInfo.arrayLayers) && (mipLevel < m_createInfo.mipLevels));
		return layerIndex * m_createInfo.mipLevels + mipLevel;
	}

	VkImageLayout VulkanImage::getCurrentLayout(uint32_t layerIndex, uint32_t mipLevel) const
	{
		uint32_t subresourceIndex = getSubresourceIndex(layerIndex, mipLevel);
		return m_subresourceStates.at(subresourceIndex).imageLayout;
	}

	ViewAndBindlessIndex VulkanImage::getOrCreateView(VkImageSubresourceRange range, VkImageViewType viewType)
	{
		VkImageViewCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		info.image = m_image;
		info.subresourceRange = range;
		info.format = m_createInfo.format;
		info.viewType = viewType;

		const uint32_t hashVal = crc::crc32((const char*)&info, sizeof(VkImageViewCreateInfo));
		if (!m_cacheImageViews.contains(hashVal))
		{
			m_cacheImageViews[hashVal].view = VK_NULL_HANDLE;
			RHICheck(vkCreateImageView(getContext()->getDevice(), &info, NULL, &m_cacheImageViews[hashVal].view));

			// Register bindless srv.
			m_cacheImageViews[hashVal].srvBindless = 
				getContext()->getBindlessTexture().updateTextureToBindlessDescriptorSet(m_cacheImageViews[hashVal].view);
		}

		return m_cacheImageViews[hashVal];
	}

	void VulkanImage::transitionLayout(RHICommandBufferBase& cmd, VkImageLayout newLayout, VkImageSubresourceRange range)
	{
		transitionLayout(cmd.cmd, cmd.queueFamily, newLayout, range);
	}

	void VulkanImage::transitionLayout(VkCommandBuffer cmd, VkImageLayout newLayout, VkImageSubresourceRange range)
	{
		transitionLayout(cmd, getContext()->getGraphiscFamily(), newLayout, range);
	}

	inline static void getVkAccessFlagsByLayout(VkImageLayout oldLayout, VkImageLayout newLayout, VkAccessFlags& srcMask, VkAccessFlags& dstMask)
	{
		switch (oldLayout)
		{
		case VK_IMAGE_LAYOUT_UNDEFINED:
			srcMask = 0;
			break;
		case VK_IMAGE_LAYOUT_PREINITIALIZED:
			srcMask = VK_ACCESS_HOST_WRITE_BIT;
			break;
		case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL: 
			srcMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; 
			break;
		case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL: 
			srcMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT; 
			break;
		case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
			srcMask = VK_ACCESS_TRANSFER_READ_BIT;
			break;
		case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
			srcMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			break;
		case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
			srcMask = VK_ACCESS_SHADER_READ_BIT;
			break;
		case VK_IMAGE_LAYOUT_GENERAL:
			srcMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
			break;
		default:
			LOG_RHI_FATAL("Image layout transition no support.");
			srcMask = ~0;
			break;
		}

		switch (newLayout)
		{
		case VK_IMAGE_LAYOUT_GENERAL:
			dstMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
			break;
		case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
			dstMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			break;
		case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
			dstMask = VK_ACCESS_TRANSFER_READ_BIT;
			break;
		case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
			dstMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
			break;
		case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
			dstMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
			break;
		case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
			if (srcMask == 0)
			{
				srcMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
			}
			dstMask = VK_ACCESS_SHADER_READ_BIT;
			break;
		default:
			LOG_RHI_FATAL("Image layout transition no support.");
			dstMask = ~0;
			break;
		}
	}

	void VulkanImage::transitionLayout(VkCommandBuffer cb, uint32_t newQueueFamily, VkImageLayout newLayout, VkImageSubresourceRange range)
	{
		std::vector<VkImageMemoryBarrier> barriers;

		VkDependencyFlags dependencyFlags{};
		VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
		VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

		uint32_t maxLayer = glm::min(range.baseArrayLayer + range.layerCount, m_createInfo.arrayLayers);
		for (uint32_t layerIndex = range.baseArrayLayer; layerIndex < maxLayer; layerIndex++)
		{
			uint32_t maxMip = glm::min(range.baseMipLevel + range.levelCount, m_createInfo.mipLevels);
			for (uint32_t mipIndex = range.baseMipLevel; mipIndex < maxMip; mipIndex++)
			{
				uint32_t flatId = getSubresourceIndex(layerIndex, mipIndex);

				auto& subresourceState = m_subresourceStates.at(flatId);

				// Get old state.
				VkImageLayout oldLayout = subresourceState.imageLayout;
				uint32_t oldFamily = subresourceState.ownerQueueFamilyIndex;

				// State no change.
				if ((newLayout == oldLayout) && (oldFamily == newQueueFamily))
				{
					continue;
				}

				// Update subresource state.
				subresourceState.imageLayout = newLayout;
				subresourceState.ownerQueueFamilyIndex = newQueueFamily;

				VkImageMemoryBarrier barrier{};
				barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
				barrier.oldLayout = oldLayout;
				barrier.newLayout = newLayout;
				barrier.srcQueueFamilyIndex = (oldFamily == VK_QUEUE_FAMILY_IGNORED) ? newQueueFamily : oldFamily;
				barrier.dstQueueFamilyIndex = newQueueFamily;
				barrier.image = m_image;

				// Build subresourceRange
				VkImageSubresourceRange rangSpecial
				{
					.aspectMask = range.aspectMask,
					.baseMipLevel = mipIndex,
					.levelCount = 1,
					.baseArrayLayer = layerIndex,
					.layerCount = 1,
				};
				barrier.subresourceRange = rangSpecial;

				// Build VkAccessFlags
				VkAccessFlags srcMask{};
				VkAccessFlags dstMask{};
				getVkAccessFlagsByLayout(oldLayout, newLayout, srcMask, dstMask);

				barrier.srcAccessMask = srcMask;
				barrier.dstAccessMask = dstMask;
				barriers.push_back(barrier);
			}
		}

		if (barriers.empty())
		{
			return;
		}

		vkCmdPipelineBarrier(
			cb,
			srcStageMask,
			dstStageMask,
			dependencyFlags,
			0,
			nullptr,
			0,
			nullptr,
			(uint32_t)barriers.size(),
			barriers.data()
		);
	}

	void VulkanImage::transitionLayoutImmediately(VkImageLayout newLayout, VkImageSubresourceRange range)
	{
		getContext()->executeImmediatelyMajorGraphics([&, this](VkCommandBuffer cb)
		{
			transitionLayout(cb, getContext()->getGraphiscFamily(), newLayout, range);
		});
	}
}