#include "rhi.h"
#include <util/cityhash/city.h>

namespace engine
{
	void GpuResource::init(const VulkanContext* context, const std::string& name, VkDeviceSize size)
	{
		m_runtimeUUID = buildRuntimeUUID64u();
		m_context = context;
		m_name = name;
		m_size = size;

		ASSERT(m_size > 0, "you should set size of buffer before create.");
	}

	VulkanBuffer::VulkanBuffer(
		const VulkanContext* context,
		const std::string& name,
		VkBufferUsageFlags usageFlags, 
		VmaAllocationCreateFlags vmaUsage, 
		VkDeviceSize size, 
		void* data)
	{
		GpuResource::init(context, name, size);

		VkBufferCreateInfo bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = m_size;
		bufferInfo.usage = usageFlags;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		VmaAllocationCreateInfo vmaallocInfo = {};
		vmaallocInfo.usage = VMA_MEMORY_USAGE_AUTO;
		vmaallocInfo.flags = vmaUsage;

		RHICheck(vmaCreateBuffer(m_context->getVMA(), &bufferInfo, &vmaallocInfo, &m_buffer, &m_allocation, nullptr));

		m_bSupportDeviceAddress = (usageFlags & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
		if (m_bSupportDeviceAddress)
		{
			// Get buffer address.
			VkBufferDeviceAddressInfo bufferDeviceAddressInfo{};
			bufferDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
			bufferDeviceAddressInfo.buffer = m_buffer;
			m_deviceAddress = vkGetBufferDeviceAddress(m_context->getDevice(), &bufferDeviceAddressInfo);
		}

		// Copy data if need upload.
		if (data != nullptr)
		{
			void* dataMapped = nullptr;
			vmaMapMemory(m_context->getVMA(), m_allocation, &dataMapped);
			memcpy(dataMapped, data, m_size);
			vmaUnmapMemory(m_context->getVMA(), m_allocation);
		}

		m_context->setResourceName(VK_OBJECT_TYPE_BUFFER, (uint64_t)m_buffer, m_name.c_str());

		m_defaultInfo.offset = 0;
		m_defaultInfo.range = m_size;
		m_defaultInfo.buffer = m_buffer;
	}

	void VulkanBuffer::rename(const std::string& newName)
	{
		m_name = newName;
		m_context->setResourceName(VK_OBJECT_TYPE_BUFFER, (uint64_t)m_buffer, newName.c_str());
	}

	void VulkanBuffer::map(VkDeviceSize size)
	{
		if (m_mapped == nullptr)
		{
			RHICheck(vmaMapMemory(m_context->getVMA(), m_allocation, &m_mapped));
		}
	}

	void VulkanBuffer::copyTo(const void* data, VkDeviceSize size)
	{
		if (m_mapped != nullptr)
		{
			LOG_ERROR("Buffer already mapped, don't use this function.");
		}

		map(size);
		memcpy(m_mapped, data, size);
		unmap();
	}

	void VulkanBuffer::unmap()
	{
		if (m_mapped != nullptr)
		{
			vmaUnmapMemory(m_context->getVMA(), m_allocation);
			m_mapped = nullptr;
		}
	}


	VulkanBuffer::~VulkanBuffer()
	{
		if (m_mapped)
		{
			unmap();
		}

		vmaDestroyBuffer(m_context->getVMA(), m_buffer, m_allocation);
	}


	void VulkanBuffer::bind(VkDeviceSize offset)
	{
		RHICheck(vmaBindBufferMemory2(m_context->getVMA(), m_allocation, offset, m_buffer, nullptr));
	}

	void VulkanBuffer::flush(VkDeviceSize size, VkDeviceSize offset)
	{
		RHICheck(vmaFlushAllocation(m_context->getVMA(), m_allocation, offset, size));
	}

	void VulkanBuffer::invalidate(VkDeviceSize size, VkDeviceSize offset)
	{
		RHICheck(vmaInvalidateAllocation(m_context->getVMA(), m_allocation, offset, size));
	}

	void VulkanBuffer::stageCopyFrom(VkBuffer inBuffer, VkDeviceSize size, VkDeviceSize srcOffset, VkDeviceSize destOffset)
	{
		m_context->executeImmediatelyMajorGraphics([&](VkCommandBuffer cb) {
			VkBufferCopy copyRegion{};

			copyRegion.srcOffset = srcOffset;
			copyRegion.dstOffset = destOffset;
			copyRegion.size = size;
			vkCmdCopyBuffer(cb, inBuffer, m_buffer, 1, &copyRegion);
		});
	}

	VulkanImage::VulkanImage(
		const VulkanContext* context,
		const std::string& name,
		const VkImageCreateInfo& createInfo,
		VkMemoryPropertyFlags preperty) : m_createInfo(createInfo)
	{
		// Create image to query memory size.
		RHICheck(vkCreateImage(context->getDevice(), &m_createInfo, nullptr, &m_image));
		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(context->getDevice(), m_image, &memRequirements);
		vkDestroyImage(context->getDevice(), m_image, nullptr);
		m_image = VK_NULL_HANDLE;

		// Parent init.
		GpuResource::init(context, name, memRequirements.size);

		// Init some subresources.
		size_t subresourceNum = m_createInfo.arrayLayers * m_createInfo.mipLevels;
		m_layouts.resize(subresourceNum);
		m_ownerQueueFamilyIndices.resize(subresourceNum);
		for (size_t i = 0; i < m_layouts.size(); i++)
		{
			m_layouts[i] = VK_IMAGE_LAYOUT_UNDEFINED;
			m_ownerQueueFamilyIndices[i] = VK_QUEUE_FAMILY_IGNORED;
		}

		VmaAllocationCreateInfo imageAllocCreateInfo = {};
		imageAllocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
		imageAllocCreateInfo.flags = VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT;
		imageAllocCreateInfo.pUserData = (void*)m_name.c_str();
		VmaAllocationInfo gpuImageAllocInfo = {};

		RHICheck(vmaCreateImage(m_context->getVMA(), &m_createInfo, &imageAllocCreateInfo, &m_image, &m_allocation, &gpuImageAllocInfo));

		m_context->setResourceName(VK_OBJECT_TYPE_IMAGE, (uint64_t)m_image, m_name.c_str());
	}

	VulkanImage::~VulkanImage()
	{
		if (m_image != VK_NULL_HANDLE)
		{
			vmaDestroyImage(m_context->getVMA(), m_image, m_allocation);
			m_image = VK_NULL_HANDLE;
		}

		for (auto& pair : m_cacheImageViews)
		{
			vkDestroyImageView(m_context->getDevice(), pair.second, nullptr);
		}
		m_cacheImageViews.clear();
	}

	size_t VulkanImage::getSubresourceIndex(uint32_t layerIndex, uint32_t mipLevel) const
	{
		CHECK((layerIndex < m_createInfo.arrayLayers) && (mipLevel < m_createInfo.mipLevels));
		return layerIndex * m_createInfo.mipLevels + mipLevel;
	}

	VkImageLayout VulkanImage::getCurrentLayout(uint32_t layerIndex, uint32_t mipLevel) const
	{
		size_t subresourceIndex = getSubresourceIndex(layerIndex, mipLevel);
		return m_layouts.at(subresourceIndex);
	}

	VkImageView VulkanImage::getOrCreateView(VkImageSubresourceRange range, VkImageViewType viewType)
	{
		VkImageViewCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		info.image = m_image;
		info.subresourceRange = range;
		info.format = m_createInfo.format;
		info.viewType = viewType;
		uint64_t hashVal = CityHash64((const char*)&info, sizeof(VkImageViewCreateInfo));
		if (!m_cacheImageViews.contains(hashVal))
		{
			m_cacheImageViews[hashVal] = VK_NULL_HANDLE;

			RHICheck(vkCreateImageView(m_context->getDevice(), &info, NULL, &m_cacheImageViews[hashVal]));
		}

		return m_cacheImageViews[hashVal];
	}

	void VulkanImage::transitionLayout(RHICommandBufferBase& cmd, VkImageLayout newLayout, VkImageSubresourceRange range)
	{
		transitionLayout(cmd.cmd, cmd.queueFamily, newLayout, range);
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
				size_t flatId = getSubresourceIndex(layerIndex, mipIndex);

				VkImageLayout oldLayout = m_layouts.at(flatId);
				uint32_t oldFamily = m_ownerQueueFamilyIndices.at(flatId);

				if ((newLayout == oldLayout) && (oldFamily == newQueueFamily))
				{
					continue;
				}


				m_layouts[flatId] = newLayout;
				m_ownerQueueFamilyIndices[flatId] = newQueueFamily;

				VkImageMemoryBarrier barrier{};
				barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
				barrier.oldLayout = oldLayout;
				barrier.newLayout = newLayout;
				barrier.srcQueueFamilyIndex = (oldFamily == VK_QUEUE_FAMILY_IGNORED) ? newQueueFamily : oldFamily;
				barrier.dstQueueFamilyIndex = newQueueFamily;
				barrier.image = m_image;

				VkImageSubresourceRange rangSpecial{
					.aspectMask = range.aspectMask,
					.baseMipLevel = mipIndex,
					.levelCount = 1,
					.baseArrayLayer = layerIndex,
					.layerCount = 1,
				};

				barrier.subresourceRange = rangSpecial;

				VkAccessFlags srcMask{};
				VkAccessFlags dstMask{};

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
					dstMask = dstMask | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
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

	void VulkanImage::transitionLayout(VkCommandBuffer cmd, VkImageLayout newLayout, VkImageSubresourceRange range)
	{
		transitionLayout(cmd, m_context->getGraphiscFamily(), newLayout, range);
	}

	void VulkanImage::transitionLayoutImmediately(VkImageLayout newLayout, VkImageSubresourceRange range)
	{
		m_context->executeImmediatelyMajorGraphics([&, this](VkCommandBuffer cb)
		{
			transitionLayout(cb, m_context->getGraphiscFamily(), newLayout, range);
		});
	}
}