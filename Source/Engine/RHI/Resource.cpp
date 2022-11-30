#include "Pch.h"
#include "Resource.h"
#include "RHI.h"

#include <vma/vk_mem_alloc.h>

#pragma warning (disable: 4297)

namespace Flower
{
	static AutoCVarInt32 cVarEnableVma(
		"r.RHI.EnableVma",
		"Enable vma allocator to manage vkBuffer create and destroy. 0 is off, others are on.",
		"RHI",
		1, // when vram > 256 MB, may allocate fail on some machine, so we use heap memory here.
		CVarFlags::ReadOnly | CVarFlags::InitOnce
	);

	// If size <= 128 MB, we use VMA, else use heap memory.
	constexpr VkDeviceSize GMaxVMASize = 128 * 1024 * 1024;

	constexpr bool canUseVMA(VkDeviceSize size)
	{
		return size <= GMaxVMASize;
	}

	bool VulkanBuffer::innerCreate(
		VkBufferUsageFlags usageFlags, 
		VkMemoryPropertyFlags memoryPropertyFlags, 
		VmaAllocationCreateFlags vmaUsage,
		void* data)
	{
		CHECK(m_size > 0 && "you should set size of buffer before create.");

		VkBufferCreateInfo bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = m_size;
		bufferInfo.usage = usageFlags;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (!isHeap())
		{
			VmaAllocationCreateInfo vmaallocInfo = {};
			vmaallocInfo.usage = VMA_MEMORY_USAGE_AUTO;
			vmaallocInfo.flags = vmaUsage;


			RHICheck(vmaCreateBuffer(RHI::VMA, &bufferInfo, &vmaallocInfo,
				&m_buffer,
				&m_allocation,
				nullptr));

			if (data != nullptr)
			{
				void* mapped;
				vmaMapMemory(RHI::VMA, m_allocation, &mapped);
				memcpy(mapped, data, m_size);
				vmaUnmapMemory(RHI::VMA, m_allocation);
			}
		}
		else
		{
			if (vkCreateBuffer(RHI::Device, &bufferInfo, nullptr, &m_buffer) != VK_SUCCESS)
			{
				LOG_RHI_FATAL("Fail to create vulkan buffer.");
			}

			VkMemoryRequirements memRequirements;
			vkGetBufferMemoryRequirements(RHI::Device, m_buffer, &memRequirements);

			VkMemoryAllocateInfo allocInfo{};
			allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			allocInfo.allocationSize = memRequirements.size;
			allocInfo.memoryTypeIndex = RHI::get()->findMemoryType(memRequirements.memoryTypeBits, memoryPropertyFlags);

			VkMemoryAllocateFlagsInfo memoryAllocateFlagsInfo = {};
			if(RHI::bSupportRayTrace)
			{
				memoryAllocateFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
				memoryAllocateFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;
				allocInfo.pNext = &memoryAllocateFlagsInfo;
			}
			

			if (vkAllocateMemory(RHI::Device, &allocInfo, nullptr, &m_memory) != VK_SUCCESS)
			{
				LOG_RHI_FATAL("Fail to allocate vulkan buffer.");
			}

			if (data != nullptr)
			{
				void* mapped;
				vkMapMemory(RHI::Device, m_memory, 0, m_size, 0, &mapped);
				memcpy(mapped, data, m_size);
				vkUnmapMemory(RHI::Device, m_memory);
			}

			vkBindBufferMemory(RHI::Device, m_buffer, m_memory, 0);
		}
		RHI::setResourceName(VK_OBJECT_TYPE_BUFFER, (uint64_t)m_buffer, m_name.c_str());
		RHI::addGpuResourceMemoryUsed(m_size);

		return true;
	}

	VulkanBuffer::~VulkanBuffer()
	{
		if (!isHeap())
		{
			vmaDestroyBuffer(RHI::VMA, m_buffer, m_allocation);
		}
		else
		{
			if (m_buffer != VK_NULL_HANDLE)
			{
				vkDestroyBuffer(RHI::Device, m_buffer, nullptr);
			}
			if (m_memory != VK_NULL_HANDLE)
			{
				vkFreeMemory(RHI::Device, m_memory, nullptr);
			}
		}

		RHI::minusGpuResourceMemoryUsed(m_size);
	}

	uint64_t VulkanBuffer::getDeviceAddress()
	{
		if (m_deviceAddress == 0)
		{
			VkBufferDeviceAddressInfo bufferDeviceAddressInfo{};
			bufferDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
			bufferDeviceAddressInfo.buffer = m_buffer;

			m_deviceAddress = vkGetBufferDeviceAddress(RHI::Device, &bufferDeviceAddressInfo);
		}

		return m_deviceAddress;
	}

	VkResult VulkanBuffer::map(VkDeviceSize size, VkDeviceSize offset)
	{
		VkResult res;

		lockMap();

		if (!isHeap())
		{
			res = vmaMapMemory(RHI::VMA, m_allocation, &mapped);
		}
		else
		{
			res = vkMapMemory(RHI::Device, m_memory, offset, size, 0, &mapped);
		}
		CHECK(mapped != nullptr && "Map fail.");
		return res;
	}

	void VulkanBuffer::copyTo(const void* data, VkDeviceSize size)
	{
		CHECK(mapped && "you must map buffer first before copy.");
		memcpy(mapped, data, size);
	}

	void VulkanBuffer::unmap()
	{
		CHECK(mapped != nullptr && "you should call unmap only once after call map.");

		if (!isHeap())
		{
			vmaUnmapMemory(RHI::VMA, m_allocation);
			mapped = nullptr;
		}
		else
		{
			vkUnmapMemory(RHI::Device, m_memory);
			mapped = nullptr;
		}

		unlockMap();
	}

	VkResult VulkanBuffer::bind(VkDeviceSize offset)
	{
		if (!isHeap())
		{
			return vmaBindBufferMemory2(RHI::VMA, m_allocation, offset, m_buffer, nullptr);
		}
		else
		{
			return vkBindBufferMemory(RHI::Device, m_buffer, m_memory, offset);
		}
	}

	VkResult VulkanBuffer::flush(VkDeviceSize size, VkDeviceSize offset)
	{
		VkResult res;
		if (!isHeap())
		{
			res = vmaFlushAllocation(RHI::VMA, m_allocation, offset, size);
		}
		else
		{
			VkMappedMemoryRange mappedRange = {};
			mappedRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
			mappedRange.memory = m_memory;
			mappedRange.offset = offset;
			mappedRange.size = size;
			res = vkFlushMappedMemoryRanges(RHI::Device, 1, &mappedRange);
		}

		return res;
	}

	VkResult VulkanBuffer::invalidate(VkDeviceSize size, VkDeviceSize offset)
	{
		VkResult res;
		if (!isHeap())
		{
			res = vmaInvalidateAllocation(RHI::VMA, m_allocation, offset, size);
		}
		else
		{
			VkMappedMemoryRange mappedRange = {};
			mappedRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
			mappedRange.memory = m_memory;
			mappedRange.offset = offset;
			mappedRange.size = size;
			res = vkInvalidateMappedMemoryRanges(RHI::Device, 1, &mappedRange);
		}

		return res;
	}

	void VulkanBuffer::setName(const char* newName)
	{
		if (m_name != newName)
		{
			m_name = newName;
			RHI::setResourceName(VK_OBJECT_TYPE_BUFFER, (uint64_t)m_buffer, newName);
		}
	}

	std::shared_ptr<VulkanBuffer> VulkanBuffer::create(
		const char* name,
		VkBufferUsageFlags usageFlags, 
		VkMemoryPropertyFlags memoryPropertyFlags, 
		EVMAUsageFlags vmaFlags,
		VkDeviceSize size, 
		void* data)
	{
		auto result = std::make_shared<VulkanBuffer>();

		result->m_bHeap = (cVarEnableVma.get() == 0) || !canUseVMA(size);
		result->m_size = size;
		result->m_name = name;

		VmaAllocationCreateFlags vmaUsage {};
		if (vmaFlags == EVMAUsageFlags::StageCopyForUpload)
		{
			vmaUsage =
				VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
				VMA_ALLOCATION_CREATE_MAPPED_BIT;
		}
		else if (vmaFlags == EVMAUsageFlags::Readback)
		{
			vmaUsage =
				VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
				VMA_ALLOCATION_CREATE_MAPPED_BIT;
		}

		result->innerCreate(usageFlags, memoryPropertyFlags, vmaUsage, data);

		return result;
	}

	std::shared_ptr<VulkanBuffer> VulkanBuffer::create2(
		const char* name,
		VkBufferUsageFlags usageFlags,
		VkMemoryPropertyFlags memoryPropertyFlags,
		VmaAllocationCreateFlags vmaUsage,
		VkDeviceSize size,
		void* data)
	{
		auto result = std::make_shared<VulkanBuffer>();

		result->m_bHeap = (cVarEnableVma.get() == 0) || !canUseVMA(size);
		result->m_size = size;
		result->m_name = name;
		result->innerCreate(usageFlags, memoryPropertyFlags, vmaUsage, data);

		return result;
	}

	std::shared_ptr<VulkanBuffer> VulkanBuffer::createRTScratchBuffer(const char* name, VkDeviceSize size)
	{
		return create(
			name, 
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, 
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 
			EVMAUsageFlags::GPUOnly, 
			size, 
			nullptr);
	}

	bool VulkanImage::innerCreate(VkMemoryPropertyFlags preperty)
	{
		RHICheck(vkCreateImage(RHI::Device, &m_createInfo, nullptr, &m_image));

		m_layouts.resize(m_createInfo.mipLevels);
		m_ownerQueueFamilys.resize(m_layouts.size());
		for (size_t i = 0; i < m_layouts.size(); i++)
		{
			m_layouts[i] = VK_IMAGE_LAYOUT_UNDEFINED;
			m_ownerQueueFamilys[i] = VK_QUEUE_FAMILY_IGNORED;
		}

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(RHI::Device, m_image, &memRequirements);

		m_size = memRequirements.size;
		m_bHeap = (cVarEnableVma.get() == 0) || !canUseVMA(m_size);

		vkDestroyImage(RHI::Device, m_image, nullptr);
		m_image = VK_NULL_HANDLE;

		if (!isHeap())
		{
			VmaAllocationCreateInfo imageAllocCreateInfo = {};
			imageAllocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
			imageAllocCreateInfo.flags = VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT;
			imageAllocCreateInfo.pUserData = (void*)m_name.c_str();
			VmaAllocationInfo gpuImageAllocInfo = {};

			RHICheck(vmaCreateImage(RHI::VMA, &m_createInfo, &imageAllocCreateInfo, &m_image, &m_allocation, &gpuImageAllocInfo));
		}
		else
		{
			RHICheck(vkCreateImage(RHI::Device, &m_createInfo, nullptr, &m_image));

			VkMemoryAllocateInfo allocInfo{};
			allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			allocInfo.allocationSize = m_size;

			VkMemoryPropertyFlags properties = preperty;

			allocInfo.memoryTypeIndex = RHI::get()->findMemoryType(memRequirements.memoryTypeBits, properties);
			RHICheck(vkAllocateMemory(RHI::Device, &allocInfo, nullptr, &m_memory));

			vkBindImageMemory(RHI::Device, m_image, m_memory, 0);
		}
		RHI::setResourceName(VK_OBJECT_TYPE_IMAGE, (uint64_t)m_image, m_name.c_str());
		RHI::addGpuResourceMemoryUsed(m_size);
		LOG_RHI_INFO("Image {0} has created.", m_name);

		return true;
	}

	VulkanImage::~VulkanImage()
	{
		CHECK(m_image != VK_NULL_HANDLE);

		if (m_allocation != nullptr)
		{
			vmaDestroyImage(RHI::VMA, m_image, m_allocation);
			m_image = VK_NULL_HANDLE;
		}
		else
		{
			vkDestroyImage(RHI::Device, m_image, nullptr);
			m_image = VK_NULL_HANDLE;

			CHECK(m_memory != VK_NULL_HANDLE);
			vkFreeMemory(RHI::Device, m_memory, nullptr);
			m_memory = VK_NULL_HANDLE;
		}

		for (auto& pair : m_cacheImageViews)
		{
			CHECK(pair.second != VK_NULL_HANDLE);
			vkDestroyImageView(RHI::Device, pair.second, nullptr);
		}
		m_cacheImageViews.clear();
		RHI::minusGpuResourceMemoryUsed(m_size);
		LOG_RHI_INFO("Image {0} has release.", m_name);
	}

	void VulkanImage::rename(const std::string& name)
	{
	// RT pool reuse will trigger rename frequently. close here.
#if 0
		if (m_name != name)
		{

			LOG_RHI_INFO("Rename resource {0} to {1}.", m_name, name);
			m_name = name;

			RHI::setResourceName(VK_OBJECT_TYPE_IMAGE, (uint64_t)m_image, m_name.c_str());

			for (auto& pair : m_cacheImageViews)
			{
				RHI::setResourceName(VK_OBJECT_TYPE_IMAGE_VIEW, (uint64_t)&pair.second, m_name.c_str());
			}

		}
#endif
	}

	std::shared_ptr<VulkanImage> VulkanImage::create(const char* name, const VkImageCreateInfo& createInfo, VkMemoryPropertyFlags preperty)
	{
		auto result = std::make_shared<VulkanImage>();

		result->m_name = name;
		result->m_createInfo = createInfo;
		result->innerCreate(preperty);

		return result;
	}

	// Try get view and create if no exist.
	VkImageView VulkanImage::getView(VkImageSubresourceRange range, VkImageViewType viewType)
	{
		VkImageViewCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		info.image = m_image;
		info.subresourceRange = range;

		info.format = m_createInfo.format;
		info.viewType = viewType;

		size_t hashVal = CRC::Calculate(&info, sizeof(VkImageViewCreateInfo), CRC::CRC_32());
		if (!m_cacheImageViews.contains(hashVal))
		{
			m_cacheImageViews[hashVal] = VK_NULL_HANDLE;

			RHICheck(vkCreateImageView(RHI::Device, &info, NULL, &m_cacheImageViews[hashVal]));
		}

		return m_cacheImageViews[hashVal];
	}

	void VulkanImage::transitionLayout(
		RHICommandBufferBase& cmd,
		VkImageLayout newLayout,
		VkImageSubresourceRange range)
	{
		transitionLayout(cmd.cmd, cmd.queueFamily, newLayout, range);
	}

	void VulkanImage::transitionLayout(
		VkCommandBuffer cmd,
		VkImageLayout newLayout,
		VkImageSubresourceRange range)
	{
		transitionLayout(cmd, RHI::get()->getGraphiscFamily(), newLayout, range);
	}

	void VulkanImage::transitionLayout(
		VkCommandBuffer cb, 
		uint32_t newQueueFamily,
		VkImageLayout newLayout, 
		VkImageSubresourceRange range)
	{
		std::vector<VkImageMemoryBarrier> barriers;

		VkDependencyFlags dependencyFlags{};
		VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
		VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

		uint32_t maxMip = glm::min(range.baseMipLevel + range.levelCount, m_createInfo.mipLevels);
		for (uint32_t i = range.baseMipLevel; i < maxMip; i++)
		{
			VkImageLayout oldLayout = m_layouts.at(i);
			uint32_t oldFamily = m_ownerQueueFamilys.at(i);

			if ((newLayout == oldLayout) && (oldFamily == newQueueFamily))
			{
				continue;
			}
				

			m_layouts[i] = newLayout;
			m_ownerQueueFamilys[i] = newQueueFamily;

			VkImageMemoryBarrier barrier{};
			barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			barrier.oldLayout = oldLayout;
			barrier.newLayout = newLayout;
			barrier.srcQueueFamilyIndex = (oldFamily == VK_QUEUE_FAMILY_IGNORED) ? newQueueFamily : oldFamily;
			barrier.dstQueueFamilyIndex = newQueueFamily;
			barrier.image = m_image;

			VkImageSubresourceRange rangSpecial {
				.aspectMask = range.aspectMask,
				.baseMipLevel = i,
				.levelCount = 1,
				.baseArrayLayer = range.baseArrayLayer, 
				.layerCount = range.layerCount,
			};

			if (range.layerCount < m_createInfo.arrayLayers)
			{
				// TODO: Maybe we also need to control per-level resource layout and ownership. wtf bullshit thing here.
				//       Which meaning we need to manage miplevelscount x layerlevelsCount layout and ownership.
				CHECK_ENTRY();
			}

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

	void VulkanImage::transitionLayoutImmediately(
		VkImageLayout newLayout, 
		VkImageSubresourceRange range)
	{
		RHI::executeImmediatelyMajorGraphics([&, this](VkCommandBuffer cb)
		{
			transitionLayout(cb, RHI::get()->getGraphiscFamily(), newLayout, range);
		});
	}
}