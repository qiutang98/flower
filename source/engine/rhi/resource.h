#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <util/noncopyable.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

#include "rhi_misc.h"

namespace engine
{

	class VulkanContext;

	// Global interface for vulkan gpu resource. 
	class GpuResource : NonCopyable
	{
	protected:
		const VulkanContext* m_context;

		// Resource name.
		std::string m_name;

		// Resource size.
		VkDeviceSize m_size;

		// Resource runtime uuid.
		UUID64u m_runtimeUUID;

		void init(const VulkanContext* context, const std::string& name, VkDeviceSize size);

	public:
		GpuResource() = default;
		virtual ~GpuResource() = default;

		// Getter.
		VkDeviceSize getSize() const { return m_size; }
		const std::string& getName() const { return m_name; }

		const UUID64u& getRuntimeUUID() const { return m_runtimeUUID; }
	};

	class VulkanBuffer : public GpuResource
	{
	public:
		explicit VulkanBuffer(
			const VulkanContext* context,
			const std::string& name,
			VkBufferUsageFlags usageFlags,
			VmaAllocationCreateFlags vmaUsage,
			VkDeviceSize size,
			void* data = nullptr);

		virtual ~VulkanBuffer();

		void rename(const std::string& newName);

		// Get buffer address.
		uint64_t getDeviceAddress() const { CHECK(m_bSupportDeviceAddress); return m_deviceAddress; }

		// Get vkbuffer handle.
		operator VkBuffer() const { return m_buffer; }
		VkBuffer getVkBuffer() const { return m_buffer; }

		// Manually copy or other usage.
		void map(VkDeviceSize size = VK_WHOLE_SIZE);
		void* getMapped() const { return m_mapped; }
		void unmap();

		// Copy whole size of buffer.
		void copyTo(const void* data, VkDeviceSize size);

		void bind(VkDeviceSize offset = 0);

		void invalidate(VkDeviceSize size = VK_WHOLE_SIZE, VkDeviceSize offset = 0);

		void flush(VkDeviceSize size = VK_WHOLE_SIZE, VkDeviceSize offset = 0);

		// Copy stage buffer data to this buffer use vkCmdCopyBuffer.
		void stageCopyFrom(VkBuffer inBuffer, VkDeviceSize size, VkDeviceSize srcOffset = 0, VkDeviceSize destOffset = 0);

		// Helper function for init config vma flags.
		inline static VmaAllocationCreateFlags getStageCopyForUploadBufferFlags()
		{
			return VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
				VMA_ALLOCATION_CREATE_MAPPED_BIT;
		}

		// Helper function for init config vma flags.
		inline static VmaAllocationCreateFlags getReadBackFlags()
		{
			return VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
				VMA_ALLOCATION_CREATE_MAPPED_BIT;
		}

		const VkDescriptorBufferInfo& getDefaultInfo() const { return m_defaultInfo; }

	protected:
		// Buffer address.
		uint64_t m_deviceAddress = 0;

		// Buffer handle.
		VkBuffer m_buffer = VK_NULL_HANDLE;

		// VMA handle.
		VmaAllocation m_allocation = nullptr;

		// Mapped pointer for buffer.
		void* m_mapped = nullptr;

		// Support device address.
		bool m_bSupportDeviceAddress = false;

		VkDescriptorBufferInfo m_defaultInfo;
	};

	class VulkanImage : public GpuResource
	{
	public:
		explicit VulkanImage(
			const VulkanContext* context,
			const std::string& name,
			const VkImageCreateInfo& createInfo,
			VkMemoryPropertyFlags preperty = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

		virtual ~VulkanImage();

		VkImage getImage() const { return m_image; }

		VkFormat getFormat() const { return m_createInfo.format; }

		VkExtent3D getExtent() const { return m_createInfo.extent; }

		const VkImageCreateInfo& getInfo() const { return m_createInfo; }

		VkImageLayout getCurrentLayout(uint32_t layerIndex, uint32_t mipLevel) const;

		// Try get view and create if no exist.
		VkImageView getOrCreateView(
			VkImageSubresourceRange range, VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D);

		void transitionLayout(RHICommandBufferBase& cmd, VkImageLayout newLayout, VkImageSubresourceRange range);


		void transitionLayout(VkCommandBuffer cb, uint32_t cmdQueueFamily, VkImageLayout newLayout, VkImageSubresourceRange range);

		// Use graphics queue family.
		void transitionLayout(VkCommandBuffer cmd, VkImageLayout newLayout, VkImageSubresourceRange range);

		// Transition on major graphics.
		void transitionLayoutImmediately(VkImageLayout newLayout, VkImageSubresourceRange range);

	protected:
		size_t getSubresourceIndex(uint32_t layerIndex, uint32_t mipLevel) const;

	protected:
		// Image handle.
		VkImage m_image = VK_NULL_HANDLE;

		VmaAllocation m_allocation = nullptr;

		// Cache image create info.
		VkImageCreateInfo m_createInfo = {};

		// Image resource queue family owner.
		std::vector<uint32_t> m_ownerQueueFamilyIndices;

		// Image layouts.
		std::vector<VkImageLayout> m_layouts;

		// Cache created image views.
		std::unordered_map<uint64_t, VkImageView> m_cacheImageViews{ };
	};
}