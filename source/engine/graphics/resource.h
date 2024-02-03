#pragma once

#include "log.h"

#include <vma/vk_mem_alloc.h>
#include <cstdint>

namespace engine
{
	extern VkDeviceSize getAllocateGpuResourceSize();
	class GpuResource : NonCopyable
	{
	protected:
		std::string m_name;
		VkDeviceSize m_size;
		UUID64u m_runtimeUUID;

	public:
		GpuResource(const std::string name, VkDeviceSize size);
		virtual ~GpuResource();

		VkDeviceSize getSize() const 
		{ 
			return m_size; 
		}

		const std::string& getName() const 
		{ 
			return m_name; 
		}

		const UUID64u& getRuntimeUUID() const 
		{ 
			return m_runtimeUUID; 
		}
	};

	class VulkanBuffer : public GpuResource
	{
	public:
		explicit VulkanBuffer(
			VmaAllocator vma,
			const std::string& name,
			VkBufferUsageFlags usageFlags,
			VmaAllocationCreateFlags vmaUsage,
			VkDeviceSize size,
			void* data = nullptr);

		explicit VulkanBuffer(
			VkBufferUsageFlags usageFlags,
			VkMemoryPropertyFlags memoryPropertyFlags,
			const std::string& name,
			VkDeviceSize size,
			void* data = nullptr,
			bool bBindAfterCreate = true);

		virtual ~VulkanBuffer();
	
	private:
		void postInit(VkBufferUsageFlags usageFlags, void* data);

	public:
		void rename(const std::string& newName);

		// Get buffer address.
		uint64_t getDeviceAddress() const;

		// Get vkbuffer handle.
		operator VkBuffer() const { return m_buffer; }
		VkBuffer getVkBuffer() const { return m_buffer; }

		// Manually copy or other usage.
		void map(VkDeviceSize size = VK_WHOLE_SIZE);
		void* getMapped() const { return m_mapped; }
		void unmap();

		// Copy whole size of buffer.
		void copyTo(const void* data, VkDeviceSize size);

		// Used current buffer as stage buffer and copy to dest buffer.
		void copyAndUpload(VkCommandBuffer cmd, const void* data, VulkanBuffer* dstBuffer);

		void bind(VkDeviceSize offset = 0);

		void invalidate(VkDeviceSize size = VK_WHOLE_SIZE, VkDeviceSize offset = 0);

		void flush(VkDeviceSize size = VK_WHOLE_SIZE, VkDeviceSize offset = 0);

		// Copy stage buffer data to this buffer use vkCmdCopyBuffer.
		void stageCopyFrom(VkBuffer inBuffer, VkDeviceSize size, VkDeviceSize srcOffset = 0, VkDeviceSize destOffset = 0);

		// Helper function for init config vma flags.
		static inline VmaAllocationCreateFlags getStageCopyForUploadBufferFlags()
		{
			return VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
				VMA_ALLOCATION_CREATE_MAPPED_BIT;
		}

		// Helper function for init config vma flags.
		static inline VmaAllocationCreateFlags getReadBackFlags()
		{
			return VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
				VMA_ALLOCATION_CREATE_MAPPED_BIT;
		}

		const VkDescriptorBufferInfo& getDefaultInfo() const { return m_defaultInfo; }

	protected:
		bool m_bSupportDeviceAddress = false; // Support device address.
		uint64_t m_deviceAddress = 0; // Buffer address.

		// Buffer handle.
		VkBuffer m_buffer = VK_NULL_HANDLE;

		// VMA handle.
		VmaAllocator m_vma;
		VmaAllocation m_allocation = nullptr;
		VkDeviceMemory m_memory = VK_NULL_HANDLE;

		// Mapped pointer for buffer.
		void* m_mapped = nullptr;

		VkDescriptorBufferInfo m_defaultInfo;
	};

	struct ViewAndBindlessIndex
	{
		VkImageView view = VK_NULL_HANDLE;
		uint32_t srvBindless = ~0U;
	};

	class VulkanImage : public GpuResource
	{
	public:
		explicit VulkanImage(VmaAllocator vma, const std::string& name, VkImageCreateInfo createInfo);

		virtual ~VulkanImage();

		VkImage getImage() const { return m_image; }

		VkFormat getFormat() const { return m_createInfo.format; }

		VkExtent3D getExtent() const { return m_createInfo.extent; }

		const VkImageCreateInfo& getInfo() const { return m_createInfo; }

		VkImageLayout getCurrentLayout(uint32_t layerIndex, uint32_t mipLevel) const;

		// Try get view and create if no exist.
		ViewAndBindlessIndex getOrCreateView(
			VkImageSubresourceRange range = buildBasicImageSubresource(), VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D);

		void transitionLayout(RHICommandBufferBase& cmd, VkImageLayout newLayout, VkImageSubresourceRange range);


		void transitionLayout(VkCommandBuffer cb, uint32_t cmdQueueFamily, VkImageLayout newLayout, VkImageSubresourceRange range);

		// Use graphics queue family.
		void transitionLayout(
			VkCommandBuffer cmd, 
			VkImageLayout newLayout, 
			VkImageSubresourceRange range = buildBasicImageSubresource());

		void transitionLayoutDepth(
			VkCommandBuffer cmd,
			VkImageLayout newLayout)
		{
			transitionLayout(cmd, newLayout, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));
		}

		void transitionShaderReadOnly(VkCommandBuffer cmd, VkImageSubresourceRange range = buildBasicImageSubresource())
		{
			transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, range);
		}

		void transitionGeneral(VkCommandBuffer cmd, VkImageSubresourceRange range = buildBasicImageSubresource())
		{
			transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, range);
		}

		void transitionAttachment(VkCommandBuffer cmd, VkImageSubresourceRange range = buildBasicImageSubresource())
		{
			transitionLayout(cmd, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, range);
		}

		void transitionTransferSrc(VkCommandBuffer cmd, VkImageSubresourceRange range = buildBasicImageSubresource())
		{
			transitionLayout(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, range);
		}

		void transitionTransferDst(VkCommandBuffer cmd, VkImageSubresourceRange range = buildBasicImageSubresource())
		{
			transitionLayout(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, range);
		}

		// Transition on major graphics.
		void transitionLayoutImmediately(VkImageLayout newLayout, VkImageSubresourceRange range);

	protected:
		uint32_t getSubresourceIndex(uint32_t layerIndex, uint32_t mipLevel) const;

	protected:
		// Image handle.
		VkImage m_image = VK_NULL_HANDLE;

		VmaAllocator m_vma;
		VmaAllocation m_allocation = nullptr;

		// Cache image create info.
		VkImageCreateInfo m_createInfo = {};

		// Image resource queue family owner.
		struct SubResourceState
		{
			uint32_t ownerQueueFamilyIndex;
			VkImageLayout imageLayout;
		};
		std::vector<SubResourceState> m_subresourceStates;

		// Cache created image views.
		std::unordered_map<uint64_t, ViewAndBindlessIndex> m_cacheImageViews{ };
	};
}