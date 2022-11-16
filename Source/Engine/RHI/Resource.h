#pragma once
#include "RHICommon.h"
#include "CommandBuffer.h"

namespace Flower
{
	class VulkanBuffer : NonCopyable
	{
	protected:
		std::string m_name;
		bool m_bHeap = false;
		VkDeviceSize m_size = 0;
		VkBuffer m_buffer = VK_NULL_HANDLE;
		VmaAllocation m_allocation = nullptr;
		VkDeviceMemory m_memory = VK_NULL_HANDLE;

		std::mutex m_mapMutex;

		uint64_t m_deviceAddress = 0;

		void lockMap()
		{
			m_mapMutex.lock();
		}

		void unlockMap()
		{
			m_mapMutex.unlock();
		}

	public:
		operator VkBuffer() const { return m_buffer; }
		operator VkDeviceMemory() const { return m_memory; }
		VkBuffer getVkBuffer() const { return m_buffer; }
		VkDeviceSize getSize() const { return m_size; }

	protected:
		// Buffer is heap or vma allocate?
		bool isHeap() const { return m_bHeap; }
		
		bool innerCreate(
			VkBufferUsageFlags usageFlags,
			VkMemoryPropertyFlags memoryPropertyFlags,
			VmaAllocationCreateFlags vmaUsage,
			void* data
		);

	public:
		virtual ~VulkanBuffer();
		VulkanBuffer() = default;

		

		uint64_t getDeviceAddress();

		// Mapped pointer for buffer.
		void* mapped = nullptr;

		VkResult map(VkDeviceSize size = VK_WHOLE_SIZE, VkDeviceSize offset = 0);
		void copyTo(const void* data, VkDeviceSize size);
		void unmap();

		VkResult bind(VkDeviceSize offset = 0);
		VkResult flush(VkDeviceSize size = VK_WHOLE_SIZE, VkDeviceSize offset = 0);
		VkResult invalidate(VkDeviceSize size = VK_WHOLE_SIZE, VkDeviceSize offset = 0);

		const char* getName() const { return m_name.c_str(); }
		void setName(const char* newName);

		static std::shared_ptr<VulkanBuffer> create(
			const char* name,
			VkBufferUsageFlags usageFlags,
			VkMemoryPropertyFlags memoryPropertyFlags,
			EVMAUsageFlags vmaFlags,
			VkDeviceSize size,
			void* data = nullptr // Copy data.
		);
		
		static std::shared_ptr<VulkanBuffer> create2(
			const char* name,
			VkBufferUsageFlags usageFlags,
			VkMemoryPropertyFlags memoryPropertyFlags,
			VmaAllocationCreateFlags vmaUsage,
			VkDeviceSize size,
			void* data = nullptr // Copy data.
		);

		static std::shared_ptr<VulkanBuffer> createRTScratchBuffer(const char* name, VkDeviceSize size);
	};

	class VulkanImage : NonCopyable
	{
	protected:
		std::string m_name;

		bool m_bHeap = false;
		VkDeviceSize m_size;

		VkImage m_image = VK_NULL_HANDLE;
		VkDeviceMemory m_memory = VK_NULL_HANDLE;
		VmaAllocation m_allocation = nullptr;
		VkImageCreateInfo m_createInfo = {};

		std::vector<uint32_t> m_ownerQueueFamilys;
		std::vector<VkImageLayout> m_layouts;

		std::unordered_map<size_t, VkImageView> m_cacheImageViews { };
	public:
		VkImage getImage() const { return m_image; }   
		VkFormat getFormat() const { return m_createInfo.format; }
		VkExtent3D getExtent() const { return m_createInfo.extent; }
		const VkImageCreateInfo& getInfo() const { return m_createInfo; }
		VkDeviceSize getMemorySize() const { return m_size; }

	protected:
		bool isHeap() const { return m_bHeap; }
		bool innerCreate(VkMemoryPropertyFlags preperty);

	public:
		virtual ~VulkanImage();
		VulkanImage() = default;

		void rename(const std::string& name);
		
		static std::shared_ptr<VulkanImage> create(
			const char* name, 
			const VkImageCreateInfo& createInfo, 
			VkMemoryPropertyFlags preperty = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

		// Try get view and create if no exist.
		VkImageView getView(VkImageSubresourceRange range, VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D);

		void transitionLayout(
			RHICommandBufferBase& cmd,
			VkImageLayout newLayout, 
			VkImageSubresourceRange range
		);

		// Use graphics queue family.
		void transitionLayout(
			VkCommandBuffer cmd,
			VkImageLayout newLayout,
			VkImageSubresourceRange range
		);

		void transitionLayout(
			VkCommandBuffer cb,
			uint32_t cmdQueueFamily,
			VkImageLayout newLayout,
			VkImageSubresourceRange range
		);

		// Transition on major graphics.
		void transitionLayoutImmediately(
			VkImageLayout newLayout,
			VkImageSubresourceRange range
		);

		VkImageLayout getCurrentLayout(uint32_t mipLevel) const { return m_layouts.at(mipLevel); }
	};
}