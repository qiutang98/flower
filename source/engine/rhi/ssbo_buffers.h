#pragma once

#include "rhi_misc.h"
#include "resource.h"

namespace engine
{
	class BufferParameterPool
	{
	public:
		class BufferParameter : NonCopyable
		{
		private:
			std::unique_ptr<VulkanBuffer> m_buffer;
			size_t m_bufferSize;
			VkDescriptorType m_type;
			VkDescriptorBufferInfo m_bufferInfo;
		public:
			BufferParameter(
				const char* name,
				size_t bufferSize,
				VkBufferUsageFlags bufferUsage,
				VkDescriptorType type,
				VmaAllocationCreateFlags vmaUsage,
				void* data);

			const VkDescriptorBufferInfo& getBufferInfo() const
			{
				return m_bufferInfo;
			}

			VulkanBuffer* getBuffer() const
			{
				return m_buffer.get();
			}

			VkDescriptorType getType() const
			{
				return m_type;
			}

			template<typename T>
			void updateData(const T& in)
			{
				CHECK(m_bufferSize == sizeof(T));
				getBuffer()->copyTo(&in, m_bufferSize);
			}

			void updateDataPtr(const void* data)
			{
				getBuffer()->copyTo(data, m_bufferSize);
			}
		};

	private:
		// Own parameter buffers.
		std::vector<std::vector<std::shared_ptr<BufferParameter>>> m_ownPtr;
		std::vector<std::unordered_map<uint64_t, std::vector<std::shared_ptr<BufferParameter>>>> m_hashBufferPtr;

		// Current frame index.
		size_t m_index = 0;

	public:

		void tick();

		std::shared_ptr<BufferParameter> getParameter(
			const char* name,
			size_t bufferSize,
			VkBufferUsageFlags bufferUsage,
			VkDescriptorType type,
			VmaAllocationCreateFlags vmaUsage,
			void* data = nullptr);

		std::shared_ptr<BufferParameter> getStaticStorage(const char* name, size_t bufferSize, void* data = nullptr)
		{
			return getParameter(
				name,
				bufferSize,
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				VulkanBuffer::getStageCopyForUploadBufferFlags(),
				data);
		}

		std::shared_ptr<BufferParameter> getStaticStorageGPUOnly(const char* name, size_t bufferSize, void* data = nullptr)
		{
			return getParameter(
				name,
				bufferSize,
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				{},
				data);
		}

		std::shared_ptr<BufferParameter> getIndirectStorage(const char* name, size_t bufferSize, void* data = nullptr)
		{
			return getParameter(
				name,
				bufferSize,
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				{},
				data);
		}

		std::shared_ptr<BufferParameter> getStaticUniform(const char* name, size_t bufferSize, void* data = nullptr)
		{
			return getParameter(
				name,
				bufferSize,
				VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
				VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				VulkanBuffer::getStageCopyForUploadBufferFlags(),
				data);
		}
	};

	using BufferParameterHandle = std::shared_ptr<BufferParameterPool::BufferParameter>;
}