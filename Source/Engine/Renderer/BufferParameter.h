#pragma once
#include "RendererCommon.h"

namespace Flower
{
	// Some buffer parameter need update by cpu, which need multi-buffer for 3-backbuffer swaphchain.

	class BufferParametersRing : NonCopyable
	{
	public:
		class BufferParameter : NonCopyable
		{
		private:
			std::shared_ptr<VulkanBuffer> m_buffer;
			VkDescriptorSet m_set;
			VkDescriptorSetLayout m_layout;
			size_t m_bufferSize;

		public:
			BufferParameter(
				const char* name,
				size_t bufferSize,
				VkBufferUsageFlags bufferUsage,
				VkDescriptorType type,
				EVMAUsageFlags vmaFlags,
				VkMemoryPropertyFlags memoryFlags);

			VkDescriptorBufferInfo getBufferInfo() const
			{
				return VkDescriptorBufferInfo{
					.buffer = m_buffer->getVkBuffer(),
					.offset = 0,
					.range = m_bufferSize
				};
			}

			std::shared_ptr<VulkanBuffer> getBuffer() const
			{
				return m_buffer;
			}

			VkDescriptorSet getSet() const
			{
				return m_set;
			}

			VkDescriptorSetLayout getLayout() const
			{
				return m_layout;
			}

			template<typename T>
			void updateData(const T& in)
			{
				CHECK(m_bufferSize == sizeof(T));

				getBuffer()->map(m_bufferSize);
				getBuffer()->copyTo(&in, m_bufferSize);
				getBuffer()->unmap();
			}

			void updateDataPtr(const void* data)
			{
				getBuffer()->map(m_bufferSize);
				getBuffer()->copyTo(data, m_bufferSize);
				getBuffer()->unmap();
			}
		};

		// BufferManager control buffer delay release state.
		// How to use:
		// You should create this guy with buffer usage flag and desciptor type.
		// And call tick every ticking.
		// When need this type buffer parameters.
		// Call getParameter and will get a new handle.
		class BufferParametersManager : NonCopyable
		{
		private:
			// Is this manager tick multi frame period.
			bool m_bMultiFrame;

			uint64_t m_tickCount = 0;
			EVMAUsageFlags m_vmaFlags;
			VkMemoryPropertyFlags m_memoryFlags;
			VkBufferUsageFlags m_bufferUsage;
			VkDescriptorType m_descriptorType;

			struct BufferParameterMisc
			{
				std::shared_ptr<BufferParameter> buffer;
				uint64_t freeTime = 0;
			};

			std::vector<BufferParameterMisc> m_unsortFreePool;
			std::vector<BufferParameterMisc> m_freePool;

			std::stack<size_t> m_unusedBusyPos;
			std::vector<BufferParameterMisc> m_busyPool;

		public:
			BufferParametersManager(
				bool bMultiFrame,
				VkBufferUsageFlags bufferUsage,
				VkDescriptorType type,
				EVMAUsageFlags vmaFlags,
				VkMemoryPropertyFlags memoryFlags);

			void tick();

			bool canClear() const { return m_busyPool.empty() && m_freePool.empty() && m_unsortFreePool.empty(); }

			bool isMultiFrame() const
			{
				return m_bMultiFrame;
			}

			class BufferParamHandle : NonCopyable
			{
			private:
				size_t m_busyPosition;
				BufferParametersManager* m_manager;

			public:
				~BufferParamHandle();

				BufferParamHandle(
					BufferParameter& inBuffer,
					BufferParametersManager* manager,
					size_t busyPosition);

				BufferParameter& buffer;
			};
			friend BufferParamHandle;

			std::shared_ptr<BufferParamHandle> getParameter(const char* name, size_t bufferSize);
		};

	private:
		size_t m_index = 0;

		std::unordered_map<size_t, std::vector<std::unique_ptr<BufferParametersManager>>> m_managersMap;

		struct BufferTypeHasher
		{
			bool bMultiFrame;
			VkBufferUsageFlags bufferUsage;
			VkDescriptorType type;
			EVMAUsageFlags vmaFlags;
			VkMemoryPropertyFlags memoryFlags;
		};

	public:
		explicit BufferParametersRing()
			: m_index(0)
		{

		}

		void tick();
		std::shared_ptr<BufferParametersManager::BufferParamHandle> getParameter(
			bool bMultiFrame,
			const char* name, 
			size_t bufferSize,
			VkBufferUsageFlags bufferUsage,
			VkDescriptorType type,
			EVMAUsageFlags vmaFlags,
			VkMemoryPropertyFlags memoryFlags);

		std::shared_ptr<BufferParametersManager::BufferParamHandle> getStaticStorage(const char* name, size_t bufferSize)
		{
			return getParameter(
				true,
				name, 
				bufferSize, 
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 
				EVMAUsageFlags::StageCopyForUpload, 
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		}

		std::shared_ptr<BufferParametersManager::BufferParamHandle> getStaticStorageGPUOnly(const char* name, size_t bufferSize)
		{
			return getParameter(
				true,
				name,
				bufferSize,
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				EVMAUsageFlags::GPUOnly,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		}

		std::shared_ptr<BufferParametersManager::BufferParamHandle> getStaticUniform(const char* name, size_t bufferSize)
		{
			return getParameter(
				true,
				name,
				bufferSize,
				VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
				VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				EVMAUsageFlags::StageCopyForUpload,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		}

		std::shared_ptr<BufferParametersManager::BufferParamHandle> getIndirectStorage(const char* name, size_t bufferSize)
		{
			return getParameter(
				false,
				name,
				bufferSize,
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				EVMAUsageFlags::GPUOnly,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		}
	};

	using BufferParamRefPointer = std::shared_ptr<BufferParametersRing::BufferParametersManager::BufferParamHandle>;
}