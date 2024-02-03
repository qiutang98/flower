#pragma once
#include "resource.h"

namespace engine
{
	constexpr int32_t kRenderTextureFullMip = -1;

	// Render texture pool can reused texture in one frame.
	class RenderTexturePool : NonCopyable
	{
	public:
		class PoolImage
		{
		private:
			friend class RenderTexturePool;

			// Hash id of this image.
			uint32_t m_hashId = ~0;

			// Id for valid state vertify.
			size_t m_id = ~0;

			uint64_t m_freeCounter = ~0;

			// Image reference.
			std::weak_ptr<VulkanImage> m_image;

			// Pool reference.
			RenderTexturePool* m_pool = nullptr;

		public:
			PoolImage() = default;
			PoolImage(RenderTexturePool* inPool) : m_pool(inPool) { }

			// Release image from pool.
			void release();

			// Valid state check.
			bool isValid();

			// Get image resource.
			VulkanImage& getImage();
		};

		// Reference wrapper used for auto release.
		class PoolImageRef : NonCopyable
		{
		private:
			friend class RenderTexturePool;

			// Owner image.
			PoolImage m_image;

			// Hide construct ensure only pool can create resource.
			explicit PoolImageRef() { }

		public:
			// Release when de-construct.
			~PoolImageRef()
			{
				m_image.release();
			}

			// Get image.
			VulkanImage& getImage()
			{
				return m_image.getImage();
			}
		};

	private:
		friend PoolImage;

		// Recent some pool image release state.
		bool m_bRecentRelease = false;

		// ID accumulate of resource create by this pool, debug purpose.
		// Can used to valid resource state.
		size_t m_idAccumulator = 0;


		uint64_t m_innerCounter = 0;

		struct PoolImageStorage
		{
			std::shared_ptr<VulkanImage> image = nullptr;
			PoolImage poolInfo;

			explicit PoolImageStorage(RenderTexturePool* inPool)
				: poolInfo(PoolImage(inPool))
			{

			}
		};

		// When texture resources release, no immediately free.
		// We push to our free pools, and when we recreate images, if can reuse, pop and push to busy images pool.
		std::unordered_map<uint32_t, std::vector<PoolImageStorage>> m_freeImages;
		std::unordered_map<uint32_t, std::vector<PoolImageStorage>> m_busyImages;

		// When free time is bigger one frame we can release.
		bool shouldRelease(uint64_t freeCounter);

		// Release pool image.
		void releasePoolImage(const PoolImage& in);

	public:
		explicit RenderTexturePool() = default;

		// 2D wrapper of pool image create.
		std::shared_ptr<PoolImageRef> createPoolImage(
			const char* name,
			uint32_t width,
			uint32_t height,
			VkFormat format,
			VkImageUsageFlags usage,
			int32_t mipmapCount = 1,
			uint32_t depth = 1,
			uint32_t arrayLayers = 1,
			VkSampleCountFlagBits sampleCount = VK_SAMPLE_COUNT_1_BIT,
			VkImageCreateFlags flags = 0
		);

		// Cube wrapper of pool image create.
		std::shared_ptr<PoolImageRef> createPoolCubeImage(
			const char* name,
			uint32_t width,
			uint32_t height,
			VkFormat format,
			VkImageUsageFlags usage,
			int32_t mipmapCount = 1,
			VkSampleCountFlagBits sampleCount = VK_SAMPLE_COUNT_1_BIT
		);

		// Create or reused pool image, resource lifetime will auto managed by pool.
		std::shared_ptr<PoolImageRef> createPoolImage(
			const char* name,
			const VkImageCreateInfo& createInfo
		);

		// Tick update pool resource state.
		void tick();
	};

	using PoolImageSharedRef = std::shared_ptr<RenderTexturePool::PoolImageRef>;

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
		std::vector<std::unordered_map<uint32_t, std::vector<std::shared_ptr<BufferParameter>>>> m_hashBufferPtr;

		// Current frame index.
		size_t m_index = 0;

	public:
		explicit BufferParameterPool();

		void tick();

		std::shared_ptr<BufferParameter> getParameter(
			const char* name,
			size_t bufferSize,
			VkBufferUsageFlags bufferUsage,
			VkDescriptorType type,
			VmaAllocationCreateFlags vmaUsage,
			void* data = nullptr);

		std::shared_ptr<BufferParameter> getStaticStorage(
			const char* name, 
			size_t bufferSize, 
			void* data = nullptr)
		{
			return getParameter(
				name,
				bufferSize,
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | 
				VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				VulkanBuffer::getStageCopyForUploadBufferFlags(),
				data);
		}

		std::shared_ptr<BufferParameter> getStaticStorageGPUOnly(
			const char* name, 
			size_t bufferSize)
		{
			return getParameter(
				name,
				bufferSize,
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | 
				VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				{},
				nullptr);
		}

		std::shared_ptr<BufferParameter> getIndirectStorage(
			const char* name, 
			size_t bufferSize, 
			void* data = nullptr)
		{
			return getParameter(
				name,
				bufferSize,
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT  | 
				VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | 
				VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				{},
				data);
		}

		std::shared_ptr<BufferParameter> getStaticUniform(
			const char* name, 
			size_t bufferSize, 
			void* data = nullptr)
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

	// NOTE: Each BufferParameterHandle use a single buffer as container, so safe to save cross multi-frame.
	using BufferParameterHandle = std::shared_ptr<BufferParameterPool::BufferParameter>;
}