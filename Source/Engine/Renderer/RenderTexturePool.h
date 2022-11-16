#pragma once

#include "Pch.h"
#include "RendererCommon.h"

#include "../RHI/RHI.h"

namespace Flower
{
	class RenderTexturePool : NonCopyable
	{
	public:
		class PoolImage
		{
		private:
			friend class RenderTexturePool;

			size_t m_hashId = ~0;

			// Id for valid state vertify.
			size_t m_id = ~0;

			uint64_t m_freeCounter = ~0;

			std::weak_ptr<VulkanImage> m_image;
			RenderTexturePool* m_pool = nullptr;

		public:
			PoolImage() = default;
			PoolImage(RenderTexturePool* inPool) : m_pool(inPool) { }

			void release();
			bool isValid();

			VulkanImage& getImage();
		};

		class PoolImageRef : NonCopyable
		{
			friend class RenderTexturePool;

		private:
			PoolImage m_image;
			explicit PoolImageRef()
			{

			}

		public:
			~PoolImageRef()
			{
				m_image.release();
			}

			VulkanImage& getImage()
			{
				return m_image.getImage();
			}
		};

	private:
		friend PoolImage;

		// Recent some pool image release?
		bool m_bRecentRelease = false;
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

		
		void poolSizeSafeCheck(size_t in) const
		{
			sizeSafeCheck(in, 999);
		}

		// When texture resources release, no immediately free.
		// We push to our free pools, and when we recreate images, if can reuse, pop and push to busy images pool.
		std::unordered_map<size_t, std::vector<PoolImageStorage>> m_freeImages;
		std::unordered_map<size_t, std::vector<PoolImageStorage>> m_busyImages;

		bool shouldRelease(uint64_t freeCounter)
		{
			constexpr size_t freePeriod = 1;
			return m_innerCounter > freeCounter + freePeriod;
		}

		void releasePoolImage(const PoolImage& in);

	public:
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

		std::shared_ptr<PoolImageRef> createPoolCubeImage(
			const char* name,
			uint32_t width,
			uint32_t height,
			VkFormat format,
			VkImageUsageFlags usage,
			int32_t mipmapCount = 1,
			VkSampleCountFlagBits sampleCount = VK_SAMPLE_COUNT_1_BIT
		);

		std::shared_ptr<PoolImageRef> createPoolImage(
			const char* name,
			const VkImageCreateInfo& createInfo
		);

		void tick();
	};

	using PoolImageSharedRef = std::shared_ptr<RenderTexturePool::PoolImageRef>;
}