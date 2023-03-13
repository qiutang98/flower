#include "Pch.h"
#include "RenderTexturePool.h"

namespace Flower
{
	bool RenderTexturePool::PoolImage::isValid()
	{
		return
			m_pool != nullptr &&
			m_hashId != ~0 &&
			m_id != ~0 &&
			m_image.lock();
	}

	VulkanImage& RenderTexturePool::PoolImage::getImage()
	{
		CHECK(isValid());
		return *m_image.lock().get();
	}

	void RenderTexturePool::PoolImage::release()
	{
		if (!isValid())
		{
			return;
		}

		m_pool->releasePoolImage(*this);

		m_hashId = ~0;
		m_id = ~0;
		m_pool = nullptr;
	}

	PoolImageSharedRef RenderTexturePool::createPoolImage(
		const char* name,
		uint32_t width,
		uint32_t height,
		VkFormat format,
		VkImageUsageFlags usage,
		int32_t mipmapCount,
		uint32_t depth,
		uint32_t arrayLayers,
		VkSampleCountFlagBits sampleCount,
		VkImageCreateFlags flags)
	{
		VkImageType imgType = depth > 1 ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
		uint32_t trueMipmapLevels = mipmapCount == -1 ? getMipLevelsCount(width, height) : mipmapCount;

		VkImageCreateInfo info{};
		info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		info.flags = flags;
		info.imageType = imgType;
		info.format = format;
		info.extent.width = width;
		info.extent.height = height;
		info.extent.depth = depth;
		info.mipLevels = trueMipmapLevels;
		info.samples = sampleCount;
		info.usage = usage;
		info.arrayLayers = arrayLayers;

		// Some hardcode input.
		info.tiling = VK_IMAGE_TILING_OPTIMAL;
		info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		info.queueFamilyIndexCount = 0;
		info.pQueueFamilyIndices = nullptr;
		info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		return createPoolImage(name, info);
	}

	PoolImageSharedRef RenderTexturePool::createPoolCubeImage(
		const char* name, 
		uint32_t width, 
		uint32_t height, 
		VkFormat format, 
		VkImageUsageFlags usage, 
		int32_t mipmapCount, 
		VkSampleCountFlagBits sampleCount)
	{

		return createPoolImage(name, width, height, format, usage, mipmapCount, 1, 6, sampleCount, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT);
	}

	PoolImageSharedRef RenderTexturePool::createPoolImage(const char* name, const VkImageCreateInfo& info)
	{
		const size_t createInfoHash = CRCHash(info);

		PoolImageStorage storage(this);
		storage.poolInfo.m_hashId = createInfoHash;

		if (m_freeImages[createInfoHash].size() <= 0)
		{
			// No free image can use, create new one.
			storage.poolInfo.m_id = m_idAccumulator;
			storage.image = VulkanImage::create(name, info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

			storage.poolInfo.m_image = storage.image;

			// update id.
			m_idAccumulator++;
		}
		else
		{
			// exist free image, reuse.
			storage = m_freeImages[createInfoHash].back();

			CHECK(storage.poolInfo.m_id != ~0);
			CHECK(storage.poolInfo.m_freeCounter != ~0);
			CHECK(storage.poolInfo.m_image.lock());
			CHECK(storage.image);

			m_freeImages[createInfoHash].pop_back();

			// reset free counter.
			storage.poolInfo.m_freeCounter = ~0;

			storage.image->rename(name);
		}

		m_busyImages[createInfoHash].push_back(storage);

		PoolImageSharedRef result = std::shared_ptr<PoolImageRef>(new PoolImageRef());
		result->m_image = storage.poolInfo;
		return result;
	}

	void RenderTexturePool::releasePoolImage(const PoolImage& in)
	{
		CHECK(in.m_hashId != ~0);
		CHECK(in.m_image.lock());

		auto& busyArray = m_busyImages[in.m_hashId];

		poolSizeSafeCheck(busyArray.size());
		CHECK(busyArray.size() >= 1);

		PoolImageStorage storage(this);
		storage.poolInfo = in;

		for (int32_t i = 0; i < busyArray.size(); ++i)
		{
			if (busyArray[i].poolInfo.m_id == in.m_id)
			{
				// Store image owner.
				storage.image = busyArray[i].image;

				auto temp = busyArray[i];
				busyArray[i] = busyArray.back();

				busyArray.pop_back();
				break;
			}
		}
		CHECK(storage.image);
		CHECK(storage.poolInfo.m_image.lock() == storage.image);

		// Update free counter.
		storage.poolInfo.m_freeCounter = m_innerCounter;
		m_freeImages[in.m_hashId].push_back(storage);

		// Mark tick state active so we can release free rt immediately.
		m_bRecentRelease = true;
	}

	void RenderTexturePool::tick()
	{
		poolSizeSafeCheck(m_freeImages.size());
		poolSizeSafeCheck(m_busyImages.size());

		// Tick find free render texture which can release.
		if (m_bRecentRelease && (m_innerCounter % 5 == 0))
		{
			std::vector<size_t> unusedKey{};
			for (auto& pair : m_freeImages)
			{
				// Safe check.
				poolSizeSafeCheck(pair.second.size());

				if (pair.second.size() == 0)
				{
					unusedKey.push_back(pair.first);
				}
				else
				{
					std::erase_if(pair.second, [this](const auto& s) 
					{
						return shouldRelease(s.poolInfo.m_freeCounter);
					});
				}
			}

			// Shrink empty free image map.
			for (auto& key : unusedKey)
			{
				m_freeImages.erase(key);
			}

			if (m_freeImages.empty())
			{
				m_bRecentRelease = false;
			}
		}

		// Busy image map shrink.
		if (m_innerCounter % 11 == 0)
		{
			std::vector<size_t> unusedKey{};
			for (auto& busyPair : m_busyImages)
			{
				// Safe check.
				poolSizeSafeCheck(busyPair.second.size());

				if (busyPair.second.size() == 0)
				{
					unusedKey.push_back(busyPair.first);
				}
			}

			for (auto& key : unusedKey)
			{
				m_busyImages.erase(key);
			}
		}

		m_innerCounter ++;
	}
}