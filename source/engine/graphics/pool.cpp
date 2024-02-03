#include "pool.h"
#include "context.h"

namespace engine
{
	static AutoCVarInt32 cVarRHIRenderTargetPoolMaxElementNum(
		"r.RHI.RenderTargetPoolMaxElementNum",
		"Max element num in render target pool.",
		"RHI",
		999,
		CVarFlags::ReadAndWrite
	);

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

		// Set state to unvalid.
		m_hashId = ~0;
		m_id = ~0;
		m_pool = nullptr;

		// Final ensure.
		CHECK(!isValid());
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
		uint32_t trueMipmapLevels = mipmapCount == kRenderTextureFullMip ? getMipLevelsCount(width, height) : mipmapCount;

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
		// Hash by image create info.
		const uint32_t createInfoHash = crc::crc32((const char*)&info, sizeof(info));

		PoolImageStorage storage(this);
		storage.poolInfo.m_hashId = createInfoHash;

		if (m_freeImages[createInfoHash].size() <= 0)
		{
			// No free image can use, create new one.
			storage.poolInfo.m_id = m_idAccumulator;
			storage.image = std::make_shared<VulkanImage>(getContext()->getVMAFrequencyImage(), name, info);
			storage.poolInfo.m_image = storage.image;

			// update id.
			m_idAccumulator++;

			// NOTE: When create resource, it never free, so should set to ~0.
			CHECK(storage.poolInfo.m_freeCounter == ~0);
		}
		else
		{
			// Exist free image, reuse.
			storage = m_freeImages[createInfoHash].back();

			// State check.
			CHECK(storage.poolInfo.m_id != ~0);
			CHECK(storage.poolInfo.m_freeCounter != ~0); // NOTE: free counter must not be ~0.
			CHECK(storage.poolInfo.m_image.lock());
			CHECK(storage.image);

			// Pop back for reused.
			m_freeImages[createInfoHash].pop_back();

			// Reset free counter.
			storage.poolInfo.m_freeCounter = ~0;
		}

		// Insert to busy image pool.
		m_busyImages[createInfoHash].push_back(storage);

		// Build resource handle.
		PoolImageSharedRef result = std::shared_ptr<PoolImageRef>(new PoolImageRef());
		result->m_image = storage.poolInfo;

		// Get final used result.
		return result;
	}

	bool RenderTexturePool::shouldRelease(uint64_t freeCounter)
	{
		return m_innerCounter > freeCounter + getContext()->getSwapchain().getBackbufferCount();
	}

	void RenderTexturePool::releasePoolImage(const PoolImage& in)
	{
		// Valid state.
		CHECK(in.m_hashId != ~0);
		CHECK(in.m_image.lock());

		auto& busyArray = m_busyImages[in.m_hashId];

		// Safe check.
		ASSERT(busyArray.size() < cVarRHIRenderTargetPoolMaxElementNum.get(), "Toon much render texture used in pool, maybe exist some logic error!");
		ASSERT(busyArray.size() >= 1, "At least one image when call release.");

		// Prepare a storage.
		PoolImageStorage storage(this);
		storage.poolInfo = in;

		// Find by id and remove.
		for (int32_t i = 0; i < busyArray.size(); ++i)
		{
			if (busyArray[i].poolInfo.m_id == in.m_id)
			{
				// Store image owner.
				storage.image = busyArray[i].image;

				// Pop from busy array.
				auto temp = busyArray[i];
				busyArray[i] = busyArray.back();
				busyArray.pop_back();

				break;
			}
		}
		CHECK(storage.image);
		CHECK(storage.poolInfo.m_image.lock() == storage.image);

		// Update free counter, note that we set free counter to inner counter to safe check.
		storage.poolInfo.m_freeCounter = m_innerCounter;
		m_freeImages[in.m_hashId].push_back(storage);

		// Mark tick state active so we can release free rt immediately.
		m_bRecentRelease = true;
	}

	void RenderTexturePool::tick()
	{
		// Tick find free render texture which can release, tick per five frame.
		if (m_bRecentRelease && (m_innerCounter % 5 == 0))
		{
			std::vector<uint32_t> unusedKey{};
			for (auto& pair : m_freeImages)
			{
				// Safe check.
				ASSERT(pair.second.size() < cVarRHIRenderTargetPoolMaxElementNum.get(), "Toon much render texture used in pool, maybe exist some logic error!");

				if (pair.second.size() == 0)
				{
					unusedKey.push_back(pair.first);
				}
				else
				{
					std::erase_if(pair.second, [this](const auto& s) { return shouldRelease(s.poolInfo.m_freeCounter); });
				}
			}

			// Shrink empty free image map.
			for (auto& key : unusedKey)
			{
				m_freeImages.erase(key);
			}

			// We only set flag when all free images free.
			if (m_freeImages.empty())
			{
				m_bRecentRelease = false;
			}
		}

		// Busy image map shrink, tick per 11 frame.
		if (m_innerCounter % 11 == 0)
		{
			std::vector<uint32_t> unusedKey{};
			for (auto& busyPair : m_busyImages)
			{
				// Safe check.
				ASSERT(busyPair.second.size() < cVarRHIRenderTargetPoolMaxElementNum.get(), "Toon much render texture used in pool, maybe exist some logic error!");

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

		// Inner counter for frame.
		m_innerCounter++;
	}

	BufferParameterPool::BufferParameter::BufferParameter(
		const char* name,
		size_t bufferSize,
		VkBufferUsageFlags bufferUsage,
		VkDescriptorType type,
		VmaAllocationCreateFlags vmaUsage,
		void* data)
		: m_bufferSize(bufferSize)
		, m_type(type)
	{
		// Create buffer and set for use convenience.
		m_buffer = std::make_unique<VulkanBuffer>(
			getContext()->getVMAFrequencyBuffer(),
			name,
			bufferUsage,
			vmaUsage,
			(uint32_t)bufferSize,
			data
		);

		m_bufferInfo = VkDescriptorBufferInfo
		{
			.buffer = m_buffer->getVkBuffer(),
			.offset = 0,
			.range = m_bufferSize
		};
	}

	size_t getSafeReusedNum()
	{
		return getContext()->getSwapchain().getBackbufferCount() + 1;
	}

	size_t getExistNum()
	{
		return getSafeReusedNum() * 2;
	}

	BufferParameterPool::BufferParameterPool()
	{
		const size_t existNum = getExistNum();

		m_ownPtr.resize(existNum);
		m_hashBufferPtr.resize(existNum);
	}

	void BufferParameterPool::tick()
	{
		m_index++;
		if (m_index >= getExistNum())
		{
			m_index = 0;
		}

		// Now can release this frame buffers.
		m_ownPtr[m_index].clear();
		m_hashBufferPtr[m_index].clear();
	}

	struct BufferIdentifier
	{
		uint32_t bufferSize;
		uint32_t bufferUsage;
		uint32_t type;
		uint32_t vmaUsage;
	};

	BufferParameterHandle BufferParameterPool::getParameter(
		const char* name,
		size_t bufferSize,
		VkBufferUsageFlags bufferUsage,
		VkDescriptorType type,
		VmaAllocationCreateFlags vmaUsage,
		void* data)
	{
		const size_t reuseIdAdd = m_index + getSafeReusedNum();
		const size_t existNum = getExistNum();

		const size_t reuseId = reuseIdAdd % existNum;
		auto& reuseMap = m_hashBufferPtr[reuseId];

		BufferIdentifier hash = {};
		hash.bufferSize  = (uint32_t)bufferSize;
		hash.bufferUsage = (uint32_t)bufferUsage;
		hash.type        = (uint32_t)type;
		hash.vmaUsage    = (uint32_t)vmaUsage;

		uint32_t requireHash = crc::crc32(&hash, sizeof(hash));

		if (reuseMap[requireHash].empty())
		{
			auto newBuffer = std::make_shared<BufferParameter>(
				name,
				bufferSize,
				bufferUsage,
				type,
				vmaUsage,
				data);

			// Push in new vector.
			m_ownPtr[m_index].push_back(newBuffer);
			m_hashBufferPtr[m_index][requireHash].push_back(newBuffer);

			return newBuffer;
		}
		else
		{
			// Reused and pop from old vector.
			auto& result = reuseMap[requireHash].back();
			reuseMap[requireHash].pop_back();

			result->getBuffer()->rename(name);

			// Push in new vector.
			m_ownPtr[m_index].push_back(result);
			m_hashBufferPtr[m_index][requireHash].push_back(result);

			if (data)
			{
				result->getBuffer()->copyTo(data, bufferSize);
			}

			return result;
		}
	}
}