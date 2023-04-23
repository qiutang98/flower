#include "ssbo_buffers.h"
#include <rhi/rhi.h>
#include <util/cityhash/city.h>
#include <util/crc.h>

namespace engine
{
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
			getContext(),
			name,
			bufferUsage,
			vmaUsage,
			bufferSize,
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

	void BufferParameterPool::tick()
	{
		m_index++;

		const size_t existNum = getExistNum();
		if (m_ownPtr.size() < existNum)
		{
			m_ownPtr.resize(existNum);
			m_hashBufferPtr.resize(existNum);
		}

		if (m_index >= existNum)
		{
			m_index = 0;
		}

		// Now can release this frame buffers.
		m_ownPtr[m_index].clear();
		m_hashBufferPtr[m_index].clear();
	}

	struct BufferIdentifier
	{
		size_t bufferSize;
		uint32_t bufferUsage;
		uint32_t type;
		uint32_t vmaUsage;

		size_t hash()
		{
			size_t seed = bufferSize;
			hashCombine(seed, bufferUsage);
			hashCombine(seed, type);
			hashCombine(seed, vmaUsage);
			return seed;
		}
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

		BufferIdentifier hash
		{
			.bufferSize  = bufferSize,
			.bufferUsage = (uint32_t)bufferUsage,
			.type        = (uint32_t)type,
			.vmaUsage    = (uint32_t)vmaUsage,
		};

		// Don't know why crc and cityhash will get different value here.
		// We use custom hasher.
		uint64_t requireHash = (uint64_t)hash.hash();

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

			return result;
		}
	}
}