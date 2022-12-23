#include "Pch.h"
#include "DynamicUniformBuffer.h"
#include "RHI.h"
#include <vma/vk_mem_alloc.h>

namespace Flower
{
	DynamicUniformBuffer::DynamicUniformBuffer(uint32_t frameNum, uint32_t totalSize)
		: m_frameLoopNum(frameNum)
		, m_totoalSize(totalSize)
		, m_currentFrameID(0)
		, m_usedSize(0)
	{
		CHECK(frameNum >= 1);

		// Create buffer.
		m_buffer = VulkanBuffer::create(
			"DynamicUniformBuffer",
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			EVMAUsageFlags::StageCopyForUpload,
			m_totoalSize,
			nullptr
		);

		m_usedSizeCurrentFrames.resize(frameNum, 0);

		m_alginMin = RHI::get()->getPhysicalDeviceProperties().limits.minUniformBufferOffsetAlignment;

		VkDescriptorBufferInfo bufInfo = {};
		bufInfo.buffer = m_buffer->getVkBuffer();
		bufInfo.offset = 0;
		bufInfo.range = RHI::get()->getPhysicalDeviceProperties().limits.maxUniformBufferRange;

		// default set to binding position zero.
		RHI::get()->descriptorFactoryBegin()
			.bindBuffers(0, 1, &bufInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
			.build(m_set, m_layout);

		m_buffer->map();
	}

	DynamicUniformBuffer::~DynamicUniformBuffer()
	{
		m_buffer->unmap();
	}

	void DynamicUniformBuffer::onFrameStart()
	{
		// FrameID loop.
		m_currentFrameID ++;
		if (m_currentFrameID == m_frameLoopNum)
		{
			m_currentFrameID = 0;
		}

		// Reset used size of this frame.
		m_usedSizeCurrentFrames[m_currentFrameID] = 0;
	}



	uint32_t DynamicUniformBuffer::alloc(uint32_t size)
	{
		size = getAlignSize(size);

		m_usedSizeCurrentFrames[m_currentFrameID] += size;
		if (overflow())
		{
			LOG_WARN("Dynamic uniform buffer overflow, increase ring size on init.");
		}

		uint32_t scrOffset = m_usedSize;

		CHECK(scrOffset + RHI::get()->getPhysicalDeviceProperties().limits.maxUniformBufferRange < m_totoalSize);

		m_usedSize += size;
		if (m_usedSize >= (m_totoalSize - RHI::get()->getPhysicalDeviceProperties().limits.maxUniformBufferRange))
		{
			m_usedSize = 0;
		}

		return scrOffset;

#if 0
		// Flush to make changes visible to the host
		VkMappedMemoryRange memoryRange = vks::initializers::mappedMemoryRange();
		memoryRange.memory = uniformBuffers.dynamic.memory;
		memoryRange.size = uniformBuffers.dynamic.size;
		vkFlushMappedMemoryRanges(device, 1, &memoryRange);
#endif
	}

	bool DynamicUniformBuffer::overflow() const
	{
		uint32_t allSize = 0;
		for (const auto& s : m_usedSizeCurrentFrames)
		{
			allSize += s;
		}

		if (allSize >= m_totoalSize)
		{
			return true;
		}

		return false;
	}

	inline uint32_t AlignUp(uint32_t val, uint32_t alignment)
	{
		return (val + alignment - 1) & ~(alignment - 1);
	}

	uint32_t DynamicUniformBuffer::getAlignSize(uint32_t src) const
	{
		return AlignUp(src, m_alginMin);
	}
}