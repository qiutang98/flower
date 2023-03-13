#include "Pch.h"
#include "DynamicUniformBuffer.h"
#include "RHI.h"
#include <vma/vk_mem_alloc.h>

namespace Flower
{
	DynamicUniformBuffer::DynamicUniformBuffer(uint32_t frameNum, uint32_t totalSize, uint32_t incSize)
		: m_frameLoopNum(frameNum)
		, m_totoalSize(totalSize)
		, m_currentFrameID(0)
		, m_usedSize(0)
		, m_incrementSize(incSize)
	{
		CHECK(frameNum >= 1);
		m_alginMin = RHI::get()->getPhysicalDeviceProperties().limits.minUniformBufferOffsetAlignment;

		releaseAndInit();
	}

	void DynamicUniformBuffer::releaseAndInit()
	{
		for (auto& buffer : m_buffers)
		{
			buffer->unmap();
		}

		m_currentFrameID = 0;
		m_usedSize = 0;

		m_buffers.resize(m_frameLoopNum);
		m_sets.resize(m_frameLoopNum);
		m_layouts.resize(m_frameLoopNum);

		for (uint32_t i = 0; i < m_frameLoopNum; i++)
		{
			// Create buffer.
			m_buffers[i] = VulkanBuffer::create(
				std::format("DynamicUniformBuffer{}", i).c_str(),
				VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				EVMAUsageFlags::StageCopyForUpload,
				m_totoalSize,
				nullptr
			);

			VkDescriptorBufferInfo bufInfo = {};
			bufInfo.buffer = m_buffers[i]->getVkBuffer();
			bufInfo.offset = 0;
			bufInfo.range = RHI::get()->getPhysicalDeviceProperties().limits.maxUniformBufferRange;

			// default set to binding position zero.
			RHI::get()->descriptorFactoryBegin()
				.bindBuffers(0, 1, &bufInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
				.build(m_sets[i], m_layouts[i]);

			m_buffers[i]->map();
		}
	}

	DynamicUniformBuffer::~DynamicUniformBuffer()
	{
		for (auto& buffer : m_buffers)
		{
			buffer->unmap();
		}
	}

	void DynamicUniformBuffer::onFrameStart()
	{
		m_usedSize = 0;

		// FrameID loop.
		m_currentFrameID ++;
		if (m_currentFrameID == m_frameLoopNum)
		{
			m_currentFrameID = 0;
		}

		// Never shrink.
		if (m_bShouldIncSize)
		{
			m_bShouldIncSize = false;
			m_totoalSize += m_incrementSize;
			vkDeviceWaitIdle(RHI::Device);
			releaseAndInit();
		}
	}

	uint32_t DynamicUniformBuffer::alloc(uint32_t size)
	{
		uint32_t scrOffset = m_usedSize;

		// Add align size.
		m_usedSize += getAlignSize(size);
		if (m_usedSize >= (m_totoalSize - RHI::get()->getPhysicalDeviceProperties().limits.maxUniformBufferRange))
		{
			LOG_TRACE("Dynamic uniform buffer overflow, will increment next time loop.");
			m_bShouldIncSize = true;

			// When overflow, this frame will render error, and will fix in next frame.
			m_usedSize = 0;
			scrOffset = 0;
		}

		return scrOffset;
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