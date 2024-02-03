#include "uniform.h"
#include "context.h"
#include "resource.h"
#include "graphics.h"

namespace engine
{
	DynamicUniformBuffer::DynamicUniformBuffer(uint32_t frameNum, uint32_t initSize, uint32_t incrementSize)
		: m_frameLoopNum(frameNum)
		, m_totoalSize(initSize * 1024 * 1024)
		, m_currentFrameID(0)
		, m_usedSize(0)
		, m_incrementSize(incrementSize * 1024 * 1024)
	{
		ASSERT(frameNum >= 1, "Frame num at least need one or bigger.");
		m_alginMin = (uint32_t)getContext()->getPhysicalDeviceProperties().limits.minUniformBufferOffsetAlignment;

		releaseAndInit();
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
		// Need to reset used size.
		m_usedSize = 0;

		// FrameID loop.
		m_currentFrameID++;
		if (m_currentFrameID == m_frameLoopNum)
		{
			m_currentFrameID = 0;
		}

		// Never shrink.
		if (m_bShouldIncSize)
		{
			m_bShouldIncSize = false;
			m_totoalSize += m_incrementSize;

			getContext()->waitDeviceIdle();
			releaseAndInit();
		}
	}

	uint32_t DynamicUniformBuffer::alloc(uint32_t size)
	{
		uint32_t scrOffset = m_usedSize;

		// Add align size.
		m_usedSize += getAlignSize(size);

		// NOTE: We use max uniform buffer range set here, so need to ensure at least it never overflow.
		if (m_usedSize >= (m_totoalSize - getContext()->getPhysicalDeviceProperties().limits.maxUniformBufferRange))
		{
			LOG_TRACE("Dynamic uniform buffer overflow, will increment next time loop.");
			m_bShouldIncSize = true;

			// When overflow, this frame will render error, and will fix in next frame.
			m_usedSize = 0;
			scrOffset = 0;
		}

		return scrOffset;
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

		for (uint32_t i = 0; i < m_frameLoopNum; i ++)
		{
			// Create buffer.
			m_buffers[i] = std::make_unique<VulkanBuffer>(
				getContext()->getVMAFrequencyBuffer(),
				std::format("DynamicUniformBuffer{}", i).c_str(),
				VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
				VulkanBuffer::getStageCopyForUploadBufferFlags(),
				m_totoalSize,
				nullptr
			);

			VkDescriptorBufferInfo bufInfo = {};
			bufInfo.buffer = m_buffers[i]->getVkBuffer();
			bufInfo.offset = 0;
			bufInfo.range = getContext()->getPhysicalDeviceProperties().limits.maxUniformBufferRange;

			// default set to binding position zero.
			getContext()->descriptorFactoryBegin()
				.bindBuffers(0, 1, &bufInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, kCommonShaderStage)
				.build(m_sets[i], m_layouts[i]);

			m_buffers[i]->map();
		}
	}

	uint32_t DynamicUniformBuffer::getAlignSize(uint32_t src) const
	{
		return alignUp(src, m_alginMin);
	}
}