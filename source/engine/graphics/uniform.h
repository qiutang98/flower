#pragma once

#include "log.h"

namespace engine
{
	class VulkanBuffer;

	// A ring dynamic uniform buffer.
	class DynamicUniformBuffer
	{
	private:
		// Increment state.
		bool m_bShouldIncSize = false;

		// Fix config.
		uint32_t m_incrementSize = 0;
		uint32_t m_frameLoopNum = 0;
		uint32_t m_alginMin = 0;
		uint32_t m_totoalSize;

		// Dynamic change state.
		std::vector<std::unique_ptr<VulkanBuffer>> m_buffers;
		std::vector<VkDescriptorSet> m_sets;
		std::vector<VkDescriptorSetLayout> m_layouts;
		uint32_t m_currentFrameID = 0;
		uint32_t m_usedSize = 0;

	public:
		explicit DynamicUniformBuffer(uint32_t frameNum, uint32_t initSize, uint32_t incrementSize);
		~DynamicUniformBuffer();

		// Call this when frame start.
		void onFrameStart();

		uint32_t getAlignSize(uint32_t src) const;
		uint32_t alloc(uint32_t size);

		const VulkanBuffer* getBuffer() const { return m_buffers[m_currentFrameID].get(); }
		VkDescriptorSet getSet() const { return m_sets[m_currentFrameID]; }
		VkDescriptorSetLayout getSetlayout() const { return m_layouts[m_currentFrameID]; }

	private:
		void releaseAndInit();
	};

}