#pragma once
#include "RHICommon.h"

namespace Flower
{
	class VulkanBuffer;
	class DynamicUniformBuffer
	{
	private:
		std::vector<std::shared_ptr<VulkanBuffer>> m_buffers;
		bool m_bShouldIncSize = false;
		uint32_t m_frameLoopNum = 0;
		uint32_t m_totoalSize;
		uint32_t m_incrementSize = 0;
		uint32_t m_alginMin = 0;

		uint32_t m_currentFrameID = 0;

		uint32_t m_usedSize = 0;

		std::vector<VkDescriptorSet> m_sets;
		std::vector<VkDescriptorSetLayout> m_layouts;

	public:
		uint32_t getAlignSize(uint32_t src) const;

		DynamicUniformBuffer(uint32_t frameNum, uint32_t totalSize, uint32_t incSize);
		~DynamicUniformBuffer();

		void onFrameStart();

		uint32_t alloc(uint32_t size);

		auto getBuffer() { return m_buffers[m_currentFrameID].get(); }

		VkDescriptorSet getSet() const { return m_sets[m_currentFrameID]; }
		VkDescriptorSetLayout getSetlayout() const { return m_layouts[m_currentFrameID]; }

		

	private:
		void releaseAndInit();
	};

}