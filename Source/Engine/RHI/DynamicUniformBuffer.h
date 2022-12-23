#pragma once
#include "RHICommon.h"

namespace Flower
{
	class VulkanBuffer;
	class DynamicUniformBuffer
	{
	private:
		std::shared_ptr<VulkanBuffer> m_buffer = nullptr;

		uint32_t m_frameLoopNum = 0;
		uint32_t m_totoalSize;

		uint32_t m_alginMin = 0;

		uint32_t m_currentFrameID = 0;

		std::vector<uint32_t> m_usedSizeCurrentFrames;
		uint32_t m_usedSize = 0;


		VkDescriptorSet m_set;
		VkDescriptorSetLayout m_layout;

	public:
		uint32_t getAlignSize(uint32_t src) const;

		DynamicUniformBuffer(uint32_t frameNum, uint32_t totalSize);
		~DynamicUniformBuffer();

		void onFrameStart();

		uint32_t alloc(uint32_t size);

		auto getBuffer() { return m_buffer.get(); }

		VkDescriptorSet getSet() const { return m_set; }
		VkDescriptorSetLayout getSetlayout() const { return m_layout; }

		

	private:
		bool overflow() const;
	};

}