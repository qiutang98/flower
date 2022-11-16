#pragma once
#include "RHICommon.h"

namespace Flower
{
	constexpr uint32_t MAX_BINDLESS_COUNT = 10000u;

	class BindlessBase
	{
	protected:
		struct BindlessTextureDescriptorHeap
		{
			VkDescriptorSetLayout setLayout{};
			VkDescriptorPool descriptorPool{};
			VkDescriptorSet descriptorSetUpdateAfterBind{};
		};
		BindlessTextureDescriptorHeap m_bindlessDescriptorHeap;

		uint32_t m_bindlessElementCount = 0;
		std::set<uint32_t> m_freeIndex;
		std::mutex m_bindlessElementCountLock;

		void initTemplate(VkDescriptorType type);

	public:
		virtual ~BindlessBase() {};

		uint32_t getCountAndAndOne();
		void freeBindless(uint32_t index);
		VkDescriptorSet getSet();
		VkDescriptorSetLayout getSetLayout();

		virtual void init() = 0;
		virtual void release();
	};

	class BindlessSampler : public BindlessBase
	{
	public:
		virtual ~BindlessSampler() {};
		virtual void init() override;

		// Register sampler to bindless descriptor set.
		// return bindless index.
		uint32_t updateSamplerToBindlessDescriptorSet(VkSampler in);
	};

	class BindlessTexture : public BindlessBase
	{
	public:
		virtual ~BindlessTexture() {};

		virtual void init() override;

		// Register vulkan image to bindless descriptor set.
		// return bindless index.
		uint32_t updateTextureToBindlessDescriptorSet(
			VkImageView view, 
			VkImageLayout layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	};

	class BindlessStorageBuffer : public BindlessBase
	{
	public:
		virtual ~BindlessStorageBuffer() {};

		virtual void init() override;

		// Register vulkan buffer to bindless descriptor set.
		// return bindless index.
		uint32_t updateBufferToBindlessDescriptorSet(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize range);
	};

	namespace Bindless
	{
		extern BindlessTexture* const Texture;
		extern BindlessSampler* const Sampler;
	}
}