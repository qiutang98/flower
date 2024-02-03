#pragma once

#include <vulkan/vulkan.h>
#include <set>
#include <mutex>

namespace engine
{
	class VulkanImage;
	class VulkanBuffer;

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

		// Mutex for this bindless set allocate.
		std::mutex m_bindlessElementCountLock;

		const char* m_name;

		// Max bindless item use in this set limit by device.
		uint32_t m_maxDeviceLimitCount;
		uint32_t m_maxCountConfig;

		void initTemplate(const char* name, VkDescriptorType type, uint32_t maxCountConfig, uint32_t maxDeviceLimit);

		// Threadsafe free function.
		void freeBindless(uint32_t index);

	public:
		virtual ~BindlessBase() {};

		// Thread safe allocate.
		uint32_t getCountAndAndOne();

		// Getter.
		VkDescriptorSet getSet() const;
		VkDescriptorSetLayout getSetLayout() const;

		virtual void init(const char* name) = 0;
		virtual void release();
	};

	// Sampler.
	class BindlessSampler : public BindlessBase
	{
	public:
		virtual ~BindlessSampler() {};
		virtual void init(const char* name) override;

		// Register sampler to bindless descriptor set.
		// return bindless index.
		uint32_t updateSamplerToBindlessDescriptorSet(VkSampler in);

		void freeBindlessImpl(uint32_t index, VkSampler fallback = VK_NULL_HANDLE);
	};

	// Sampled image.
	class BindlessTexture : public BindlessBase
	{
	public:
		virtual ~BindlessTexture() {};

		virtual void init(const char* name) override;

		// Register vulkan image to bindless descriptor set.
		// return bindless index.
		uint32_t updateTextureToBindlessDescriptorSet(VkImageView view, VkImageLayout layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		void freeBindlessImpl(uint32_t index, VulkanImage* fallback = nullptr);
	};

	// Storage buffer.
	class BindlessStorageBuffer : public BindlessBase
	{
	public:
		virtual ~BindlessStorageBuffer() {};

		virtual void init(const char* name) override;

		// Register vulkan buffer to bindless descriptor set.
		// return bindless index.
		uint32_t updateBufferToBindlessDescriptorSet(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize range);

		virtual void freeBindlessImpl(uint32_t index, VulkanBuffer* fallback = nullptr);
	};
}