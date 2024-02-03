#pragma once

#include "base.h"
#include "log.h"

namespace engine
{
	class SamplerCache : NonCopyable
	{
	public:
		struct SamplerWithIndex
		{
			VkSampler sampler; // sampler handle.
			uint32_t index;    // index in bindless set.
		};

	private:
		struct SamplerCreateInfo
		{
			VkSamplerCreateInfo info { };

			bool operator==(const SamplerCreateInfo& other) const
			{
				return other.hash() == hash();
			}

			uint32_t hash() const
			{
				return crc::crc32((const char*)&info, sizeof(info));
			}
		};

		struct SamplerCreateInfoHash
		{
			uint32_t operator()(const SamplerCreateInfo& k) const
			{
				return k.hash();
			}
		};

		// Bindless index and sampler combine.
		using Cache = std::unordered_map<SamplerCreateInfo, SamplerWithIndex, SamplerCreateInfoHash>;

		// Cache all sampler.
		Cache m_cache;

		// Common sampler.
		VkDescriptorSet m_cacheCommonDescriptor = VK_NULL_HANDLE;
		VkDescriptorSetLayout m_cacheCommonDescriptorSetLayout = VK_NULL_HANDLE;

		void initCommonDescriptorSet();

	public:
		void init();
		void release();

		VkSampler createSampler(VkSamplerCreateInfo info);
		VkSampler createSampler(VkSamplerCreateInfo info, uint32_t& bindless);
		SamplerWithIndex createSamplerAndUpdateToBindless(VkSamplerCreateInfo info);

		// Common descriptor set.
		// See layout in /shader/common_sampler.glsl
		VkDescriptorSet getCommonDescriptorSet() const { return m_cacheCommonDescriptor; }
		VkDescriptorSetLayout getCommonDescriptorSetLayout() const { return m_cacheCommonDescriptorSetLayout; }
	};
}