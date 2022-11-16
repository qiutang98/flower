#pragma once
#include "RHICommon.h"

namespace Flower
{
    class SamplerCache
    {
    private:
        struct SamplerCreateInfo
        {
            VkSamplerCreateInfo info;
            bool operator==(const SamplerCreateInfo& other) const;
            size_t hash() const;
        };
    
        struct SamplerCreateInfoHash
        {
            std::size_t operator()(const SamplerCreateInfo& k) const
            {
                return k.hash();
            }
        };

        // Bindless index and sampler combine.
        using Cache = std::unordered_map<SamplerCreateInfo, std::pair<uint32_t, VkSampler>, SamplerCreateInfoHash>;
        Cache m_cache;

        VkDescriptorSet m_cacheCommonDescriptor = VK_NULL_HANDLE;
        VkDescriptorSetLayout m_cacheCommonDescriptorSetLayout = VK_NULL_HANDLE;

        void initCommonDescriptorSet();
    public:
        void init();
        void release();

        VkSampler createSampler(VkSamplerCreateInfo info);
        VkSampler createSampler(VkSamplerCreateInfo info, uint32_t& outBindlessIndex);


        // Common descriptor set.
        // See layout in CommonSampler.glsl
        VkDescriptorSet getCommonDescriptorSet();
        VkDescriptorSetLayout getCommonDescriptorSetLayout();
    };
}