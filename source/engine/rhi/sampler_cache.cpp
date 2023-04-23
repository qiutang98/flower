#include "rhi.h"

namespace engine
{
    void SamplerCache::init(VulkanContext* context)
    {
        m_context = context;
        initCommonDescriptorSet();
    }

    void SamplerCache::release()
    {
        for (auto& cache : m_cache)
        {
            VkSampler sampler = cache.second.sampler;
            vkDestroySampler(m_context->getDevice(), sampler, nullptr);
        }
        m_cache.clear();
    }

    VkSampler SamplerCache::createSampler(VkSamplerCreateInfo info)
    {
        return createSamplerAndUpdateToBindless(info).sampler;
    }

    VkSampler SamplerCache::createSampler(VkSamplerCreateInfo info, uint32_t& bindless)
    {
        auto res = createSamplerAndUpdateToBindless(info);
        bindless = res.index;
        return res.sampler;
    }

    SamplerCache::SamplerWithIndex SamplerCache::createSamplerAndUpdateToBindless(VkSamplerCreateInfo info)
    {
        SamplerWithIndex result {};
        
        SamplerCreateInfo sci{};
        sci.info = info;

        auto it = m_cache.find(sci);
        if (it != m_cache.end())
        {
            result = it->second;
        }
        else
        {
            VkSampler sampler;
            RHICheck(vkCreateSampler(m_context->getDevice(), &sci.info, nullptr, &sampler));

            result.sampler = sampler;
            result.index = m_context->getBindlessSampler().updateSamplerToBindlessDescriptorSet(sampler);
           
            m_cache[sci] = result;
        }

        return result;
    }

    void SamplerCache::initCommonDescriptorSet()
    {
        // Point mipmap filter & point.
        VkDescriptorImageInfo pointClampEdgeInfo       { .sampler = createSampler(SamplerFactory::pointClampEdge()) };
        VkDescriptorImageInfo pointClampBorder0000Info { .sampler = createSampler(SamplerFactory::pointClampBorder0000()) };
        VkDescriptorImageInfo pointClampBorder1111Info { .sampler = createSampler(SamplerFactory::pointClampBorder1111()) };
        VkDescriptorImageInfo pointRepeatInfo          { .sampler = createSampler(SamplerFactory::pointRepeat()) };

        // Point mipmap filter & linear.
        VkDescriptorImageInfo linearClampEdgeInfo      { .sampler = createSampler(SamplerFactory::linearClampEdgeMipPoint()) };
        VkDescriptorImageInfo linearClampBorder0000Info{ .sampler = createSampler(SamplerFactory::linearClampBorder0000MipPoint()) };
        VkDescriptorImageInfo linearClampBorder1111Info{ .sampler = createSampler(SamplerFactory::linearClampBorder1111MipPoint()) };
        VkDescriptorImageInfo linearRepeatInfo         { .sampler = createSampler(SamplerFactory::linearRepeatMipPoint()) };

        VkDescriptorImageInfo linearClampEdgeMipInfo{ .sampler = createSampler(SamplerFactory::linearClampEdge()) };
        VkDescriptorImageInfo linearRepeatMipInfo{ .sampler = createSampler(SamplerFactory::linearRepeat()) };

        // See /shader/common_sampler.glsl
        m_context->descriptorFactoryBegin()
            .bindImages(0, 1, &pointClampEdgeInfo,        VK_DESCRIPTOR_TYPE_SAMPLER, kCommonShaderStage)
            .bindImages(1, 1, &pointClampBorder0000Info,  VK_DESCRIPTOR_TYPE_SAMPLER, kCommonShaderStage)
            .bindImages(2, 1, &pointRepeatInfo,           VK_DESCRIPTOR_TYPE_SAMPLER, kCommonShaderStage)
            .bindImages(3, 1, &linearClampEdgeInfo,       VK_DESCRIPTOR_TYPE_SAMPLER, kCommonShaderStage)
            .bindImages(4, 1, &linearClampBorder0000Info, VK_DESCRIPTOR_TYPE_SAMPLER, kCommonShaderStage)
            .bindImages(5, 1, &linearRepeatInfo,          VK_DESCRIPTOR_TYPE_SAMPLER, kCommonShaderStage)
            .bindImages(6, 1, &linearClampBorder1111Info, VK_DESCRIPTOR_TYPE_SAMPLER, kCommonShaderStage)
            .bindImages(7, 1, &pointClampBorder1111Info,  VK_DESCRIPTOR_TYPE_SAMPLER, kCommonShaderStage)
            .bindImages(8, 1, &linearClampEdgeMipInfo, VK_DESCRIPTOR_TYPE_SAMPLER, kCommonShaderStage)
            .bindImages(9, 1, &linearRepeatMipInfo, VK_DESCRIPTOR_TYPE_SAMPLER, kCommonShaderStage)
            .build(m_cacheCommonDescriptor, m_cacheCommonDescriptorSetLayout);
    }
}