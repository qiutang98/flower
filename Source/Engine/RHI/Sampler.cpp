#include "Pch.h"
#include "RHI.h"
#include "Sampler.h"

namespace Flower
{
    void SamplerCache::init()
    {

    }

    void SamplerCache::release()
    {
        for (auto& cache : m_cache)
        {
            vkDestroySampler(RHI::Device, cache.second.second, nullptr);
        }
        m_cache.clear();
    }

    VkSampler SamplerCache::createSampler(VkSamplerCreateInfo info)
    {
        uint32_t index = 0;
        return createSampler(info, index);
    }

    VkSampler SamplerCache::createSampler(VkSamplerCreateInfo info, uint32_t& outBindlessIndex)
    {
        SamplerCreateInfo sci{};
        sci.info = info;

        auto it = m_cache.find(sci);
        if (it != m_cache.end())
        {
            outBindlessIndex = it->second.first;
            return (*it).second.second;
        }
        else
        {
            VkSampler sampler;
            RHICheck(vkCreateSampler(RHI::Device, &sci.info, nullptr, &sampler));

            uint32_t bindlessIndex = Bindless::Sampler->updateSamplerToBindlessDescriptorSet(sampler);
            outBindlessIndex = bindlessIndex;
            m_cache[sci] = { bindlessIndex, sampler };
            return sampler;
        }
    }

    bool SamplerCache::SamplerCreateInfo::operator==(const SamplerCreateInfo& other) const
    {
        return other.hash() == hash();
    }

    size_t SamplerCache::SamplerCreateInfo::hash() const
    {
        return CRCHash(info);
    }

    VkDescriptorSet SamplerCache::getCommonDescriptorSet()
    {
        if (m_cacheCommonDescriptor == VK_NULL_HANDLE)
        {
            initCommonDescriptorSet();
        }

        return m_cacheCommonDescriptor;
    }

    void SamplerCache::initCommonDescriptorSet()
    {
        VkDescriptorImageInfo pointClampEdgeInfo{};
        pointClampEdgeInfo.sampler = createSampler(SamplerFactory::pointClampEdge());

        VkDescriptorImageInfo pointClampBorder0000Info{};
        pointClampBorder0000Info.sampler = createSampler(SamplerFactory::pointClampBorder0000());

        VkDescriptorImageInfo pointRepeatInfo{};
        pointRepeatInfo.sampler = createSampler(SamplerFactory::pointRepeat());

        VkDescriptorImageInfo linearClampEdgeInfo{};
        linearClampEdgeInfo.sampler = createSampler(SamplerFactory::linearClampEdgeMipPoint());

        VkDescriptorImageInfo linearClampBorder0000Info{};
        linearClampBorder0000Info.sampler = createSampler(SamplerFactory::linearClampBorder0000MipPoint());


        VkDescriptorImageInfo linearRepeatInfo{};
        linearRepeatInfo.sampler = createSampler(SamplerFactory::linearRepeatMipPoint());

        VkDescriptorImageInfo linearClampBorder1111Info{};
        linearClampBorder1111Info.sampler = createSampler(SamplerFactory::linearClampBorder1111MipPoint());


        VkDescriptorImageInfo pointClampBorder1111Info{};
        pointClampBorder1111Info.sampler = createSampler(SamplerFactory::pointClampBorder1111());

        RHI::get()->descriptorFactoryBegin()
            .bindImages(0, 1, &pointClampEdgeInfo, VK_DESCRIPTOR_TYPE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
            .bindImages(1, 1, &pointClampBorder0000Info, VK_DESCRIPTOR_TYPE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
            .bindImages(2, 1, &pointRepeatInfo, VK_DESCRIPTOR_TYPE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
            .bindImages(3, 1, &linearClampEdgeInfo, VK_DESCRIPTOR_TYPE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
            .bindImages(4, 1, &linearClampBorder0000Info, VK_DESCRIPTOR_TYPE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
            .bindImages(5, 1, &linearRepeatInfo, VK_DESCRIPTOR_TYPE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
            .bindImages(6, 1, &linearClampBorder1111Info, VK_DESCRIPTOR_TYPE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
            .bindImages(7, 1, &pointClampBorder1111Info, VK_DESCRIPTOR_TYPE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
            .build(m_cacheCommonDescriptor, m_cacheCommonDescriptorSetLayout);
    }

    VkDescriptorSetLayout SamplerCache::getCommonDescriptorSetLayout()
    {
        if (m_cacheCommonDescriptorSetLayout == VK_NULL_HANDLE)
        {
            initCommonDescriptorSet();
        }
        return m_cacheCommonDescriptorSetLayout;
    }
}