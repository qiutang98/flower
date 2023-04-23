#pragma once

#include <vulkan/vulkan.h>
#include <util/util.h>

namespace engine
{
    class VulkanContext;

    class DescriptorAllocator
    {
        friend class DescriptorFactory;
    public:
        struct PoolSizes
        {
            std::vector<std::pair<VkDescriptorType, float>> sizes =
            {
                { VK_DESCRIPTOR_TYPE_SAMPLER,                .5f },
                { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4.f },
                { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,          4.f },
                { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1.f },
                { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,   1.f },
                { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,   1.f },
                { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         2.f },
                { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         2.f },
                { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1.f },
                { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1.f },
                { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,       .5f }
            };
        };

    private:
        const VulkanContext* m_context = nullptr;

        VkDescriptorPool m_currentPool = VK_NULL_HANDLE;
        PoolSizes m_descriptorSizes;
        std::vector<VkDescriptorPool> m_usedPools;
        std::vector<VkDescriptorPool> m_freePools;

        VkDescriptorPool requestPool();
    public:
        // reset all using pool to free.
        void resetPools();

        // allocate descriptor, maybe fail.
        [[nodiscard]] bool allocate(VkDescriptorSet* set, VkDescriptorSetLayout layout);

        void init(const VulkanContext* context);
        void release();

        const VulkanContext* getContext() const { return m_context; }
    };

    class DescriptorLayoutCache
    {
    public:
        struct DescriptorLayoutInfo
        {
            std::vector<VkDescriptorSetLayoutBinding> bindings;
            bool operator==(const DescriptorLayoutInfo& other) const;
            uint64_t hash() const;
        };

    private:
        struct DescriptorLayoutHash
        {
            uint64_t operator()(const DescriptorLayoutInfo& k) const
            {
                return k.hash();
            }
        };

        typedef std::unordered_map<DescriptorLayoutInfo, VkDescriptorSetLayout, DescriptorLayoutHash> LayoutCache;
        LayoutCache m_layoutCache;

        const VulkanContext* m_context;

    public:
        void init(const VulkanContext* context);
        void release();

        VkDescriptorSetLayout createDescriptorLayout(VkDescriptorSetLayoutCreateInfo* info);
    };

    class DescriptorFactory
    {
    public:
        // start building.
        static DescriptorFactory begin(DescriptorLayoutCache* layoutCache, DescriptorAllocator* allocator);

        // use for buffers.
        DescriptorFactory& bindBuffers(uint32_t binding, uint32_t count, VkDescriptorBufferInfo* bufferInfo, VkDescriptorType type, VkShaderStageFlags stageFlags);

        // use for textures.
        DescriptorFactory& bindImages(uint32_t binding, uint32_t count, VkDescriptorImageInfo*, VkDescriptorType type, VkShaderStageFlags stageFlags);

        bool build(VkDescriptorSet& set, VkDescriptorSetLayout& layout);
        bool build(VkDescriptorSet& set);

        // No info bind, need update descriptor set before rendering.
        // use vkCmdPushDescriptorKHR to update.
        DescriptorFactory& bindNoInfo(
            VkDescriptorType type,
            VkShaderStageFlags stageFlags,
            uint32_t binding,
            uint32_t count = 1);

        void buildNoInfo(VkDescriptorSetLayout& layout, VkDescriptorSet& set);
        void buildNoInfo(VkDescriptorSet& set);
        void buildNoInfoPush(VkDescriptorSetLayout& layout);

    private:
        struct DescriptorWriteContainer
        {
            VkDescriptorImageInfo* imgInfo;
            VkDescriptorBufferInfo* bufInfo;
            uint32_t binding;
            VkDescriptorType type;
            uint32_t count;
            bool isImg = false;
        };

        std::vector<DescriptorWriteContainer> m_descriptorWriteBufInfos{ };
        std::vector<VkDescriptorSetLayoutBinding> m_bindings;

        DescriptorLayoutCache* m_cache;
        DescriptorAllocator* m_allocator;
    };

}