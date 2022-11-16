#pragma once

#include "RHILogger.h"

namespace Flower
{
    inline uint32_t getMipLevelsCount(uint32_t texWidth, uint32_t texHeight)
    {
        return static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;
    }

    struct GPUQueuesInfo
    {
        uint32_t graphicsFamily = ~0;
        uint32_t copyFamily = ~0;
        uint32_t computeFamily = ~0;

        std::vector<VkQueue> computeQueues;
        std::vector<VkQueue> copyQueues;
        std::vector<VkQueue> graphcisQueues;
    };

    struct GPUCommandPool
    {
        VkQueue queue = VK_NULL_HANDLE;
        VkCommandPool pool = VK_NULL_HANDLE;
    };

    struct SwapchainSupportDetails
    {
        VkSurfaceCapabilitiesKHR        capabilities;
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR>   presentModes;
    };

    inline VkCommandBufferBeginInfo RHICommandbufferBeginInfo(VkCommandBufferUsageFlags flags)
    {
        VkCommandBufferBeginInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        info.pNext = nullptr;
        info.pInheritanceInfo = nullptr;
        info.flags = flags;
        return info;
    }

    inline VkImageMemoryBarrier2 RHIImageBarrier(
        VkImage image, 
        VkPipelineStageFlags2 srcStageMask, 
        VkAccessFlags2 srcAccessMask, 
        VkImageLayout oldLayout, 
        VkPipelineStageFlags2 dstStageMask, 
        VkAccessFlags2 dstAccessMask, 
        VkImageLayout newLayout, 
        VkImageAspectFlags aspectMask, 
        uint32_t baseMipLevel, 
        uint32_t levelCount)
    {
        VkImageMemoryBarrier2 result = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };

        result.srcStageMask = srcStageMask;
        result.srcAccessMask = srcAccessMask;
        result.dstStageMask = dstStageMask;
        result.dstAccessMask = dstAccessMask;
        result.oldLayout = oldLayout;
        result.newLayout = newLayout;
        result.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        result.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        result.image = image;
        result.subresourceRange.aspectMask = aspectMask;
        result.subresourceRange.baseMipLevel = baseMipLevel;
        result.subresourceRange.levelCount = levelCount;
        result.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;

        return result;
    }

    inline VkBufferMemoryBarrier2 RHIBufferBarrier(
        VkBuffer buffer, 
        VkPipelineStageFlags2 srcStageMask, 
        VkAccessFlags2 srcAccessMask, 
        VkPipelineStageFlags2 dstStageMask, 
        VkAccessFlags2 dstAccessMask)
    {
        VkBufferMemoryBarrier2 result = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2 };

        result.srcStageMask = srcStageMask;
        result.srcAccessMask = srcAccessMask;
        result.dstStageMask = dstStageMask;
        result.dstAccessMask = dstAccessMask;
        result.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        result.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        result.buffer = buffer;
        result.offset = 0;
        result.size = VK_WHOLE_SIZE;

        return result;
    }

    inline void RHIPipelineBarrier(
        VkCommandBuffer commandBuffer, 
        VkDependencyFlags dependencyFlags, 
        size_t bufferBarrierCount, 
        const VkBufferMemoryBarrier2* bufferBarriers, 
        size_t imageBarrierCount, 
        const VkImageMemoryBarrier2* imageBarriers)
    {
        VkDependencyInfo dependencyInfo = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dependencyInfo.dependencyFlags = dependencyFlags;
        dependencyInfo.bufferMemoryBarrierCount = unsigned(bufferBarrierCount);
        dependencyInfo.pBufferMemoryBarriers = bufferBarriers;
        dependencyInfo.imageMemoryBarrierCount = unsigned(imageBarrierCount);
        dependencyInfo.pImageMemoryBarriers = imageBarriers;

        vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
    }

    inline VkPipelineLayoutCreateInfo RHIPipelineLayoutCreateInfo()
    {
        VkPipelineLayoutCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        info.pNext = nullptr;
        info.flags = 0;
        info.setLayoutCount = 0;
        info.pSetLayouts = nullptr;
        info.pushConstantRangeCount = 0;
        info.pPushConstantRanges = nullptr;
        return info;
    }

    inline VkWriteDescriptorSet RHIWriteDescriptorSetBuffer(
        VkDescriptorSet set, uint32_t binding, VkDescriptorType type, const VkDescriptorBufferInfo* bufferInfo)
    {
        return VkWriteDescriptorSet
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = binding,
            .descriptorCount = 1,
            .descriptorType = type,
            .pBufferInfo = bufferInfo,
        };
    }

    inline VkDescriptorImageInfo RHIDescriptorImageInfoStorage(VkImageView view)
    {
        return VkDescriptorImageInfo
        {
            .imageView = view,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
    }

    inline VkDescriptorImageInfo RHIDescriptorImageInfoSample(VkImageView view)
    {
        return VkDescriptorImageInfo
        {
            .imageView = view,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };
    }

    inline VkWriteDescriptorSet RHIPushWriteDescriptorSetImage(uint32_t binding, VkDescriptorType type, const VkDescriptorImageInfo* imageInfo, uint32_t count = 1)
    {
        return VkWriteDescriptorSet
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = binding,
            .descriptorCount = count,
            .descriptorType = type,
            .pImageInfo = imageInfo,
        };
    }

    inline VkWriteDescriptorSet RHIPushWriteDescriptorSetBuffer(uint32_t binding, VkDescriptorType type, const VkDescriptorBufferInfo* bufferInfo, uint32_t count = 1)
    {
        return VkWriteDescriptorSet
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = binding,
            .descriptorCount = count,
            .descriptorType = type,
            .pBufferInfo = bufferInfo,
        };
    }

    class RHISubmitInfo
    {
    private:
        VkSubmitInfo submitInfo{};

    public:
        RHISubmitInfo()
        {
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            std::vector<VkPipelineStageFlags> waitStages = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
            submitInfo.pWaitDstStageMask = waitStages.data();
        }

        operator VkSubmitInfo()
        {
            return submitInfo;
        }

        VkSubmitInfo& get() { return submitInfo; }

        RHISubmitInfo& setWaitStage(VkPipelineStageFlags* waitStages)
        {
            submitInfo.pWaitDstStageMask = waitStages;
            return *this;
        }

        RHISubmitInfo& setWaitStage(std::vector<VkPipelineStageFlags>&& waitStages) = delete;

        RHISubmitInfo& setWaitSemaphore(VkSemaphore* wait, int32_t count)
        {
            submitInfo.waitSemaphoreCount = count;
            submitInfo.pWaitSemaphores = wait;
            return *this;
        }

        RHISubmitInfo& setSignalSemaphore(VkSemaphore* signal, int32_t count)
        {
            submitInfo.signalSemaphoreCount = count;
            submitInfo.pSignalSemaphores = signal;
            return *this;
        }

        RHISubmitInfo& setCommandBuffer(VkCommandBuffer* cb, int32_t count)
        {
            submitInfo.commandBufferCount = count;
            submitInfo.pCommandBuffers = cb;
            return *this;
        }

        RHISubmitInfo& clear()
        {
            submitInfo = {};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            std::vector<VkPipelineStageFlags> waitStages = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
            submitInfo.pWaitDstStageMask = waitStages.data();
            return *this;
        }
    };

    // Build default opaque blend state.
    inline VkPipelineColorBlendAttachmentState RHIColorBlendAttachmentOpauqeState()
    {
        VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
        colorBlendAttachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;
        return colorBlendAttachment;
    }

    inline VkPipelineShaderStageCreateInfo RHIPipelineShaderStageCreateInfo(VkShaderStageFlagBits stage, VkShaderModule shaderModule)
    {
        VkPipelineShaderStageCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        info.pNext = nullptr;
        info.stage = stage;
        info.module = shaderModule;
        info.pName = "main";
        return info;
    }

    inline VkPipelineVertexInputStateCreateInfo RHIVertexInputStateCreateInfo()
    {
        VkPipelineVertexInputStateCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        info.pNext = nullptr;
        info.vertexBindingDescriptionCount = 0;
        info.vertexAttributeDescriptionCount = 0;
        return info;
    }

    inline VkViewport RHIViewportDefault()
    {
        return VkViewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = 64.0f,
            .height = 64.0f,
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
    }

    inline VkRect2D RHIScissorDefault()
    {
        return VkRect2D{
            .offset = {0, 0},
            .extent = {64, 64},
        };
    }

    inline VkBufferMemoryBarrier RHIBufferMemoryBarrier(VkBuffer buffer, uint32_t size, VkAccessFlags srcAccess, VkAccessFlags dstAccess)
    {
        VkBufferMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;

        barrier.buffer = buffer;
        barrier.size = size;
        barrier.srcAccessMask = srcAccess;
        barrier.dstAccessMask = dstAccess;
        return barrier;
    }

    inline VkImageSubresourceRange RHIDefaultImageSubresourceRange(VkImageAspectFlags aspectMask)
    {
        return VkImageSubresourceRange{
            .aspectMask = aspectMask,
            .baseMipLevel = 0,
            .levelCount = VK_REMAINING_MIP_LEVELS,
            .baseArrayLayer = 0,
            .layerCount = VK_REMAINING_ARRAY_LAYERS
        };
    }

    inline VkRenderingAttachmentInfo RHIRenderingAttachmentInfo(
        VkImageView view,
        VkImageLayout layout,
        VkAttachmentLoadOp loadOp,
        VkAttachmentStoreOp storeOp,
        VkClearValue clearValue)
    {
        return VkRenderingAttachmentInfo
        {
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = view,
            .imageLayout = layout,
            .loadOp = loadOp,
            .storeOp = storeOp,
            .clearValue = clearValue,
        };
    }

    inline VkImageMemoryBarrier RHIImageMemoryBarrier(
        VkImage image, 
        VkImageLayout srcLayout, 
        VkImageLayout dstLayout,
        VkAccessFlags srcAccess, 
        VkAccessFlags dstAccess,
        VkImageSubresourceRange subResRange)
    {
        VkImageMemoryBarrier imageBarrier
        {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = srcAccess,
            .dstAccessMask = dstAccess,
            .oldLayout = srcLayout,
            .newLayout = dstLayout,
            .image = image,
            .subresourceRange = subResRange
        };

        return imageBarrier;
    }


    inline VkPipelineViewportStateCreateInfo RHIDefaultViewportState()
    {
        static VkViewport defaultViewport = RHIViewportDefault();
        static VkRect2D defaultScissors = RHIScissorDefault();

        VkPipelineViewportStateCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        info.viewportCount = 1;
        info.pViewports = &defaultViewport;
        info.scissorCount = 1;
        info.pScissors = &defaultScissors;

        return info;
    }

    inline const VkPipelineDynamicStateCreateInfo& RHIDefaultDynamicStateCreateInfo()
    {
        static const std::vector<VkDynamicState> dynamicStates =
        {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR,
            VK_DYNAMIC_STATE_DEPTH_BIAS
        };
        static const VkPipelineDynamicStateCreateInfo pipelineDynamicState
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .dynamicStateCount = (uint32_t)dynamicStates.size(),
            .pDynamicStates = dynamicStates.data(),
        };

        return pipelineDynamicState;
    }

    inline VkPipelineRasterizationStateCreateInfo RHIRasterizationStateCreateInfo(VkPolygonMode polygonMode)
    {
        VkPipelineRasterizationStateCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        info.pNext = nullptr;
        info.depthClampEnable = VK_FALSE;
        info.rasterizerDiscardEnable = VK_FALSE;
        info.polygonMode = polygonMode;
        info.lineWidth = 1.0f;
        info.cullMode = VK_CULL_MODE_NONE;
        info.frontFace = VK_FRONT_FACE_CLOCKWISE;
        info.depthBiasEnable = VK_FALSE;
        info.depthBiasConstantFactor = 0.0f;
        info.depthBiasClamp = 0.0f;
        info.depthBiasSlopeFactor = 0.0f;
        return info;
    }

    inline VkPipelineInputAssemblyStateCreateInfo RHIInputAssemblyCreateInfo(VkPrimitiveTopology topology)
    {
        VkPipelineInputAssemblyStateCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        info.pNext = nullptr;
        info.topology = topology;
        info.primitiveRestartEnable = VK_FALSE;
        return info;
    }

    // Default no multi sample raster state.
    inline VkPipelineMultisampleStateCreateInfo RHIMultisamplingStateCreateInfo()
    {
        VkPipelineMultisampleStateCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        info.pNext = nullptr;
        info.sampleShadingEnable = VK_FALSE;
        info.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        info.minSampleShading = 1.0f;
        info.pSampleMask = nullptr;
        info.alphaToCoverageEnable = VK_FALSE;
        info.alphaToOneEnable = VK_FALSE;
        return info;
    }

    inline VkPipelineDepthStencilStateCreateInfo RHIDepthStencilCreateInfo(
        bool bDepthTest, 
        bool bDepthWrite, 
        VkCompareOp compareOp)
    {
        VkPipelineDepthStencilStateCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        info.pNext = nullptr;

        info.depthTestEnable  = bDepthTest  ? VK_TRUE : VK_FALSE;
        info.depthWriteEnable = bDepthWrite ? VK_TRUE : VK_FALSE;
        info.depthCompareOp   = bDepthTest  ? compareOp : VK_COMPARE_OP_ALWAYS;

        info.depthBoundsTestEnable = VK_FALSE;
        info.stencilTestEnable     = VK_FALSE;

        info.minDepthBounds = 0.0f; // Optional
        info.maxDepthBounds = 1.0f; // Optional

        return info;
    }

    namespace SamplerFactory
    {
        inline VkSamplerCreateInfo buildBasic()
        {
            VkSamplerCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;

            info.magFilter = VK_FILTER_NEAREST;
            info.minFilter = VK_FILTER_NEAREST;
            info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;

            info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

            info.minLod = -10000.0f;
            info.maxLod = 10000.0f;
            info.mipLodBias = 0.0f;

            info.maxAnisotropy = 1.0f;
            info.anisotropyEnable = VK_FALSE;

            info.compareEnable = VK_FALSE;
            info.unnormalizedCoordinates = VK_FALSE;

            return info;
        }

        inline VkSamplerCreateInfo pointClampEdge()
        {
            VkSamplerCreateInfo info = buildBasic();

            info.magFilter = VK_FILTER_NEAREST;
            info.minFilter = VK_FILTER_NEAREST;
            info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;

            info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

            return info;
        }

        inline VkSamplerCreateInfo pointClampBorder0000()
        {
            VkSamplerCreateInfo info = buildBasic();

            info.magFilter = VK_FILTER_NEAREST;
            info.minFilter = VK_FILTER_NEAREST;
            info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;

            info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
            info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
            info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
            info.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;

            return info;
        }

        inline VkSamplerCreateInfo pointClampBorder1111()
        {
            VkSamplerCreateInfo info = buildBasic();

            info.magFilter = VK_FILTER_NEAREST;
            info.minFilter = VK_FILTER_NEAREST;
            info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;

            info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
            info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
            info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
            info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;

            return info;
        }

        inline VkSamplerCreateInfo pointRepeat()
        {
            VkSamplerCreateInfo info = buildBasic();

            info.magFilter = VK_FILTER_NEAREST;
            info.minFilter = VK_FILTER_NEAREST;
            info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;

            info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;

            return info;
        }

        // Linear clamp, mipmap also linear.
        inline VkSamplerCreateInfo linearClampEdge()
        {
            VkSamplerCreateInfo info = buildBasic();

            info.magFilter = VK_FILTER_LINEAR;
            info.minFilter = VK_FILTER_LINEAR;
            info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

            info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

            return info;
        }

        inline VkSamplerCreateInfo linearClampEdgeMipPoint()
        {
            VkSamplerCreateInfo info = buildBasic();

            info.magFilter = VK_FILTER_LINEAR;
            info.minFilter = VK_FILTER_LINEAR;
            info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;

            info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

            return info;
        }

        inline VkSamplerCreateInfo linearClampBorder0000MipPoint()
        {
            VkSamplerCreateInfo info = buildBasic();

            info.magFilter = VK_FILTER_LINEAR;
            info.minFilter = VK_FILTER_LINEAR;
            info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;

            info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
            info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
            info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
            info.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;

            return info;
        }

        inline VkSamplerCreateInfo linearClampBorder1111MipPoint()
        {
            VkSamplerCreateInfo info = buildBasic();

            info.magFilter = VK_FILTER_LINEAR;
            info.minFilter = VK_FILTER_LINEAR;
            info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;

            info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
            info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
            info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
            info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;

            return info;
        }

        inline VkSamplerCreateInfo linearRepeatMipPoint()
        {
            VkSamplerCreateInfo info = buildBasic();

            info.magFilter = VK_FILTER_LINEAR;
            info.minFilter = VK_FILTER_LINEAR;
            info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;

            info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;

            return info;
        }
    }
    
    inline VkImageSubresourceRange buildBasicImageSubresource()
    {
        VkImageSubresourceRange range{};

        range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

        range.baseMipLevel = 0;
        range.levelCount = VK_REMAINING_MIP_LEVELS;

        range.baseArrayLayer = 0;
        range.layerCount = VK_REMAINING_ARRAY_LAYERS;

        return range;
    }

    inline VkImageSubresourceRange buildBasicImageSubresourceCube()
    {
        VkImageSubresourceRange range{};

        range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

        range.baseMipLevel = 0;
        range.levelCount = VK_REMAINING_MIP_LEVELS;

        range.baseArrayLayer = 0;
        range.layerCount = 6;
        return range;
    }
    
    enum class EVMAUsageFlags
    {
        GPUOnly,
        StageCopyForUpload,
        Readback,
    };
}

