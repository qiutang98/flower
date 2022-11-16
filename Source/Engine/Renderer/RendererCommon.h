#pragma once
#include "../Core/Core.h"
#include "../RHI/RHI.h"
#include "../Engine.h"

namespace Flower
{
	class Renderer;

	constexpr size_t GMinRenderDim = 64;
	constexpr size_t GMaxRenderDim = 3840;
	constexpr size_t GBackBufferCount = 3;

	constexpr size_t GLazyDestroyTime = GBackBufferCount + 1;

	inline uint32_t getGroupCount(uint32_t threadCount, uint32_t localSize)
	{
		return (threadCount + localSize - 1) / localSize;
	}

    inline void RHISafeRelease(VkPipeline& pipeline)
    {
        if (pipeline != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(RHI::Device, pipeline, nullptr);
            pipeline = VK_NULL_HANDLE;
        }
    }

    inline void RHISafeRelease(VkPipelineLayout& pipelineLayout)
    {
        if (pipelineLayout != VK_NULL_HANDLE)
        {
            vkDestroyPipelineLayout(RHI::Device, pipelineLayout, nullptr);
            pipelineLayout = VK_NULL_HANDLE;
        }
    }

	inline VkDescriptorSetLayout GetLayoutStatic(VkDescriptorType type, uint32_t descriptorCount = 1)
	{
		DescriptorLayoutCache& cache = RHI::get()->getDescriptorLayoutCache();

		VkDescriptorSetLayoutBinding binding{};
		binding.binding = 0;
		binding.descriptorType = type;
		binding.descriptorCount = descriptorCount;
		binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutCreateInfo info{};
		info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		info.bindingCount = 1;
		info.pBindings = &binding;

		return cache.createDescriptorLayout(&info);
	}

	namespace colorspace
	{
		inline glm::vec3 srgb_2_rec2020(const glm::vec3& src)
		{
			static const glm::mat3 sRGB_2_XYZ = glm::mat3(
				0.4124564, 0.2126729, 0.0193339,
				0.3575761, 0.7151522, 0.1191920,
				0.1804375, 0.0721750, 0.9503041
			);
			static const  glm::mat3 XYZ_2_sRGB = glm::mat3(
				3.2404542, -0.9692660, 0.0556434,
				-1.5371385, 1.8760108, -0.2040259,
				-0.4985314, 0.0415560, 1.0572252
			);

			// REC 2020 primaries
			static const  glm::mat3 XYZ_2_Rec2020 = glm::mat3(
				1.7166084, -0.6666829, 0.0176422,
				-0.3556621, 1.6164776, -0.0427763,
				-0.2533601, 0.0157685, 0.94222867
			);

			static const  glm::mat3 Rec2020_2_XYZ = glm::mat3(
				0.6369736, 0.2627066, 0.0000000,
				0.1446172, 0.6779996, 0.0280728,
				0.1688585, 0.0592938, 1.0608437
			);

			static const glm::mat3 sRGB_2_Rec2020 = XYZ_2_Rec2020 * sRGB_2_XYZ;
			static const glm::mat3 Rec2020_2_sRGB = XYZ_2_sRGB * Rec2020_2_XYZ;
		
			return sRGB_2_Rec2020 * src;
		}
	}
}