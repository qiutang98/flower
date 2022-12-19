#pragma once
#include "RendererCommon.h"

namespace Flower
{
	namespace RTFormats
	{
		inline VkFormat hdrSceneColor() 
		{ 
			return VK_FORMAT_R16G16B16A16_SFLOAT; 
		}

		// GBuffer A: r8g8b8a8 unorm, .rgb store base color.
		inline VkFormat gbufferA()
		{
			return VK_FORMAT_R8G8B8A8_UNORM;
		}

		// GBuffer B: r16g16b16a16 sfloat, .rgb store worldspace normal, .a is mesh id.
		inline VkFormat gbufferB()
		{
			return VK_FORMAT_R16G16B16A16_SFLOAT;
		}

		// GBuffer S: r8g8b8a8 unorm, .r is metal, .g is roughness, .b is mesh ao.
		inline VkFormat gbufferS()
		{
			return VK_FORMAT_R8G8B8A8_UNORM;
		}

		// GBuffer V: r16g16 float, is uv space velocity.
		inline VkFormat gbufferV()
		{
			return VK_FORMAT_R16G16_SFLOAT;
		}

		inline VkFormat gbufferUpscaleReactive()
		{
			return VK_FORMAT_R8_UNORM;
		}

		inline VkFormat gbufferUpscaleTranslucencyAndComposition()
		{
			return VK_FORMAT_R8_UNORM;
		}

		inline VkFormat depth()
		{ 
			return VK_FORMAT_D32_SFLOAT; 
		}

		inline VkFormat displayOutput()
		{ 
			// R16G16B16A16 keep high precision so when draw to back buffer, still looking good.
			return VK_FORMAT_R16G16B16A16_SFLOAT;
		}

		inline VkFormat sdsmShadowMask()
		{
			return VK_FORMAT_R8_UNORM;
		}

		inline VkFormat brdfLut()
		{
			return VK_FORMAT_R16G16_SFLOAT;
		}
	}

	namespace RTUsages
	{
		inline VkImageUsageFlags hdrSceneColor() 
		{
			return
				VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
				VK_IMAGE_USAGE_STORAGE_BIT |
				VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		}

		inline VkImageUsageFlags gbuffer()
		{
			return
				VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
				VK_IMAGE_USAGE_STORAGE_BIT |
				VK_IMAGE_USAGE_SAMPLED_BIT |
				VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		}

		inline VkImageUsageFlags depth()
		{ 
			return
				VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
				VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		}

		inline VkImageUsageFlags displayOutput()
		{ 
			return 
				VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | 
				VK_IMAGE_USAGE_STORAGE_BIT | 
				VK_IMAGE_USAGE_SAMPLED_BIT; 
		}

		inline VkImageUsageFlags sdsmMask()
		{
			return VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		}

		inline VkImageUsageFlags brdfLut()
		{
			return VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		}
	}
}