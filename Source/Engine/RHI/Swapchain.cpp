#include "Pch.h"
#include "RHI.h"
#include "SwapChain.h"

namespace Flower
{
	VkSurfaceFormatKHR Swapchain::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
	{
		switch (RHI::eDisplayMode)
		{
		case RHI::DisplayMode::DISPLAYMODE_HDR10_2084:
		{
			for (const auto& availableFormat : availableFormats)
			{
				if (availableFormat.colorSpace == VK_COLOR_SPACE_HDR10_ST2084_EXT)
				{
					if (availableFormat.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32)
					{
						return availableFormat;
					}
				}
			}
		}
		break;
		case RHI::DisplayMode::DISPLAYMODE_HDR10_SCRGB:
		{
			CHECK_ENTRY();
		}
		break;
		case RHI::DisplayMode::DISPLAYMODE_SDR:
		{
			for (const auto& availableFormat : availableFormats)
			{
				if (availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
				{
					if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM)
					{
						return availableFormat;
					}
				}
			}
		}
		default:
			break;
		}

		LOG_WARN("Current using non srgb unorm format back buffer. may cause some problem!");
		return availableFormats[0];
	}

	VkPresentModeKHR Swapchain::chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes)
	{
		for (const auto& availablePresentMode : availablePresentModes)
		{
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
			{
				return availablePresentMode;
			}
		}

		for (const auto& availablePresentMode : availablePresentModes)
		{
			if (availablePresentMode == VK_PRESENT_MODE_IMMEDIATE_KHR)
			{
				return availablePresentMode;
			}
		}

		return VK_PRESENT_MODE_FIFO_KHR;
	}

	VkExtent2D Swapchain::chooseSwapchainExtent(const VkSurfaceCapabilitiesKHR& capabilities)
	{
		if (capabilities.currentExtent.width != UINT32_MAX)
		{
			return capabilities.currentExtent;
		}
		else
		{
			int width, height;
			glfwGetFramebufferSize(RHI::get()->getWindow(), &width, &height);

			VkExtent2D actualExtent = {
				static_cast<uint32_t>(width),
				static_cast<uint32_t>(height)
			};

			actualExtent.width = std::max(
				capabilities.minImageExtent.width, 
				std::min(capabilities.maxImageExtent.width, actualExtent.width));

			actualExtent.height = std::max(
				capabilities.minImageExtent.height, 
				std::min(capabilities.maxImageExtent.height, actualExtent.height));

			return actualExtent;
		}
	}

	void Swapchain::init()
	{
		auto swapchain_support = RHI::get()->querySwapchainSupportDetail();

		VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapchain_support.formats);
		VkPresentModeKHR presentMode = chooseSwapPresentMode(swapchain_support.presentModes);
		VkExtent2D extent = chooseSwapchainExtent(swapchain_support.capabilities);
		uint32_t imageCount = swapchain_support.capabilities.minImageCount + 1;
		if (swapchain_support.capabilities.maxImageCount > 0 && imageCount > swapchain_support.capabilities.maxImageCount)
		{
			imageCount = swapchain_support.capabilities.maxImageCount;
		}

		VkSwapchainCreateInfoKHR createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		createInfo.surface = RHI::get()->getSurface();
		createInfo.minImageCount = imageCount;
		createInfo.imageFormat = surfaceFormat.format;
		createInfo.imageColorSpace = surfaceFormat.colorSpace;
		createInfo.imageExtent = extent;
		createInfo.imageArrayLayers = 1;
		createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

		// We use graphics family queue to draw and swap.
		createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		createInfo.queueFamilyIndexCount = 0;
		createInfo.pQueueFamilyIndices = nullptr;

		createInfo.preTransform = swapchain_support.capabilities.currentTransform;
		createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		createInfo.presentMode = presentMode;
		createInfo.clipped = VK_TRUE;

		RHICheck(vkCreateSwapchainKHR(RHI::Device, &createInfo, nullptr, &m_swapchain));

		// allocate images.
		vkGetSwapchainImagesKHR(RHI::Device, m_swapchain, &imageCount, nullptr);
		m_swapchainImages.resize(imageCount);
		vkGetSwapchainImagesKHR(RHI::Device, m_swapchain, &imageCount, m_swapchainImages.data());

		m_swapchainImageFormat = surfaceFormat.format;
		m_swapchainExtent = extent;

		// create image views need for swapchain.
		m_swapchainImageViews.resize(m_swapchainImages.size());
		for (size_t i = 0; i < m_swapchainImages.size(); i++)
		{
			VkImageViewCreateInfo createInfo{};
			createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			createInfo.image = m_swapchainImages[i];
			createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
			createInfo.format = m_swapchainImageFormat;
			createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			createInfo.subresourceRange.baseMipLevel = 0;
			createInfo.subresourceRange.levelCount = 1;
			createInfo.subresourceRange.baseArrayLayer = 0;
			createInfo.subresourceRange.layerCount = 1;

			RHICheck(vkCreateImageView(RHI::Device, &createInfo, nullptr, &m_swapchainImageViews[i]));
		}

		LOG_RHI_TRACE("Create vulkan swapChain succeed, backBuffer count is {0}. ", m_swapchainImageViews.size());
	}

	void Swapchain::rebuild()
	{
		release();
		init();
	}

	void Swapchain::release()
	{
		for (auto imageView : m_swapchainImageViews)
		{
			vkDestroyImageView(RHI::Device, imageView, nullptr);
		}
		m_swapchainImageViews.resize(0);
		vkDestroySwapchainKHR(RHI::Device, m_swapchain, nullptr);
	}
}