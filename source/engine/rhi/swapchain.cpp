#include "rhi.h"

namespace engine
{
	VkPresentModeKHR Swapchain::chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes)
	{
		// Use mailbox if can use.
		for (const auto& availablePresentMode : availablePresentModes)
		{
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
			{
				return availablePresentMode;
			}
		}

		// Other case try immediate mode.
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
			glfwGetFramebufferSize(m_context->getWindow(), &width, &height);

			VkExtent2D actualExtent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height) };
			actualExtent.width  = math::clamp(actualExtent.width,  capabilities.minImageExtent.width,  capabilities.maxImageExtent.width);
			actualExtent.height = math::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

			return actualExtent;
		}
	}

	struct SwapchainSupportDetails
	{
		VkSurfaceCapabilitiesKHR        capabilities;
		std::vector<VkSurfaceFormatKHR> formats;
		std::vector<VkPresentModeKHR>   presentModes;
	};

	static inline SwapchainSupportDetails querySwapchainSupportDetail(const VulkanContext* context)
	{
		SwapchainSupportDetails details;
		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(context->getGPU(), context->getSurface(), &details.capabilities);

		uint32_t formatCount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(context->getGPU(), context->getSurface(), &formatCount, nullptr);
		if (formatCount != 0)
		{
			details.formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(context->getGPU(), context->getSurface(), &formatCount, details.formats.data());
		}

		uint32_t presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(context->getGPU(), context->getSurface(), &presentModeCount, nullptr);
		if (presentModeCount != 0)
		{
			details.presentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(context->getGPU(), context->getSurface(), &presentModeCount, details.presentModes.data());
		}

		return details;
	}

	struct BackBufferFormatState
	{
		VkSurfaceFormatKHR format = {};
		bool bSupport = false;
		bool bSupportAlphaBlend = false;
	};

	const std::vector<BackBufferFormatState>& getBackBufferFormatStateStatic(const VulkanContext* inContext)
	{
		auto getBackBufferFormatState = [](const VulkanContext* context)
		{
			size_t maxQueryNum = size_t(EBackBufferFormat::Max);
			std::vector<BackBufferFormatState> result(maxQueryNum);

			auto swapchainSupport = querySwapchainSupportDetail(context);

			for (const auto& availableFormat : swapchainSupport.formats)
			{
				// SRGB non-linear format check.
				if ((availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
				 && (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM))
				{
					result.at(size_t(EBackBufferFormat::SRGB_NonLinear)) = { .format = availableFormat, .bSupport = true, .bSupportAlphaBlend = true };
				}

				// Hdr10 st2084 format check, alpha 2 bit not support alpha blend.
				if ((availableFormat.colorSpace == VK_COLOR_SPACE_HDR10_ST2084_EXT)
				&& ((availableFormat.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32) || (availableFormat.format == VK_FORMAT_A2R10G10B10_UNORM_PACK32)))
				{
					result.at(size_t(EBackBufferFormat::HDR10_2084)) = { .format = availableFormat, .bSupport = true, .bSupportAlphaBlend = false };
				}
			}

			return result;
		};

		// Init backbuffer state.
		static std::vector<BackBufferFormatState> backbufferFormatSate = getBackBufferFormatState(inContext);
		return backbufferFormatSate;
	}

	bool Swapchain::isBackBufferFormatSupport(const VulkanContext* context)
	{
		return getBackBufferFormatStateStatic(context).at(size_t(context->getBackbufferFormatType())).bSupport;
	}

	bool Swapchain::isBackBufferSupportAlphaBlend(const VulkanContext* context)
	{
		return getBackBufferFormatStateStatic(context).at(size_t(context->getBackbufferFormatType())).bSupportAlphaBlend;
	}

	void Swapchain::init(const VulkanContext* context)
	{
		ASSERT(context->getEngine()->isWindowApp(), "Swapchain only init for windows app.");

		// Init context.
		m_context = context;
		auto swapchainSupport = querySwapchainSupportDetail(m_context);

		m_surfaceFormat = chooseSwapSurfaceFormat();
		m_presentMode = chooseSwapPresentMode(swapchainSupport.presentModes);

		VkExtent2D extent = chooseSwapchainExtent(swapchainSupport.capabilities);

		// We use min image count plus one as swapchain backbuffer count.
		// Commonly we can get 3 as result.
		uint32_t imageCount = swapchainSupport.capabilities.minImageCount + 1;
		if (swapchainSupport.capabilities.maxImageCount > 0 && imageCount > swapchainSupport.capabilities.maxImageCount)
		{
			imageCount = swapchainSupport.capabilities.maxImageCount;
		}

		VkSwapchainCreateInfoKHR createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		createInfo.surface = m_context->getSurface();
		createInfo.minImageCount = imageCount;
		createInfo.imageFormat = m_surfaceFormat.format;
		createInfo.imageColorSpace = m_surfaceFormat.colorSpace;
		createInfo.imageExtent = extent;
		createInfo.imageArrayLayers = 1;
		createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

		// We use graphics family queue to draw and swap.
		createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		createInfo.queueFamilyIndexCount = 0;
		createInfo.pQueueFamilyIndices = nullptr;
		createInfo.preTransform = swapchainSupport.capabilities.currentTransform;
		createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		createInfo.presentMode = m_presentMode;
		createInfo.clipped = VK_TRUE;

		RHICheck(vkCreateSwapchainKHR(m_context->getDevice(), &createInfo, nullptr, &m_swapchain));

		// Get swapchain images.
		vkGetSwapchainImagesKHR(m_context->getDevice(), m_swapchain, &imageCount, nullptr);
		m_swapchainImages.resize(imageCount);
		vkGetSwapchainImagesKHR(m_context->getDevice(), m_swapchain, &imageCount, m_swapchainImages.data());

		// Assign format and extent result.
		m_swapchainImageFormat = m_surfaceFormat.format;
		m_swapchainExtent = extent;

		// create image views need for swapchain.
		m_swapchainImageViews.resize(m_swapchainImages.size());
		for (size_t i = 0; i < m_swapchainImages.size(); i++)
		{
			VkImageViewCreateInfo createInfo{};
			createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
			createInfo.image  = m_swapchainImages[i];
			createInfo.format = m_swapchainImageFormat;
			createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			createInfo.subresourceRange.baseMipLevel   = 0;
			createInfo.subresourceRange.levelCount     = 1;
			createInfo.subresourceRange.baseArrayLayer = 0;
			createInfo.subresourceRange.layerCount     = 1;

			// Create swapchain image view.
			RHICheck(vkCreateImageView(m_context->getDevice(), &createInfo, nullptr, &m_swapchainImageViews[i]));
		}

		LOG_RHI_TRACE("Create vulkan swapChain succeed, backBuffer count is {0}. ", m_swapchainImageViews.size());
	}

	void Swapchain::rebuild()
	{
		release();
		init(m_context);
	}

	void Swapchain::release()
	{
		for (auto imageView : m_swapchainImageViews)
		{
			vkDestroyImageView(m_context->getDevice(), imageView, nullptr);
		}
		m_swapchainImageViews.resize(0);
		vkDestroySwapchainKHR(m_context->getDevice(), m_swapchain, nullptr);
	}

	VkSurfaceFormatKHR Swapchain::chooseSwapSurfaceFormat()
	{
		return getBackBufferFormatStateStatic(m_context).at(size_t(m_context->getBackbufferFormatType())).format;
	}

	void VulkanContext::recreateSwapChain()
	{
		ASSERT(getEngine()->isWindowApp(), "Swapchain recreate only used for windows app.");

		vkDeviceWaitIdle(m_device);

		static int width = 0, height = 0;
		glfwGetFramebufferSize(m_window, &width, &height);

		// just return if swapchain width or height is 0.
		if (width == 0 || height == 0)
		{
			m_presentContext.bSwapchainChange = true;
			return;
		}

		// Broadcast swapchain before recreate.
		onBeforeSwapchainRecreate.broadcast();

		destroyPresentContext();
		{
			m_swapchain.release();
			m_swapchain.init(this);
		}
		initPresentContext();

		m_presentContext.imagesInFlight.resize(m_swapchain.getBackbufferCount(), VK_NULL_HANDLE);

		// Broadcast swapchain after recreate.
		onAfterSwapchainRecreate.broadcast();
	}

	uint32_t VulkanContext::acquireNextPresentImage()
	{
		ASSERT(getEngine()->isWindowApp(), "Acquire image only used for windows app.");

		auto swapchainRebuildState = [this]()
		{
			auto& ct = m_swapchainRebuildContext;
			glfwGetWindowSize(m_window, &ct.currentWidth, &ct.currentHeight);

			if (ct.currentWidth != ct.lastWidth
				|| ct.currentHeight != ct.lastHeight)
			{
				ct.lastWidth = ct.currentWidth;
				ct.lastHeight = ct.currentHeight;
				return true;
			}

			return false;
		};

		m_presentContext.bSwapchainChange |= swapchainRebuildState();

		vkWaitForFences(m_device, 1, &m_presentContext.inFlightFences[m_presentContext.currentFrame], VK_TRUE, UINT64_MAX);

		VkResult result = vkAcquireNextImageKHR(
			m_device,
			m_swapchain.get(),
			UINT64_MAX,
			m_presentContext.semaphoresImageAvailable[m_presentContext.currentFrame],
			VK_NULL_HANDLE,
			&m_presentContext.imageIndex
		);

		if (result == VK_ERROR_OUT_OF_DATE_KHR)
		{
			recreateSwapChain();
		}
		else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
		{
			LOG_RHI_FATAL("Fail to requeset present image.");
		}

		if (m_presentContext.imagesInFlight[m_presentContext.imageIndex] != VK_NULL_HANDLE)
		{
			vkWaitForFences(m_device, 1, &m_presentContext.imagesInFlight[m_presentContext.imageIndex], VK_TRUE, UINT64_MAX);
		}

		m_presentContext.imagesInFlight[m_presentContext.imageIndex] = m_presentContext.inFlightFences[m_presentContext.currentFrame];

		return m_presentContext.imageIndex;
	}

	void VulkanContext::present()
	{
		ASSERT(getEngine()->isWindowApp(), "Present only call for windows app.");

		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

		VkSemaphore signalSemaphores[] = { m_presentContext.semaphoresRenderFinished[m_presentContext.currentFrame] };
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = signalSemaphores;

		VkSwapchainKHR swapchains[] = { m_swapchain.get() };
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapchains;
		presentInfo.pImageIndices = &m_presentContext.imageIndex;

		auto result = vkQueuePresentKHR(m_majorGraphicsPool.queue, &presentInfo);
		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || m_presentContext.bSwapchainChange)
		{
			m_presentContext.bSwapchainChange = false;
			recreateSwapChain();
		}
		else if (result != VK_SUCCESS)
		{
			LOG_RHI_FATAL("Fail to present image.");
		}

		// if swapchain rebuild and on minimized, still add frame.
		m_presentContext.currentFrame = (m_presentContext.currentFrame + 1) % m_swapchain.getBackbufferCount();
	}

	void VulkanContext::submit(uint32_t count, VkSubmitInfo* infos)
	{
		RHICheck(vkQueueSubmit(m_majorGraphicsPool.queue, count, infos, m_presentContext.inFlightFences[m_presentContext.currentFrame]));
	}

	void VulkanContext::submit(uint32_t count, VkSubmitInfo* infos, VkFence fence)
	{
		RHICheck(vkQueueSubmit(m_majorGraphicsPool.queue, count, infos, fence));
	}

	void VulkanContext::submit(uint32_t count, const RHISubmitInfo* infoRHI, VkFence fence)
	{
		VkSubmitInfo info = infoRHI->get();
		RHICheck(vkQueueSubmit(m_majorGraphicsPool.queue, count, &info, fence));
	}

	void VulkanContext::submitNoFence(uint32_t count, VkSubmitInfo* infos)
	{
		RHICheck(vkQueueSubmit(m_majorGraphicsPool.queue, count, infos, nullptr));
	}

	void VulkanContext::resetFence()
	{
		RHICheck(vkResetFences(m_device, 1, &m_presentContext.inFlightFences[m_presentContext.currentFrame]));
	}

	void VulkanContext::initPresentContext()
	{
		ASSERT(getEngine()->isWindowApp(), "PresentContext only init for windows app.");

		CHECK(getBackBufferCount() > 0);

		auto& pct = m_presentContext;

		pct.semaphoresImageAvailable.resize(getBackBufferCount());
		pct.semaphoresRenderFinished.resize(getBackBufferCount());

		pct.inFlightFences.resize(getBackBufferCount());
		pct.imagesInFlight.resize(getBackBufferCount());
		for (auto& fence : pct.imagesInFlight)
		{
			fence = VK_NULL_HANDLE;
		}

		VkSemaphoreCreateInfo semaphoreInfo{};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		for (size_t i = 0; i < getBackBufferCount(); i++)
		{
			RHICheck(vkCreateSemaphore(getDevice(), &semaphoreInfo, nullptr, &pct.semaphoresImageAvailable[i]));
			RHICheck(vkCreateSemaphore(getDevice(), &semaphoreInfo, nullptr, &pct.semaphoresRenderFinished[i]));
			RHICheck(vkCreateFence(getDevice(), &fenceInfo, nullptr, &pct.inFlightFences[i]));
		}

	}

	void VulkanContext::destroyPresentContext()
	{
		ASSERT(getEngine()->isWindowApp(), "PresentContext only destroy for windows app.");

		auto& pct = m_presentContext;

		for (size_t i = 0; i < getBackBufferCount(); i++)
		{
			vkDestroySemaphore(getDevice(), pct.semaphoresImageAvailable[i], nullptr);
			vkDestroySemaphore(getDevice(), pct.semaphoresRenderFinished[i], nullptr);
			vkDestroyFence(getDevice(), pct.inFlightFences[i], nullptr);
		}

	}
}