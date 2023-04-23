#pragma once

#include <vulkan/vulkan.h>
#include <vector>

#include "rhi_misc.h"

namespace engine
{
	class VulkanContext;

	class Swapchain
	{
	private:
		const VulkanContext* m_context;

		// Swapchain images.
		std::vector<VkImage> m_swapchainImages = {};
		
		// Swapchain image views.
		std::vector<VkImageView> m_swapchainImageViews = {};

		// Swapchain format.
		VkFormat m_swapchainImageFormat = {};
		
		// Extent of swapchain.
		VkExtent2D m_swapchainExtent = {};

		// Swapchain handle.
		VkSwapchainKHR m_swapchain = {};

		// Current swapchain using surface format.
		VkSurfaceFormatKHR m_surfaceFormat = {};

		// Current swapchain present mode.
		VkPresentModeKHR m_presentMode = {};

	private:
		VkSurfaceFormatKHR chooseSwapSurfaceFormat();
		VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
		VkExtent2D chooseSwapchainExtent(const VkSurfaceCapabilitiesKHR& capabilities);

	public:
		inline const auto& get() const { return m_swapchain; }
		inline const auto& getImages() const { return m_swapchainImages; }
		inline const auto& getImageViews() const { return m_swapchainImageViews; }
		inline const auto& getExtent() const { return m_swapchainExtent; }
		inline const auto& getImageFormat() const { return m_swapchainImageFormat; }
		inline const auto& getSurfaceFormat() const { return m_surfaceFormat; }
		inline const auto& getSwapchainPresentMode() const { return m_presentMode; }

		inline const uint32_t getBackbufferCount() const { return (uint32_t)m_swapchainImageViews.size(); }
	public:
		void init(const VulkanContext* context);
		void rebuild();
		void release();

		static bool isBackBufferFormatSupport(const VulkanContext* context);
		static bool isBackBufferSupportAlphaBlend(const VulkanContext* context);
	};
}