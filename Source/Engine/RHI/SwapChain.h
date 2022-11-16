#pragma once
#include "RHICommon.h"

namespace Flower
{
	class Swapchain
	{
	private:
		std::vector<VkImage> m_swapchainImages = {};
		std::vector<VkImageView> m_swapchainImageViews = {};
		VkFormat m_swapchainImageFormat = {};
		VkExtent2D m_swapchainExtent = {};
		VkSwapchainKHR m_swapchain = {};

	private:
		VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
		VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
		VkExtent2D chooseSwapchainExtent(const VkSurfaceCapabilitiesKHR& capabilities);

	public:
		inline auto get() const { return m_swapchain; }

		inline auto& getImages() { return m_swapchainImages; }
		inline const auto& getImages() const { return m_swapchainImages; }
		inline auto& getImageViews() { return m_swapchainImageViews; }
		inline const auto& getImageViews() const { return m_swapchainImageViews; }

		inline auto getExtent() const { return m_swapchainExtent; }
		inline auto getImageFormat() const { return m_swapchainImageFormat; }

	public:
		void init();
		void rebuild();
		void release();
	};
}