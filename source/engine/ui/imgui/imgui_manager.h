#pragma once

#include "../../graphics/graphics.h"

#include <iconFontcppHeaders/IconsFontAwesome6Brands.h>
#include <iconFontcppHeaders/IconsFontAwesome6.h>

namespace engine
{
    class ImguiManager
    {
	public:
		void init();
		void release();

		void render();
		void newFrame();
		void renderFrame(uint32_t backBufferIndex);
		void updateAfterSubmit();

		bool isMainMinimized();

		VkCommandBuffer getCommandBuffer(uint32_t index) const { return m_resources.commandBuffers.at(index); }

		void drawImGuiDemo();

	private:
		// Delegate handle cache when swapchain rebuild.
		DelegateHandle m_beforeSwapChainRebuildHandle;
		DelegateHandle m_afterSwapChainRebuildHandle;

		struct ImguiPassGpuResource
		{
			VkDescriptorPool descriptorPool;
			VkRenderPass renderPass = VK_NULL_HANDLE;

			std::vector<VkFramebuffer>   framebuffers;
			std::vector<VkCommandPool>   commandPools;
			std::vector<VkCommandBuffer> commandBuffers;
		} m_resources;

		// UI backbuffer clear value.
		math::vec4 m_clearColor = { 0.45f, 0.55f, 0.60f, 1.00f };

		// Optional, only create when backbuffer ui format no support alpha blend.
		VkFormat m_drawUIFormat = VK_FORMAT_UNDEFINED;
		std::unique_ptr<VulkanImage> m_drawUIImages;

		// Imgui render pass.
		void renderpassBuild();
		void renderpassRelease(bool bFullRelease);

		bool shouldBlitAfterRenderUI() const;
    };
}