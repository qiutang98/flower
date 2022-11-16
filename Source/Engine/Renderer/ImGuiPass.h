#pragma once
#include "../RHI/RHI.h"

namespace Flower
{
	class ImguiPass
	{
	private:
		bool m_bInit = false;
		DelegateHandle m_beforeSwapChainRebuildHandle;
		DelegateHandle m_afterSwapChainRebuildHandle;

		struct ImguiPassGpuResource
		{
			VkDescriptorPool descriptorPool;
			VkRenderPass renderPass = VK_NULL_HANDLE;

			std::vector<VkFramebuffer>   framebuffers;
			std::vector<VkCommandPool>   commandPools;
			std::vector<VkCommandBuffer> commandBuffers;
		} m_renderResource;

		glm::vec4 m_clearColor{ 0.45f, 0.55f, 0.60f, 1.00f };

	private:
		const VkFormat m_drawUIFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
		std::shared_ptr<VulkanImage> m_drawUIImages;

		void renderpassBuild();
		void renderpassRelease(bool bFullRelease);

	public:
		VkCommandBuffer getCommandBuffer(uint32_t index)
		{
			return m_renderResource.commandBuffers[index];
		}

	public:
		~ImguiPass() = default;

		void init();
		void release();

		void renderFrame(uint32_t backBufferIndex);
	};
}