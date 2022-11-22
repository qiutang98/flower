#include "Pch.h"
#include "Renderer.h"
#include "ImGuiPass.h"
#include "DeferredRenderer/DeferredRenderer.h"
#include "SceneTextures.h"
#include "../UI/UIManager.h"
#include "RenderSettingContext.h"

namespace Flower
{
	static AutoCVarFloat cVarRequireUIFPS(
		"r.Render.UIFps",
		"Require ui fps, if world renderer is slow, will skip some frame to keep a smooth fps.",
		"Render",
		60,
		CVarFlags::ReadAndWrite
	);

	Renderer::Renderer(ModuleManager* in, std::string name)
		: IRuntimeModule(in, name)
	{
		
	}

	bool Renderer::init()
	{
		RenderSettingManager::get()->reset();

		UIManager::get()->init();
		m_uiPass.init();
		m_sceneData = std::make_unique<RenderSceneData>();

		// prepare common cmdbuffer and semaphore.
		{
			VkSemaphoreCreateInfo semaphoreInfo{};
			semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

			for (size_t i = 0; i < GBackBufferCount; i++)
			{
				m_dynamicGraphicsCommandBuffers[i] = RHI::get()->createMajorGraphicsCommandBuffer();
				RHICheck(vkCreateSemaphore(RHI::Device, &semaphoreInfo, nullptr, &m_dynamicGraphicsCommandExecuteSemaphores[i]));
			}
		}

		StaticTexturesManager::get()->init();
		return true;
	}

	void Renderer::tick(const RuntimeModuleTickData& tickData)
	{
		if (RenderSettingManager::get()->displayMode != RHI::eDisplayMode)
		{
			RHI::eDisplayMode = RenderSettingManager::get()->displayMode;
			RHI::get()->recreateSwapChain();
		}



		UIManager::get()->newFrame();

		imguiTickFunctions.broadcast(tickData);

		// Update scene data.
		m_sceneData->tick(tickData);

		ImGui::Render();



		ImDrawData* mainDrawData = ImGui::GetDrawData();
		const bool bMainMinimized = (mainDrawData->DisplaySize.x <= 0.0f || mainDrawData->DisplaySize.y <= 0.0f);
		if (!bMainMinimized)
		{
			uint32_t backBufferIndex = RHI::get()->acquireNextPresentImage();
			CHECK(backBufferIndex < GBackBufferCount && "Swapchain backbuffer count should equal to flighting count.");

			StaticTexturesManager::get()->tick();

			VkCommandBuffer graphicsCmd = m_dynamicGraphicsCommandBuffers[backBufferIndex];
			RHICheck(vkResetCommandBuffer(graphicsCmd, 0));
			VkCommandBufferBeginInfo cmdBeginInfo = RHICommandbufferBeginInfo(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
			RHICheck(vkBeginCommandBuffer(graphicsCmd, &cmdBeginInfo));
			{
				// Rebuild some global assset.
				if (RenderSettingManager::get()->ibl.needRebuild())
				{
					StaticTexturesManager::get()->rebuildIBL(graphicsCmd, false);
				}

				// Broadcast renderer tick functions.
				rendererTickHooks.broadcast(tickData, graphicsCmd);
			}
			RHICheck(vkEndCommandBuffer(graphicsCmd));

			// Record ui render.
			m_uiPass.renderFrame(backBufferIndex);

			auto frameStartSemaphore = RHI::get()->getCurrentFrameWaitSemaphore();
			auto* graphicsCmdEndSemaphore = &m_dynamicGraphicsCommandExecuteSemaphores[backBufferIndex];

			auto frameEndSemaphore = RHI::get()->getCurrentFrameFinishSemaphore();

			VkPipelineStageFlags waitFlags = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;

			std::vector<VkSemaphore> graphicsCmdWaitSemaphores = { frameStartSemaphore };
			RHISubmitInfo graphicsCmdSubmitInfo{};
			graphicsCmdSubmitInfo.setWaitStage(&waitFlags)
				.setWaitSemaphore(&frameStartSemaphore, 1)
				.setSignalSemaphore(graphicsCmdEndSemaphore, 1)
				.setCommandBuffer(&graphicsCmd, 1);

			RHISubmitInfo uiCmdSubmitInfo{};
			VkCommandBuffer uiCmdBuffer = m_uiPass.getCommandBuffer(backBufferIndex);
			uiCmdSubmitInfo.setWaitStage(&waitFlags)
				.setWaitSemaphore(graphicsCmdEndSemaphore, 1)
				.setSignalSemaphore(&frameEndSemaphore, 1)
				.setCommandBuffer(&uiCmdBuffer, 1);

			std::vector<VkSubmitInfo> infosRawSubmit{ graphicsCmdSubmitInfo, uiCmdSubmitInfo };

			RHI::get()->resetFence();
			RHI::get()->submit((uint32_t)infosRawSubmit.size(), infosRawSubmit.data());
		}

		UIManager::get()->updateAfterSubmit();

		if (!bMainMinimized)
		{
			RHI::get()->present();
		}
	}

	void Renderer::release()
	{
		// Free graphics command buffer misc.
		for (size_t i = 0; i < GBackBufferCount; i++)
		{
			vkDestroySemaphore(RHI::Device, m_dynamicGraphicsCommandExecuteSemaphores[i], nullptr);
		}
		StaticTexturesManager::get()->release();
		m_uiPass.release();
		UIManager::get()->release();

		RenderSettingManager::get()->release();
	}
}