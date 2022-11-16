#pragma once

#include "../RuntimeModule.h"

#include "ImGuiPass.h"
#include "RendererCommon.h"
#include "RenderTexturePool.h"
#include "RenderSceneData.h"

namespace Flower
{
	class DeferredRenderer;
	class Renderer : public IRuntimeModule
	{
	private:
		ImguiPass m_uiPass;
		std::unique_ptr<RenderSceneData> m_sceneData;

		// Major graphics queue's command and semaphores.
		std::array<VkCommandBuffer, GBackBufferCount> m_dynamicGraphicsCommandBuffers;
		std::array<VkSemaphore, GBackBufferCount> m_dynamicGraphicsCommandExecuteSemaphores;

	public:
		MulticastDelegate<const RuntimeModuleTickData&> imguiTickFunctions;
		MulticastDelegate<const RuntimeModuleTickData&, VkCommandBuffer> rendererTickHooks;

		RenderSceneData* getRenderScene() const
		{
			return m_sceneData.get();
		}

	public:
		Renderer(ModuleManager* in, std::string name = "Renderer");

		virtual bool init() override;
		virtual void release() override;
		virtual void tick(const RuntimeModuleTickData& tickData) override;
	};
}