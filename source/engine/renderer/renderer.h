#pragma once

#include <util/util.h>
#include <rhi/rhi.h>

#include <imgui/imgui_manager.h>

namespace engine
{
	struct TemporalBlueNoise;
	class SharedTextures;
	class Renderer final : public IRuntimeModule
	{
	public:
		Renderer(Engine* engine) : IRuntimeModule(engine) { }
		~Renderer() = default;

		virtual void registerCheck(Engine* engine) override;
		virtual bool init() override;
		virtual bool tick(const RuntimeModuleTickData& tickData) override;
		virtual void release() override;

		// Get cache context.
		VulkanContext* getContext() const { return m_context; }

		// Delegate of tick functions.
		MulticastDelegate<const RuntimeModuleTickData&, VulkanContext*> tickFunctions; // Tick without command buffer, this tick first.
        MulticastDelegate<const RuntimeModuleTickData&, VkCommandBuffer, VulkanContext*> tickCmdFunctions; // Tick with command buffer, this tick second.

		class RenderScene* getScene() { return m_renderScene; }

		const TemporalBlueNoise& getBlueNoise() const { return *m_temporalBlueNoise; }
		const SharedTextures& getSharedTextures() const { return *m_sharedTextures; }

    private:
        void initWindowCommandContext();
		void destroyWindowCommandContext();

    private:
		// RHI context.
        VulkanContext* m_context;

		class RenderScene* m_renderScene = nullptr;

		TemporalBlueNoise* m_temporalBlueNoise = nullptr;
		SharedTextures* m_sharedTextures = nullptr;

		// Imgui manager.
		ImguiManager m_imguiManager;

		// Window command buffer context.
		struct
		{
			// Main command buffer and semaphore used when main window no minimized.
			std::vector<VkCommandBuffer> mainCmdRing;
			std::vector<VkSemaphore> mainSemaphoreRing;

			// Second command buffer and fence used when main window minimized.
			VkFence secondCmdFence;
			VkCommandBuffer secondCmd;

		} m_windowCmdContext;
	};

	extern Renderer* getRenderer();
}