#pragma once

#include "../utils/utils.h"
#include "../graphics/graphics.h"
#include "../ui/ui.h"

namespace engine
{
	constexpr size_t kMinRenderDim = 64;
	constexpr size_t kMaxRenderDim = 4096;

	struct TemporalBlueNoise;
	class SharedTextures;

	class RenderScene;
	class RendererManager final : public IRuntimeModule
	{
	public:
		RendererManager(Engine* engine) : IRuntimeModule(engine) { }
		~RendererManager() = default;

		virtual void registerCheck(Engine* engine) override;
		virtual bool init() override;
		virtual bool tick(const RuntimeModuleTickData& tickData) override;
		virtual bool beforeRelease() override;
		virtual bool release() override;

		// Delegate of tick functions.

		// Tick without command buffer, this tick first.
		MulticastDelegate<const RuntimeModuleTickData&, VulkanContext*> tickFunctions; 

		// Tick with command buffer, this tick second.
		MulticastDelegate<const RuntimeModuleTickData&, VkCommandBuffer, VulkanContext*> tickCmdFunctionsBefore;
		MulticastDelegate<const RuntimeModuleTickData&, VkCommandBuffer, VulkanContext*> tickCmdFunctions; 

		RenderScene* getScene() { return m_renderScene; }
		const RenderScene* getScene() const { return m_renderScene; }

		const TemporalBlueNoise& getBlueNoise() const { return *m_temporalBlueNoise; }
		const SharedTextures& getSharedTextures() const { return *m_sharedTextures; }

		const VulkanBuffer& getSSBODump() const { return *m_fallbackSSBO; }

	private:
		void initWindowCommandContext();
		void destroyWindowCommandContext();

	private:
		// Imgui manager.
		ImguiManager m_imguiManager;

		// Window command buffer context.
		struct
		{
			// Main command buffer and semaphore used when main window no minimized.
			std::vector<VkCommandBuffer> mainCmdRing;
			std::vector<VkSemaphore> mainSemaphoreRing;
		} m_windowCmdContext;

		RenderScene* m_renderScene = nullptr;
		SharedTextures* m_sharedTextures = nullptr;
		TemporalBlueNoise* m_temporalBlueNoise = nullptr;

		std::unique_ptr<VulkanBuffer> m_fallbackSSBO = nullptr;
	};

	extern RendererManager* getRenderer();
}