#include "Pch.h"
#include "Engine.h"
#include "RHI/RHI.h"

namespace Flower
{
	Engine* GEngine = Singleton<Engine>::get();

	void Engine::preInit(const EnginePreInitInfo& info)
	{
		CHECK(info.window && "Please pass a useful windows for engine init.");
		RHI::get()->init(info.window);

		CHECK(m_moduleManager == nullptr && "Module manager is non empty, some memory leak happen.");
		m_moduleManager = std::make_unique<ModuleManager>();
		m_moduleManager->m_engine = this;

		m_soundEngine = irrklang::createIrrKlangDevice();
		CHECK(m_soundEngine);
	}

	void Engine::init()
	{
		CHECK(m_moduleManager && "You should init module manager before engine init.");
		CHECK(m_moduleManager->init());

		m_timer.init();
	}

	bool Engine::tick(const EngineTickData& data)
	{
		const bool bSmoothFpsUpdate = m_timer.tick();

		RuntimeModuleTickData tickData{};

		tickData.windowWidth = data.windowWidth;
		tickData.windowHeight = data.windowHeight;
		tickData.bLoseFocus = data.bLoseFocus;
		tickData.bIsMinimized = data.bIsMinimized;

		tickData.deltaTime = m_timer.getDt();
		tickData.smoothDeltaTime = m_timer.getSmoothDt();
		tickData.fps = m_timer.getFps();
		tickData.smoothFps = m_timer.getSmoothFps();
		tickData.tickCount = m_timer.getTickCount();
		tickData.bSmoothFpsUpdate = bSmoothFpsUpdate;
		tickData.runTime = m_timer.getRuntime();

		m_moduleManager->tick(tickData);

		return true;
	}

	void Engine::release()
	{
		// Ensure all vulkan task finish when starting release.
		vkDeviceWaitIdle(RHI::Device);

		// Release all module.
		m_moduleManager->release();

		// RHI resource release.
		RHI::get()->release();

		// sound engine close.
		m_soundEngine->drop();
	}
}