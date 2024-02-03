#include "engine.h"
#include "graphics/context.h"
#include "renderer/renderer.h"
#include "asset/asset_manager.h"
#include "scene/scene_manager.h"

#include <filesystem>
#include "profile/profile.h"

namespace engine
{
	inline void tryImportOrCreateIni(const std::string& path)
	{
		if (std::filesystem::exists(path))
		{
			CVarSystem::get()->importConfig(path);
		}
		else
		{
			CVarSystem::get()->exportAllConfig(path);
		}
	};

	void engine::initBasicCVarConfigs()
	{
		// Create cvar configs paths.
		std::filesystem::create_directories("config");

		std::filesystem::create_directories(kLogCacheFolder);
		std::filesystem::create_directories(kShaderCacheFolder);
		std::filesystem::create_directories(kConfigCacheFolder);

		// Always override cvar configs.
		CVarSystem::get()->exportAllConfig("config/default.ini");

		// try import or create engine and editor ini.
		tryImportOrCreateIni("config/engine.ini");
		tryImportOrCreateIni("config/editor.ini");
	}

	bool Engine::init()
	{
		m_moduleState = EModuleState::Initing;

		bool bResult = true;

		// Engine timer init, use 5 frame to smooth fps and dt, use 5.0 as min fps to compute smooth time.
		m_timer.init(5.0, 5.0);
		m_threadPool = std::make_unique<ThreadPool>();

		// Register basic module of the engine.
		bResult &= registerRuntimeModule<AssetManager>();
		bResult &= registerRuntimeModule<VulkanContext>();
		bResult &= registerRuntimeModule<RendererManager>();
		bResult &= registerRuntimeModule<SceneManager>();

		return bResult;
	}

	bool Engine::loop(const ApplicationTickData& applicationInfo)
	{
		m_moduleState = EModuleState::Ticking;

		bool bContinue = true;
		if (m_runtimeModules.size() > 0)
		{
			// Update engine timer.
			const bool bSmoothFpsUpdate = m_timer.tick();

			// Prepare module update tick data.
			RuntimeModuleTickData tickData{ };
			tickData.applicationInfo  = applicationInfo;
			tickData.deltaTime        = m_timer.getDt();
			tickData.smoothDeltaTime  = m_timer.getSmoothDt();
			tickData.fps              = m_timer.getFps();
			tickData.smoothFps        = m_timer.getSmoothFps();
			tickData.tickCount        = m_timer.getTickCount();
			tickData.bSmoothFpsUpdate = bSmoothFpsUpdate;
			tickData.runTime          = m_timer.getRuntime();

			// Game state of engine.
			tickData.bGameRuning = isRuningGame();
			if (tickData.bGameRuning)
			{
				// If game paused, we store increment delta time and frame count.
				if (!isGamePaused())
				{
					m_gameStates.time += tickData.deltaTime;
					m_gameStates.tickCount++;
				}

				tickData.gameTime = m_gameStates.time;
				tickData.gameTickCount = m_gameStates.tickCount;
			}

			// Modules tick.
			for (const auto& runtimeModule : m_runtimeModules)
			{
				bContinue &= runtimeModule->tick(tickData);
			}
		}

		FrameMark;
		return bContinue;
	}

	bool Engine::release()
	{
		m_moduleState = EModuleState::Releasing;

		// Before release state.
		{
			// Stop game before release.
			if (isRuningGame())
			{
				setGameStop();
			}

			// Before release runtime modules.
			if (m_runtimeModules.size() > 0)
			{
				bool bAllModuleReady = true;

				// Before release from end to start.
				for (size_t i = m_runtimeModules.size(); i > 0; i--)
				{
					if (!m_runtimeModules[i - 1]->beforeRelease())
					{
						bAllModuleReady = false;
					}
				}

				if (!bAllModuleReady)
				{
					LOG_ERROR("No all module before release correctly, check prev logs to fix the bugs.");
					return false;
				}
			}
		}

		// Releasing.
		{
			// Release runtime modules.
			if (m_runtimeModules.size() > 0)
			{
				bool bAllModuleReady = true;

				// Release from end to start.
				for (size_t i = m_runtimeModules.size(); i > 0; i--)
				{
					if (!m_runtimeModules[i - 1]->release())
					{
						bAllModuleReady = false;
					}
				}

				if (!bAllModuleReady)
				{
					LOG_ERROR("No all module release correctly, check prev logs to fix the bugs.");
					return false;
				}

				// Manually destruct from end to start.
				for (size_t i = m_runtimeModules.size() - 1; i > 0; i--)
				{
					m_runtimeModules[i].reset();
				}

				m_runtimeModules.clear();
			}
		}

		return true;
	}

	Engine* Engine::get()
	{
		static Engine engine;
		return &engine;
	}

	// Hook application.
	bool Engine::initGLFWWindowsHook(GLFWWindows& in)
	{
		if (m_windowsInfo.isCompleted())
		{
			LOG_ERROR("Can't init GLFW windows hook twice!");
			return false;
		}

		// Cache windows.
		m_windowsInfo.windows = &in;

		// Cache hook handles.
		m_windowsInfo.initHookHandle = in.registerInitBody([this](const GLFWWindows*) 
		{ 
			return this->init(); 
		});
		m_windowsInfo.loopHookHandle = in.registerLoopBody([this](const GLFWWindows*, const ApplicationTickData& tickData)
		{ 
			return this->loop(tickData); 
		});
		m_windowsInfo.releaseHookHandle = in.registerReleaseBody([this](const GLFWWindows*) 
		{ 
			return this->release(); 
		});

		CHECK(m_windowsInfo.isCompleted());
		return true;
	}

	// Hook application.
	bool Engine::releaseGLFWWindowsHook()
	{
		if (!m_windowsInfo.isCompleted())
		{
			LOG_ERROR("Can't release GLFW windows hook when it uncompleted!");
			return false;
		}

		m_windowsInfo.windows->unregisterInitBody(m_windowsInfo.initHookHandle);
		m_windowsInfo.windows->unregisterLoopBody(m_windowsInfo.loopHookHandle);
		m_windowsInfo.windows->unregisterReleaseBody(m_windowsInfo.releaseHookHandle);

		m_windowsInfo.windows = nullptr;
		m_windowsInfo.initHookHandle = {};
		m_windowsInfo.loopHookHandle = {};
		m_windowsInfo.releaseHookHandle = {};

		CHECK(!m_windowsInfo.isCompleted());
		return true;
	}

	bool Engine::GLFWWindowsInfo::isCompleted() const
	{
		return windows != nullptr
			&& initHookHandle.isValid()
			&& loopHookHandle.isValid()
			&& releaseHookHandle.isValid();
	}

	void Engine::setGameStart()
	{
		if (!isRuningGame())
		{
			m_gameStates.bRuningGame = true;
			m_gameStates.bPauseGame = false;
			m_gameStates.time = 0.0f;
			m_gameStates.tickCount = 0;

			onGameStart.broadcast();
		}
	}

	void Engine::setGameStop()
	{
		if (isRuningGame())
		{
			m_gameStates.bRuningGame = false;
			m_gameStates.bPauseGame = false;
			m_gameStates.time = 0.0f;
			m_gameStates.tickCount = 0;

			onGameStop.broadcast();
		}
	}

	void Engine::setGamePause()
	{
		if (isRuningGame() && !isGamePaused())
		{
			m_gameStates.bPauseGame = true;
			onGamePause.broadcast();
		}
	}

	void Engine::setGameContinue()
	{
		if (isRuningGame() && isGamePaused())
		{
			m_gameStates.bPauseGame = false;
			onGameContinue.broadcast();
		}
	}
}

