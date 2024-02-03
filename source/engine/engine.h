#pragma once

#include "utils/utils.h"
#include "profile/profile.h"

namespace engine
{
	static const std::string kShaderCacheFolder = "save/shader/";
	static const std::string kLogCacheFolder    = "save/log/";
	static const std::string kConfigCacheFolder = "save/config/";

	extern void initBasicCVarConfigs();

	class Engine : NonCopyable
	{
	protected:
		Engine() = default;

		bool init();

		// Result is end of loop, will also end of application.
		bool loop(const ApplicationTickData& tickData);

		bool release();

	public:
		enum class EModuleState
		{
			None = 0,

			Initing,
			Ticking,
			Releasing,
		};

		static Engine* get();

		EModuleState getModuleState() const { return m_moduleState; }

		bool initGLFWWindowsHook(GLFWWindows& in);

		bool releaseGLFWWindowsHook();

		template<typename T>
		bool existRegisteredModule()
		{
			return m_registeredModulesIndexMap.contains(typeid(T).name());
		}

		bool isRuntimeModuleEmpty() const { return m_runtimeModules.empty(); }

		// Runtime module will tick sequence by register time, all run within main thread.
		template<typename T>
		[[nodiscard]] bool registerRuntimeModule()
		{
			checkRuntimeModule<T>();

			const std::string className = typeid(T).name();
			if (!m_registeredModulesIndexMap.contains(className))
			{
				auto newModule = std::make_unique<T>(this);
				newModule->registerCheck(this);

				m_registeredModulesIndexMap[className] = m_runtimeModules.size();

				LOG_TRACE("Register module: [{0}] at position [{1}].", className, m_runtimeModules.size());

				// Push back to module vector.
				m_runtimeModules.push_back(std::move(newModule));
				if (!m_runtimeModules.back()->init())
				{
					LOG_ERROR("Fail to init module: {}.", className);
					return false;
				}
				return true;
			}

			return false;
		}

		template <typename T>
		T* getRuntimeModule() const
		{
			checkRuntimeModule<T>();

			size_t moduleIndex = m_registeredModulesIndexMap.at(typeid(T).name());
			return static_cast<T*>(m_runtimeModules.at(moduleIndex).get());
		}

		// You should call this function after initGLFWWindowsHook called.
		bool isWindowApplication() const { return m_windowsInfo.isCompleted(); }

		const GLFWWindows* getGLFWWindows() const { return m_windowsInfo.windows; }
		GLFWWindows* getGLFWWindows() { return m_windowsInfo.windows; }

		ThreadPool* getThreadPool() { return m_threadPool.get(); }

	public:
		// Game state delegates.
		MulticastDelegate<> onGameStart;
		MulticastDelegate<> onGameStop;
		MulticastDelegate<> onGamePause;
		MulticastDelegate<> onGameContinue;

		// Game states control functions.
		void setGameStart();
		void setGameStop();
		void setGamePause();
		void setGameContinue();

		// Get game state.
		bool isRuningGame() const { return m_gameStates.bRuningGame; }
		bool isGamePaused() const { return isRuningGame() && m_gameStates.bPauseGame; }
		float getGameTime() const { return m_gameStates.time; }
		uint64_t getGameTickCount() const { return m_gameStates.tickCount; }

	private:
		struct GLFWWindowsInfo
		{
			bool isCompleted() const;

			// Windows info.
			GLFWWindows* windows = nullptr;

			// Handles of hook.
			DelegateHandle initHookHandle;
			DelegateHandle loopHookHandle;
			DelegateHandle releaseHookHandle;
		} m_windowsInfo;

		std::vector<std::unique_ptr<IRuntimeModule>> m_runtimeModules;
		std::map<std::string, size_t> m_registeredModulesIndexMap;

		// Engine timer.
		Timer m_timer;

		std::unique_ptr<ThreadPool> m_threadPool;

		// Game states.
		struct GameStates
		{
			bool bRuningGame = false;
			bool bPauseGame  = false;

			float time = 0.0f;
			uint64_t tickCount = 0U;
		} m_gameStates;

		EModuleState m_moduleState = EModuleState::None;
	};
}