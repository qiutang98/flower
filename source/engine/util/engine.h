#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <map>

#include "timer.h"
#include "noncopyable.h"
#include "macro.h"

namespace engine
{
	struct EngineTickData
	{
		// [Window] width.
		int32_t windowWidth;

		// [Window] height.
		int32_t windowHeight;

		// [Window] Application minimized?
		bool bIsMinimized;

		// [Window] Application lose focus?
		bool bLoseFocus;
	};
	
    // Per module tick data. time in seconds unit.
	struct RuntimeModuleTickData
	{
		// [Window] width.
        int32_t windowWidth;

		// [Window] height.
		int32_t windowHeight;

		// [Window] Application minimized?
		bool  bIsMinimized = false;

		// [Window] Application lose focus?
		bool  bLoseFocus = false;

		// Application realtime delta time.
		float deltaTime = 0.0f;

		// Application delta time with smooth process. It diff from smoothFps, it update every frame with lerp smooth.
		float smoothDeltaTime = 0.0f;

		// Realtime fps.
		float fps = 0.0f;

		// Smooth fps, update per seconds.
		float smoothFps = 0.0f;

		// Application run time.
		float runTime = 0.0f;

		// Total tick frame count.
		uint64_t tickCount = 0;

		// Is smooth fps update this frame?
		bool bSmoothFpsUpdate = false;
	};

	class Engine;
	class IRuntimeModule : private NonCopyable
	{
	protected:
		Engine* m_engine;

	public:
		IRuntimeModule(Engine* in) : m_engine(in) { }

		virtual ~IRuntimeModule() = default;

		virtual bool init() { return true; }
		virtual bool tick(const RuntimeModuleTickData& tickData) = 0;
		virtual void release() {  }

		// Used to check module dependency.
		virtual void registerCheck(Engine* engine) { }

		const Engine* getEngine() const { return m_engine; }
	};

	template<typename T>
	constexpr void checkRuntimeModule()
	{
		static_assert(std::is_base_of<IRuntimeModule, T>::value, "This type doesn't match runtime module.");
	}

	class Engine final : private NonCopyable
	{
	private:
		class Framework* m_framework = nullptr;
		std::vector<std::unique_ptr<IRuntimeModule>> m_runtimeModules;
		std::map<std::string, size_t> m_registeredModulesIndexMap;

		Timer m_timer;

	public:
		template<typename T>
		bool existRegisteredModule()
		{
			return m_registeredModulesIndexMap.contains(typeid(T).name());
		}

		bool isRuntimeModuleEmpty() const { return m_runtimeModules.empty(); }

		// Runtime module will tick sequence by register time, all run within main thread.
		template<typename T>
		void registerRuntimeModule()
		{
			checkRuntimeModule<T>();

			const std::string className = typeid(T).name();
			if (!m_registeredModulesIndexMap.contains(className))
			{
				auto newModule = std::make_unique<T>(this);
				newModule->registerCheck(this);

				m_registeredModulesIndexMap[className] = m_runtimeModules.size();

				LOG_TRACE("Register module: [{0}] at position [{1}].", className, m_runtimeModules.size());
				m_runtimeModules.push_back(std::move(newModule));
			}
		}

		template <typename T>
		T* getRuntimeModule() const
		{
			checkRuntimeModule<T>();

			size_t moduleIndex = m_registeredModulesIndexMap.at(typeid(T).name());
			return static_cast<T*>(m_runtimeModules.at(moduleIndex).get());
		}

		bool init(Framework* framework);
		bool tick(const EngineTickData& tickData);
		void release();

		bool isConsoleApp() const;
		bool isWindowApp() const;

		const Framework* getFramework() const { return m_framework; }
	};
}