#pragma once

#include "Core/Core.h"

namespace Flower 
{
	class ModuleManager;
	class Engine;

	struct RuntimeModuleTickData
	{
		int32_t windowWidth;
		int32_t windowHeight;

		// Application minimized?
		bool  bIsMinimized = false;

		// Application lose focus?
		bool  bLoseFocus = false;

		// Application realtime delta time.
		float deltaTime = 0.0f;

		// Application delta time with smooth process.
		float smoothDeltaTime = 0.0f;

		float fps = 0.0f;
		float smoothFps = 0.0f;

		float runTime = 0.0f;

		uint64_t tickCount = 0;

		bool bSmoothFpsUpdate = false;
	};

	class IRuntimeModule : private NonCopyable
	{
	protected:
		ModuleManager* m_moduleManager;
		std::string m_moduleName;

	public:
		IRuntimeModule(ModuleManager* in, std::string name)
			: m_moduleManager(in), m_moduleName(name)
		{

		}

		virtual ~IRuntimeModule() = default;

		virtual bool canInitCorrectly(size_t moduleIndex) { return true; }

		virtual bool init() { return true; }
		virtual void tick(const RuntimeModuleTickData& tickData) = 0;
		virtual void release() {  }

		virtual void registerCheck() { }
	};

	template<typename T>
	constexpr void checkRuntimeModule()
	{
		static_assert(std::is_base_of<IRuntimeModule, T>::value,
			"This type doesn't match runtime module.");
	}

	class ModuleManager : private NonCopyable
	{
		friend Engine;
	private:
		Engine* m_engine;
		std::vector<std::unique_ptr<IRuntimeModule>> m_runtimeModules;

	public:
		~ModuleManager() { }

		Engine* getEngine() { return m_engine; }

		// Runtime module will tick sequence by register time, all run within main thread.
		template<typename T>
		void registerRuntimeModule()
		{
			checkRuntimeModule<T>();

			auto newModule = std::make_unique<T>(this);
			newModule->registerCheck();

			m_runtimeModules.push_back(std::move(newModule));
		}

		bool init();
		void tick(const RuntimeModuleTickData& tickData);
		void release();

		template <typename T>
		T* getRuntimeModule() const
		{
			checkRuntimeModule<T>();
			for (const auto& runtimeModule : m_runtimeModules)
			{
				if (runtimeModule && typeid(T) == typeid(*runtimeModule))
				{
					// we has used checkRuntimeModule function to ensure T is base of IRuntimeModule.
					// so just use static_cast.
					return static_cast<T*>(runtimeModule.get());
				}
			}
			return nullptr;
		}
	};
}