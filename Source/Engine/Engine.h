#pragma once

#include "Core/Core.h"
#include "RuntimeModule.h"
#include "EngineTimer.h"

namespace Flower
{
	struct EnginePreInitInfo
	{
		GLFWwindow* window;
	};

	struct EngineTickData
	{
		int32_t windowWidth;
		int32_t windowHeight;

		bool bIsMinimized;
		bool bLoseFocus;
	};

	class Engine final : private NonCopyable
	{
	private:
		std::unique_ptr<ModuleManager> m_moduleManager{ nullptr };

		EngineTimer m_timer;

	public:
		template <typename T>
		T* getRuntimeModule() const
		{
			return m_moduleManager->getRuntimeModule<T>();
		}
		 
		template<typename T>
		void registerRuntimeModule()
		{
			m_moduleManager->registerRuntimeModule<T>();
		}

		void preInit(const EnginePreInitInfo& info);
		void init();
		bool tick(const EngineTickData& data);
		void release();
	};

	extern Engine* GEngine;
}