#pragma once
#include "noncopyable.h"
#include <map>

namespace engine
{
	struct ApplicationTickData
	{
		// Application type.
		enum class ApplicationType
		{
			Windows,
			Console,
		} applicationType;

		// Useful when application is windows.
		struct WindowInfo
		{
			uint32_t wdith;
			uint32_t height;
			bool bIsMinimized;
			bool bLoseFocus;
		} windowInfo;
	};

	class Engine;

	// Per module tick data, time in seconds unit, used in engine module.
	struct RuntimeModuleTickData
	{
		// Application tick data.
		ApplicationTickData applicationInfo;

		// Engine realtime delta time.
		float deltaTime = 0.0f;

		// Engine delta time with smooth process. It diff from smoothFps, it update every frame with lerp smooth.
		float smoothDeltaTime = 0.0f;

		// Realtime fps.
		float fps = 0.0f;

		// Smooth fps which update per seconds.
		float smoothFps = 0.0f;

		// Engine run time.
		float runTime = 0.0f;

		// Total tick frame count of engine.
		uint64_t tickCount = 0;

		// Is smooth fps update this frame?
		bool bSmoothFpsUpdate = false;

		// Is game runing or not?
		bool bGameRuning = false;

		// Is game in pause state?
		bool bGamePause = false;

		// Game run time.
		float gameTime = 0.0f;

		// Total tick frame count of game.
		uint64_t gameTickCount = 0;
	};

	class IRuntimeModule : NonCopyable
	{
	public:
		explicit IRuntimeModule(Engine* in) : m_engine(in) { }

		virtual ~IRuntimeModule() = default;

		// Used to check module dependency.
		virtual void registerCheck(Engine* engine) { }

		virtual bool init() = 0;
		virtual bool tick(const RuntimeModuleTickData& tickData) = 0;

		virtual bool beforeRelease() { return true; }
		virtual bool release() = 0;

		const Engine* getEngine() const { return m_engine; }

	protected:
		Engine* m_engine;
	};

	template<typename T>
	constexpr void checkRuntimeModule()
	{
		static_assert(std::is_base_of<IRuntimeModule, T>::value, "This type doesn't match runtime module.");
		static_assert(std::is_constructible<T, Engine*>::value, "All module must can construct by Engine class pointer.");
	}
}