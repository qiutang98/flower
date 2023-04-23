#pragma once

#include "window_data.h"
#include "engine.h"
#include "config.h"

namespace engine
{
	// This should be singleton, alway only exist one.
	class Framework
	{
    public:
		static Framework* get();

		// Only init framework.
		void initFramework(Config config);

		// App init.
		[[nodiscard]] bool init();

		// App loop.
		void loop();

		// App release.
		void release();

		// Current application run in console mode?
		bool isConsole() const { return m_config.bConsole; }

		// Get application engine.
		const Engine& getEngine() const { return m_engine; }
		Engine& getEngine() { return m_engine; }
		Engine* getEnginePtr() { return &m_engine; }

		// Get glfw window handle.
		GLFWwindow* getWindow() const { return m_data.window; }

		// Get framework config.
		const Config& getConfig() const { return m_config; }

		const GLFWWindowData& getWindowData() const { return m_data; }

	protected:
		Framework() = default;



	private:
		void configInit(const Config& in);

		// Window app init.
		void windowInit();
        void windowRelease();

		// Console app init.
        void consoleInit();
        void consoleRelease();

    private:
		// Init state check, use this state to ensure framework only init once.
		bool m_bInit = false;

		// GLFW window data.
        GLFWWindowData m_data;

		// Engine cache.
		Engine m_engine;

		// Framework config.
		Config m_config;
	};
}