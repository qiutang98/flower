#pragma once
#include "Pch.h"

#include "KeyCode.h"

namespace Flower
{
	struct LauncherInfo;
	extern void GLFWInit(const LauncherInfo&);

	// Global glfw window data.
	class GLFWWindowData
	{
		friend void GLFWInit(const LauncherInfo&);

	public:
		static GLFWWindowData* get();

		bool isKeyPressed(const KeyCode key);
		bool isMouseButtonPressed(MouseCode button);

		float getMouseX();
		float getMouseY();

		float getScrollOffsetX();
		float getScrollOffsetY();

		glm::vec2 getScrollOffset();
		glm::vec2 getMousePosition();

		int32_t  getWidth() { return m_width; }
		int32_t getHeight() { return m_height; }

		GLFWwindow* getWindow() { return m_window; }

		// Application is on focus or not.
		bool  isFocus() const { return m_bFocus; }

		// Should engine should still run main loop or exit.
		bool shouldRun() const { return m_bShouldRun; }

		// Set engine run state.
		void setShouldRun(bool bState);

		void callbackOnResize(int width, int height);
		void callbackOnMouseMove(double xpos, double ypos);
		void callbackOnMouseButton(int button, int action, int mods);
		void callbackOnScroll(double xoffset, double yoffset);
		void callbackOnSetFoucus(bool bFoucus);

	private:
		GLFWwindow* m_window;

		bool m_bFocus = true;
		bool m_bShouldRun = true;

		int32_t  m_width = 64;
		int32_t m_height = 64;

		glm::vec2 m_scrollOffset = { 0.0f, 0.0f };
	};
}