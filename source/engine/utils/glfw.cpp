#include "glfw.h"

#include <stb/stb_image.h>

namespace engine
{
	bool GLFWWindowData::isKeyPressed(const KeyCode key) const
	{
		auto state = glfwGetKey(window, static_cast<int32_t>(key));
		return state == GLFW_PRESS || state == GLFW_REPEAT;
	}

	bool GLFWWindowData::isMouseButtonPressed(const MouseCode button) const
	{
		auto state = glfwGetMouseButton(window, static_cast<int32_t>(button));
		return state == GLFW_PRESS;
	}

	void GLFWWindowData::callbackOnResize(int newWidth, int newHeight)
	{
		width = newWidth;
		height = newHeight;
	}

	void GLFWWindowData::callbackOnMouseMove(double xpos, double ypos)
	{
		mousePosition.x = (float)xpos;
		mousePosition.y = (float)ypos;
	}

	void GLFWWindowData::callbackOnMouseButton(int button, int action, int mods)
	{

	}

	void GLFWWindowData::callbackOnScroll(double xoffset, double yoffset)
	{
		scrollOffset.x = (float)xoffset;
		scrollOffset.y = (float)yoffset;
	}

	void GLFWWindowData::callbackOnSetFoucus(bool inFocusState)
	{
		bFocus = inFocusState;
	}

	// GLFW window callbacks.
	static void resizeCallBack(GLFWwindow* window, int width, int height)
	{
		auto app = reinterpret_cast<GLFWWindowData*>(glfwGetWindowUserPointer(window));
		app->callbackOnResize(width, height);
	}

	static void mouseMoveCallback(GLFWwindow* window, double xpos, double ypos)
	{
		auto app = reinterpret_cast<GLFWWindowData*>(glfwGetWindowUserPointer(window));
		app->callbackOnMouseMove(xpos, ypos);
	}

	static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
	{
		auto app = reinterpret_cast<GLFWWindowData*>(glfwGetWindowUserPointer(window));
		app->callbackOnMouseButton(button, action, mods);
	}

	static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
	{
		auto app = reinterpret_cast<GLFWWindowData*>(glfwGetWindowUserPointer(window));
		app->callbackOnScroll(xoffset, yoffset);
	}

	static void windowFocusCallBack(GLFWwindow* window, int focused)
	{
		auto app = reinterpret_cast<GLFWWindowData*>(glfwGetWindowUserPointer(window));
		app->callbackOnSetFoucus((bool)focused);
	}

	bool GLFWWindows::init(InitConfig info)
	{
		m_name = info.appName;

		if (glfwInit() != GLFW_TRUE)
		{
			LOG_ERROR("Fail to init glfw, application will pre-exit!");
			return false;
		}

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, info.bResizable ? GL_TRUE : GL_FALSE);

		using ShowModeEnum = InitConfig::EWindowMode;

		// Get monitor information.
		const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
		m_data.monitorMaxSize.x = mode->width;
		m_data.monitorMaxSize.y = mode->height;

		if (info.windowShowMode == ShowModeEnum::FullScreenWithoutTile)
		{
			glfwWindowHint(GLFW_RED_BITS, mode->redBits);
			glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
			glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
			glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
			glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);

			m_data.window = glfwCreateWindow(mode->width, mode->height, info.appName.c_str(), nullptr, nullptr);
		}
		else
		{
			auto clampWidth = std::max(1u, info.initWidth);
			auto clampHeight = std::max(1u, info.initHeight);

			if (info.windowShowMode == ShowModeEnum::FullScreenWithTile)
			{
				glfwWindowHint(GLFW_MAXIMIZED, GL_TRUE);
			}

			m_data.window = glfwCreateWindow(clampWidth, clampHeight, info.appName.c_str(), nullptr, nullptr);
			{
				int32_t width, height;
				glfwGetWindowSize(m_data.window, &width, &height);

				m_data.width = static_cast<uint32_t>(width);
				m_data.height = static_cast<uint32_t>(height);
			}

			// Center free mode to window.
			if (info.windowShowMode == ShowModeEnum::Free)
			{
				glfwSetWindowPos(m_data.window, (mode->width - m_data.width) / 2, (mode->height - m_data.height) / 2);
			}
		}

		{
			int32_t width, height;
			glfwGetWindowSize(m_data.window, &width, &height);

			m_data.width = static_cast<uint32_t>(width);
			m_data.height = static_cast<uint32_t>(height);
		}
		LOG_INFO("Create window size: ({0},{1}).", m_data.width, m_data.height);

		glfwSetWindowUserPointer(m_data.window, (void*)(&m_data));
		glfwSetFramebufferSizeCallback(m_data.window, resizeCallBack);
		glfwSetMouseButtonCallback(m_data.window, mouseButtonCallback);
		glfwSetCursorPosCallback(m_data.window, mouseMoveCallback);
		glfwSetScrollCallback(m_data.window, scrollCallback);
		glfwSetWindowFocusCallback(m_data.window, windowFocusCallBack);

		m_data.icon.pixels = stbi_load(info.appIcon.c_str(), &m_data.icon.width, &m_data.icon.height, 0, 4);
		glfwSetWindowIcon(m_data.window, 1, &m_data.icon);

		bool bInitResult = true;
		m_initBodies.broadcast(this, bInitResult);

		return bInitResult;
	}

	void GLFWWindows::loop()
	{
		while (m_data.bShouldRun)
		{
			glfwPollEvents();

			ApplicationTickData tickData{};
			tickData.applicationType = ApplicationTickData::ApplicationType::Windows;
			tickData.windowInfo.wdith = m_data.width;
			tickData.windowInfo.height = m_data.height;
			tickData.windowInfo.bIsMinimized = m_data.width <= 0 || m_data.height <= 0;
			tickData.windowInfo.bLoseFocus = !m_data.bFocus;

			m_loopBodies.broadcast(this, tickData, m_data.bShouldRun);

			const bool bWindowRequireClose = glfwWindowShouldClose(m_data.window);
			m_data.bShouldRun &= (!bWindowRequireClose);

			if (!m_data.bShouldRun && m_closedEventBodies.getSize() > 0)
			{
				m_data.bShouldRun = true;
				m_closedEventBodies.broadcast(this, m_data.bShouldRun);

				glfwSetWindowShouldClose(m_data.window, !m_data.bShouldRun);
			}
		}
	}

	bool GLFWWindows::release()
	{
		bool bReleaseResult = true;
		m_releaseBodies.broadcast(this, bReleaseResult);

		// Free icon memory.
		stbi_image_free(m_data.icon.pixels);

		// Release windows.
		glfwDestroyWindow(m_data.window);
		glfwTerminate();

		return bReleaseResult;
	}

	DelegateHandle GLFWWindows::registerInitBody(std::function<bool(const GLFWWindows* windows)> function)
	{
		return m_initBodies.addLambda([function](const GLFWWindows* windows, bool& bResult)
		{
			bResult &= function(windows);
		});
	}

	void GLFWWindows::unregisterInitBody(DelegateHandle delegate)
	{
		CHECK(delegate.isValid());
		CHECK(m_initBodies.remove(delegate));
	}

	DelegateHandle GLFWWindows::registerLoopBody(std::function<bool(const GLFWWindows* windows, const ApplicationTickData&)> function)
	{
		return m_loopBodies.addLambda([function](const GLFWWindows* windows, const ApplicationTickData& tickData, bool& bContinue)
		{
			bContinue &= function(windows, tickData);
		});
	}

	void GLFWWindows::unregisterLoopBody(DelegateHandle delegate)
	{
		CHECK(delegate.isValid());
		CHECK(m_loopBodies.remove(delegate));
	}

	DelegateHandle GLFWWindows::registerReleaseBody(std::function<bool(const GLFWWindows* windows)> function)
	{
		return m_releaseBodies.addLambda([function](const GLFWWindows* windows, bool& bResult)
		{
			bResult &= function(windows);
		});
	}

	void GLFWWindows::unregisterReleaseBody(DelegateHandle delegate)
	{
		CHECK(delegate.isValid());
		CHECK(m_releaseBodies.remove(delegate));
	}

	DelegateHandle GLFWWindows::registerClosedEventBody(std::function<bool(const GLFWWindows* windows)> function)
	{
		return m_closedEventBodies.addLambda([function](const GLFWWindows* windows, bool& bResult)
		{
			bResult &= function(windows);
		});
	}

	void GLFWWindows::unregisterClosedEventBody(DelegateHandle delegate)
	{
		CHECK(delegate.isValid());
		CHECK(m_closedEventBodies.remove(delegate));
	}
}