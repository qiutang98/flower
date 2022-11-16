#include "Pch.h"

#include "WindowData.h"

namespace Flower
{
	GLFWWindowData* GLFWWindowData::get()
	{
		static GLFWWindowData window{};
		return &window;
	}

	float GLFWWindowData::getMouseX()
	{
		return getMousePosition().x;
	}

	float GLFWWindowData::getMouseY()
	{
		return getMousePosition().y;
	}

	glm::vec2 GLFWWindowData::getScrollOffset()
	{
		return m_scrollOffset;
	}

	float GLFWWindowData::getScrollOffsetX()
	{
		return m_scrollOffset.x;
	}

	float GLFWWindowData::getScrollOffsetY()
	{
		return m_scrollOffset.y;
	}

	bool GLFWWindowData::isKeyPressed(const KeyCode key)
	{
		auto state = glfwGetKey(m_window, static_cast<int32_t>(key));
		return state == GLFW_PRESS || state == GLFW_REPEAT;
	}

	bool GLFWWindowData::isMouseButtonPressed(const MouseCode button)
	{
		auto state = glfwGetMouseButton(m_window, static_cast<int32_t>(button));
		return state == GLFW_PRESS;
	}

	glm::vec2 GLFWWindowData::getMousePosition()
	{
		double xpos, ypos;
		glfwGetCursorPos(m_window, &xpos, &ypos);

		return { (float)xpos, (float)ypos };
	}

	void GLFWWindowData::setShouldRun(bool bState)
	{
		m_bShouldRun &= bState;
	}

	void GLFWWindowData::callbackOnResize(int width, int height)
	{
		m_width = width;
		m_height = height;
	}

	void GLFWWindowData::callbackOnMouseMove(double xpos, double ypos)
	{

	}

	void GLFWWindowData::callbackOnMouseButton(int button, int action, int mods)
	{

	}

	void GLFWWindowData::callbackOnScroll(double xoffset, double yoffset)
	{
		m_scrollOffset.x = (float)xoffset;
		m_scrollOffset.y = (float)yoffset;
	}

	void GLFWWindowData::callbackOnSetFoucus(bool bFocus)
	{
		m_bFocus = bFocus;
	}
}