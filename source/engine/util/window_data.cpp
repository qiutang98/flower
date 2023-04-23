#include "window_data.h"

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
		width  = newWidth;
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
}