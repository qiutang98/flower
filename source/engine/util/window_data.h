#pragma once

#include <vulkan/vulkan.h>
#include <glfw/glfw3.h>

#include "math.h"
#include "keycode.h"

namespace engine
{
    struct GLFWWindowData
    {
        // Cache windows pointer.
        GLFWwindow* window = nullptr;

        GLFWimage icon;

        bool bFocus = true; // Current windows under focus state.
        bool bShouldRun = true; // Windows app should run or not?

        // Windows state cache which will update by callback.
        int32_t width  = 64;
        int32_t height = 64;
        math::vec2 scrollOffset  = { 0.0f, 0.0f };
        math::vec2 mousePosition = { 0.0f, 0.0f };

        // Get key state.
        bool isKeyPressed(const KeyCode key) const;
		bool isMouseButtonPressed(MouseCode button) const;

        // Get mouse scroll.
		math::vec2 getScrollOffset() const { return scrollOffset; }
        float getScrollOffsetX() const { return getScrollOffset().x; }
		float getScrollOffsetY() const { return getScrollOffset().y; }

        // Get mouse position.
		math::vec2 getMousePosition() const { return mousePosition; }
        float getMouseX() const { return getMousePosition().x; }
		float getMouseY() const { return getMousePosition().y; }

        // Callback function bind to glfw.
        void callbackOnResize(int width, int height);
		void callbackOnMouseMove(double xpos, double ypos);
		void callbackOnMouseButton(int button, int action, int mods);
		void callbackOnScroll(double xoffset, double yoffset);
		void callbackOnSetFoucus(bool bFoucus);
    };
}
