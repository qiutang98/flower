#pragma once

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <cstdint>

#include "math.h"
#include "delegate.h"
#include "base.h"

namespace engine
{
	typedef uint16_t KeyCode;
	namespace Key
	{
		enum : KeyCode
		{
			// From glfw3.h
			Space = 32,
			Apostrophe = 39, /* ' */
			Comma = 44,      /* , */
			Minus = 45,      /* - */
			Period = 46,     /* . */
			Slash = 47,      /* / */

			D0 = 48, /* 0 */
			D1 = 49, /* 1 */
			D2 = 50, /* 2 */
			D3 = 51, /* 3 */
			D4 = 52, /* 4 */
			D5 = 53, /* 5 */
			D6 = 54, /* 6 */
			D7 = 55, /* 7 */
			D8 = 56, /* 8 */
			D9 = 57, /* 9 */

			Semicolon = 59, /* ; */
			Equal = 61,     /* = */

			A = 65,
			B = 66,
			C = 67,
			D = 68,
			E = 69,
			F = 70,
			G = 71,
			H = 72,
			I = 73,
			J = 74,
			K = 75,
			L = 76,
			M = 77,
			N = 78,
			O = 79,
			P = 80,
			Q = 81,
			R = 82,
			S = 83,
			T = 84,
			U = 85,
			V = 86,
			W = 87,
			X = 88,
			Y = 89,
			Z = 90,

			LeftBracket = 91,  /* [ */
			Backslash = 92,    /* \ */
			RightBracket = 93, /* ] */
			GraveAccent = 96,  /* ` */

			World1 = 161, /* non-US #1 */
			World2 = 162, /* non-US #2 */

			/* Function keys */
			Escape = 256,
			Enter = 257,
			Tab = 258,
			Backspace = 259,
			Insert = 260,
			Delete = 261,
			Right = 262,
			Left = 263,
			Down = 264,
			Up = 265,
			PageUp = 266,
			PageDown = 267,
			Home = 268,
			End = 269,
			CapsLock = 280,
			ScrollLock = 281,
			NumLock = 282,
			PrintScreen = 283,
			Pause = 284,
			F1 = 290,
			F2 = 291,
			F3 = 292,
			F4 = 293,
			F5 = 294,
			F6 = 295,
			F7 = 296,
			F8 = 297,
			F9 = 298,
			F10 = 299,
			F11 = 300,
			F12 = 301,
			F13 = 302,
			F14 = 303,
			F15 = 304,
			F16 = 305,
			F17 = 306,
			F18 = 307,
			F19 = 308,
			F20 = 309,
			F21 = 310,
			F22 = 311,
			F23 = 312,
			F24 = 313,
			F25 = 314,

			/* Keypad */
			KP0 = 320,
			KP1 = 321,
			KP2 = 322,
			KP3 = 323,
			KP4 = 324,
			KP5 = 325,
			KP6 = 326,
			KP7 = 327,
			KP8 = 328,
			KP9 = 329,
			KPDecimal = 330,
			KPDivide = 331,
			KPMultiply = 332,
			KPSubtract = 333,
			KPAdd = 334,
			KPEnter = 335,
			KPEqual = 336,

			LeftShift = 340,
			LeftControl = 341,
			LeftAlt = 342,
			LeftSuper = 343,
			RightShift = 344,
			RightControl = 345,
			RightAlt = 346,
			RightSuper = 347,
			Menu = 348
		};
	}

	typedef uint16_t MouseCode;
	namespace Mouse
	{
		enum : MouseCode
		{
			// From glfw3.h
			Button0 = 0,
			Button1 = 1,
			Button2 = 2,
			Button3 = 3,
			Button4 = 4,
			Button5 = 5,
			Button6 = 6,
			Button7 = 7,

			ButtonLast = Button7,
			ButtonLeft = Button0,
			ButtonRight = Button1,
			ButtonMiddle = Button2
		};
	}

    struct GLFWWindowData
    {
        // Cache windows pointer.
        GLFWwindow* window = nullptr;

        GLFWimage icon;

        bool bFocus = true; // Current windows under focus state.
        bool bShouldRun = true; // Windows app should run or not?

        // Windows state cache which will update by callback.
        uint32_t width  = 64U;
        uint32_t height = 64U;

		// Scroll offset, no delta!!!
        math::vec2 scrollOffset = { 0.0f, 0.0f };
        math::vec2 mousePosition = { 0.0f, 0.0f };

		// Monitor max size.
		math::ivec2 monitorMaxSize = { 0.0f, 0.0f };

        // Get key state.
        bool isKeyPressed(const KeyCode key) const;
        bool isMouseButtonPressed(MouseCode button) const;


		// Scroll offset, no delta!!!
        math::vec2 getScrollOffset() const { return scrollOffset; }
		// Scroll offset, no delta!!!
        float getScrollOffsetX() const { return getScrollOffset().x; }
		// Scroll offset, no delta!!!
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

    class GLFWWindows
    {
	public:
		GLFWWindows() = default;

		struct InitConfig
		{
			// Init app name.
			std::string appName;
			std::string appIcon;

			enum class EWindowMode
			{
				FullScreenWithoutTile = 0,
				FullScreenWithTile,
				Free,
			} windowShowMode = EWindowMode::FullScreenWithTile;

			bool bResizable = true;
			uint32_t initWidth  = 800U;
			uint32_t initHeight = 480U;
		};

        [[nodiscard]] bool init(InitConfig initConfig);

        void loop();
        [[nodiscard]] bool release();

		// Hooks for init functions.
		[[nodiscard]] DelegateHandle registerInitBody(std::function<bool(const GLFWWindows* windows)> function);
		void unregisterInitBody(DelegateHandle delegate);

		// Hooks for loop functions.
		[[nodiscard]] DelegateHandle registerLoopBody(std::function<bool(const GLFWWindows* windows, const ApplicationTickData&)> function);
		void unregisterLoopBody(DelegateHandle delegate);

		// Hooks for release functions.
		[[nodiscard]] DelegateHandle registerReleaseBody(std::function<bool(const GLFWWindows* windows)> function);
		void unregisterReleaseBody(DelegateHandle delegate);

		// Hooks for windows closest event.
		[[nodiscard]] DelegateHandle registerClosedEventBody(std::function<bool(const GLFWWindows* windows)> function);
		void unregisterClosedEventBody(DelegateHandle delegate);

		GLFWwindow* getGLFWWindowHandle() const { return m_data.window; }
		const auto& getData() const { return m_data; }
		const auto& getName() const { return m_name; }

		void close() { m_data.bShouldRun = false; }

    private:
		std::string m_name;
		GLFWWindowData m_data;

		MulticastDelegate<const GLFWWindows*, bool&> m_initBodies;
		MulticastDelegate<const GLFWWindows*, const ApplicationTickData&, bool&> m_loopBodies;
		MulticastDelegate<const GLFWWindows*, bool&> m_releaseBodies;

		MulticastDelegate<const GLFWWindows*, bool&> m_closedEventBodies;
    };
}