#include "framework.h"
#include "macro.h"

#include <chrono>
#include <stb/stb_image.h>
#include <filesystem>

namespace engine
{
    Framework* Framework::get()
    {
        static Framework framework;
        return &framework;
    }

    void Framework::initFramework(Config config)
    {
        // Init state check.
        ASSERT(!m_bInit, "Only allow init framework once!");
        m_bInit = true;

        configInit(config);
    }

    void Framework::configInit(const Config& in)
    {
        // Assign config.
        m_config = in;

        // Prepare folder for config/log etc.
        if (!std::filesystem::exists(m_config.logFolder))  std::filesystem::create_directory(m_config.logFolder);
        if (!std::filesystem::exists(m_config.configFolder))  std::filesystem::create_directory(m_config.configFolder);
    }

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

    void Framework::consoleInit()
    {
        ASSERT(m_config.bConsole, "You can't call this function when application is window.");
    }

    void Framework::consoleRelease()
    {

    }

    void Framework::windowInit()
    {
        ASSERT(!m_config.bConsole, "You can't call this function when application is console.");

        const auto& info = m_config.windowInfo;

        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        // Windows resizeable set.
        if (info.bResizeable)
        {
            glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);
        }
        else
        {
            glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
        }

        using ShowModeEnum = Config::InitWindowInfo::EWindowMode;

        const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
        if (info.windowShowMode == ShowModeEnum::FullScreenWithoutTile)
        {
            glfwWindowHint(GLFW_RED_BITS, mode->redBits);
            glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
            glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
            glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
            glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);

            m_data.window = glfwCreateWindow(mode->width, mode->height, m_config.appName.c_str(), nullptr, nullptr);
        }
        else
        {
            auto clampWidth  = std::max(1u, info.initWidth);
            auto clampHeight = std::max(1u, info.initHeight);

            if (info.windowShowMode == ShowModeEnum::FullScreenWithTile)
            {
                glfwWindowHint(GLFW_MAXIMIZED, GL_TRUE);
            }

            m_data.window = glfwCreateWindow(clampWidth, clampHeight, m_config.appName.c_str(), nullptr, nullptr);
            glfwGetWindowSize(m_data.window, &m_data.width, &m_data.height);

            // Center free mode to window.
            if (info.windowShowMode == ShowModeEnum::Free)
            {
                glfwSetWindowPos(m_data.window, (mode->width - m_data.width) / 2, (mode->height - m_data.height) / 2);
            }
        }

        glfwGetWindowSize(m_data.window, &m_data.width, &m_data.height);
        LOG_INFO("Create window size: ({0},{1}).", m_data.width, m_data.height);

        // Register callback functions.
        glfwSetWindowUserPointer(m_data.window, (void*)(&m_data));
        glfwSetFramebufferSizeCallback(m_data.window, resizeCallBack);
        glfwSetMouseButtonCallback(m_data.window, mouseButtonCallback);
        glfwSetCursorPosCallback(m_data.window, mouseMoveCallback);
        glfwSetScrollCallback(m_data.window, scrollCallback);
        glfwSetWindowFocusCallback(m_data.window, windowFocusCallBack);

        // Init icon.
        m_data.icon.pixels = stbi_load(m_config.iconPath.c_str(), &m_data.icon.width, &m_data.icon.height, 0, 4);
        glfwSetWindowIcon(m_data.window, 1, &m_data.icon);
    }

    void Framework::windowRelease()
    {
        // Free icon memory.
        stbi_image_free(m_data.icon.pixels);

        glfwDestroyWindow(m_data.window);
        glfwTerminate();
    }

    bool Framework::init()
    {
        try 
        {
            if(m_config.bConsole)
            {
                consoleInit();
            }
            else
            {
                windowInit();
            }

            m_engine.init(this);
        }
        catch(...)
        {
            LOG_ERROR("Framework init glfw window fail, this is a fatal error, please feedback to developer to fix me!");
            return false;
        }
        

        return true;
    }

    void Framework::loop()
    {
        auto engineLoopBody = [this]()
        {
            EngineTickData tickData{};

            tickData.windowWidth  =  m_data.width;
            tickData.windowHeight =  m_data.height;
            tickData.bLoseFocus   = !m_data.bFocus;
            tickData.bIsMinimized =  m_data.width <= 0 || m_data.height <= 0;

            bool bContinue = true;
            bContinue &= m_engine.tick(tickData);

            return bContinue;
        };

        if(m_config.bConsole)
        {
            while (m_data.bShouldRun)
            {
                m_data.bShouldRun = engineLoopBody();
            }
        }
        else
        {
            while (!glfwWindowShouldClose(m_data.window) && m_data.bShouldRun)
            {
                glfwPollEvents();
                m_data.bShouldRun = engineLoopBody();
            }
        }
    }

    void Framework::release()
    {
        m_engine.release();

        if(m_config.bConsole)
        {
            consoleRelease();
        }
        else
        {
            // Windows release.
            windowRelease();
        }
    }


}
