#include "Pch.h"

#include "Launcher.h"
#include "WindowData.h"
#include "Engine.h"
#include "Core/Core.h"
#include "RHI/RHI.h"
#include <stb/stb_image.h>

namespace Flower
{
    static AutoCVarInt32 cVarWindowDefaultShowMode(
        "r.Window.ShowMode",
        "Window display mode. 0 is full screen without tile, 1 is full screen with tile, 2 is custom size by r.Window.Width & .Height",
        "Window",
        1,
        CVarFlags::ReadOnly | CVarFlags::InitOnce
    );

    static AutoCVarInt32 cVarWindowDefaultWidth(
        "r.Window.Width",
        "Window default width which only work when r.Window.ShowMode equal 2.",
        "Window",
        720,
        CVarFlags::ReadOnly | CVarFlags::InitOnce
    );

    static AutoCVarInt32 cVarWindowDefaultHeight(
        "r.Window.Height",
        "Window default height which only work when r.Window.ShowMode equal 2.",
        "Window",
        480,
        CVarFlags::ReadOnly | CVarFlags::InitOnce
    );

    static AutoCVarString cVarTileName(
        "r.Window.TileName",
        "Window tile name.",
        "Window",
        "Flower",
        CVarFlags::ReadOnly | CVarFlags::InitOnce
    );

    MulticastDelegate<const LauncherInfo&> Launcher::preInitHookFunction;
    MulticastDelegate<> Launcher::initHookFunction;
    MulticastDelegate<const EngineTickData&> Launcher::tickFunction;
    MulticastDelegate<> Launcher::releaseHookFunction;

    enum class EWindowShowMode : int32_t
    {
        Min = -1,
        FullScreenWithoutTile = 0,
        FullScreenWithTile = 1,
        Free = 2,
        Max,
    };

    EWindowShowMode getWindowShowMode()
    {
        int32_t type = glm::clamp(cVarWindowDefaultShowMode.get(), (int32_t)EWindowShowMode::Min + 1, (int32_t)EWindowShowMode::Max - 1);
        return EWindowShowMode(type);
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

    void GLFWInit(const LauncherInfo& info)
    {
        GLFWWindowData* windowData = GLFWWindowData::get();

        const bool bValidInSize = info.initWidth.has_value() && info.initHeight.has_value();
        const bool bValidTile = info.titleName.has_value();

        std::string finalTileName = bValidTile ? info.titleName.value() : cVarTileName.get();

        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        if (info.bResizeable)
        {
            glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);
        }
        else
        {
            glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
        }

        EWindowShowMode showMode = getWindowShowMode();

        if (bValidInSize)
        {
            showMode = EWindowShowMode::Free;
        }

        uint32_t width;
        uint32_t height;
        if (showMode == EWindowShowMode::FullScreenWithoutTile)
        {
            const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());

            width = mode->width;
            height = mode->height;

            glfwWindowHint(GLFW_RED_BITS, mode->redBits);
            glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
            glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
            glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);

            windowData->m_window = glfwCreateWindow(width, height, finalTileName.c_str(), glfwGetPrimaryMonitor(), nullptr);

            glfwSetWindowPos(windowData->m_window, (mode->width - width) / 2, (mode->height - height) / 2);

        }
        else if (showMode == EWindowShowMode::FullScreenWithTile)
        {
            const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
            width = mode->width;
            height = mode->height;
            glfwWindowHint(GLFW_MAXIMIZED, GL_TRUE);
            windowData->m_window = glfwCreateWindow(width, height, finalTileName.c_str(), nullptr, nullptr);
        }
        else if (showMode == EWindowShowMode::Free)
        {
            if (bValidInSize)
            {
                width = std::max(10, (int32_t)info.initWidth.value());
                height = std::max(10, (int32_t)info.initHeight.value());
            }
            else
            {
                width = std::max(10, cVarWindowDefaultWidth.get());
                height = std::max(10, cVarWindowDefaultHeight.get());
            }

            const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());

            windowData->m_window = glfwCreateWindow(width, height, finalTileName.c_str(), nullptr, nullptr);
            glfwSetWindowPos(windowData->m_window, (mode->width - width) / 2, (mode->height - height) / 2);
        }
        else
        {
            LOG_FATAL("Unknown windowShowMode: {0}.", (int32_t)showMode);
        }
        LOG_INFO("Create window size: ({0},{1}).", width, height);
        GLFWWindowData::get()->m_width = width;
        GLFWWindowData::get()->m_height = height;

        // Register callback functions.
        glfwSetWindowUserPointer(windowData->m_window, windowData);
        glfwSetFramebufferSizeCallback(windowData->m_window, resizeCallBack);
        glfwSetMouseButtonCallback(windowData->m_window, mouseButtonCallback);
        glfwSetCursorPosCallback(windowData->m_window, mouseMoveCallback);
        glfwSetScrollCallback(windowData->m_window, scrollCallback);
        glfwSetWindowFocusCallback(windowData->m_window, windowFocusCallBack);

        ImGui::setWindowIcon(windowData->m_window);
    }

    void GLFWRelease()
    {
        glfwDestroyWindow(GLFWWindowData::get()->getWindow());
        glfwTerminate();
    }

    bool Launcher::preInit(const LauncherInfo& info)
    {
        GLFWInit(info);


        EnginePreInitInfo engineInitInfo{ };
        engineInitInfo.window = GLFWWindowData::get()->getWindow();

        GEngine->preInit(engineInitInfo);

        preInitHookFunction.broadcast(info);
        return true;
    }

    bool Launcher::init()
    {
        GEngine->init();
        initHookFunction.broadcast();
        return true;
    }

    void Launcher::guardedMain()
    {
        while (!glfwWindowShouldClose(GLFWWindowData::get()->getWindow()) && GLFWWindowData::get()->shouldRun())
        {
            glfwPollEvents();

            EngineTickData tickData{};

            tickData.windowWidth = GLFWWindowData::get()->getWidth();
            tickData.windowHeight = GLFWWindowData::get()->getHeight();
            tickData.bLoseFocus = !GLFWWindowData::get()->isFocus();
            tickData.bIsMinimized = tickData.windowWidth <= 0 || tickData.windowHeight <= 0;

            GLFWWindowData::get()->setShouldRun(GEngine->tick(tickData));

            tickFunction.broadcast(tickData);
        }
    }

    void Launcher::release()
    {
        // Sleep 1s before release.
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        vkDeviceWaitIdle(RHI::Device);

        releaseHookFunction.broadcast();

        GEngine->release();


        // GLFW release.
        GLFWRelease();

        // Sleep 1s before close.
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    void Launcher::setWindowTileName(const char* projectName, const char* sceneName)
    {
        std::stringstream finalTileName;
        finalTileName << cVarTileName.get();
        finalTileName << " - ";
        finalTileName << projectName;
        finalTileName << " - ";
        finalTileName << sceneName;

        glfwSetWindowTitle(GLFWWindowData::get()->getWindow(), finalTileName.str().c_str());
    }
}

