#include "engine.h"
#include "framework.h"
#include <util/openal.h>

namespace engine
{
    bool Engine::soundEngineValid() const
    {
        return m_openALDevice && m_openALContext && (m_contextMadeCurrent == ALC_TRUE);
    }

    bool Engine::initSoundEngine()
    {
        // Init openal.
        m_openALDevice = alcOpenDevice(nullptr);
        if (!m_openALDevice)
        {
            return false;
        }

        if (!alcCall(alcCreateContext, m_openALContext, m_openALDevice, m_openALDevice, nullptr) || !m_openALContext)
        {
            LOG_ERROR("ERROR: Could not create audio context.");
            return false;
        }

        if (!alcCall(alcMakeContextCurrent, m_contextMadeCurrent, m_openALDevice, m_openALContext)
            || m_contextMadeCurrent != ALC_TRUE)
        {
            LOG_ERROR("ERROR: Could not make audio context current.");
            return false;
        }
    }

    void Engine::closedSoundEngine()
    {
        alcCall(alcMakeContextCurrent, m_contextMadeCurrent, m_openALDevice, nullptr);
        alcCall(alcDestroyContext, m_openALDevice, m_openALContext);

        ALCboolean closed;
        alcCall(alcCloseDevice, closed, m_openALDevice, m_openALDevice);
    }

    bool Engine::init(Framework* framework)
    {
        if (!framework)
        {
            LOG_ERROR("You must pass one valid framework pointer to init engine!");
            return false;

        }

        // Assign framework pointer.
        m_framework = framework;

        initSoundEngine();



        // Engine timer init, use 5 frame to smooth fps and dt, use 5.0 as min fps to compute smooth time.
        m_timer.init(5.0, 5.0);

        // Init runtime module by registered order.
        if (m_runtimeModules.size() > 0)
        {
            bool bAllModuleReady = true;
            for (size_t moduleIndex = 0; moduleIndex < m_runtimeModules.size(); moduleIndex++)
            {
                if (!m_runtimeModules[moduleIndex]->init())
                {
                    LOG_ERROR("Runtime module {0} failed to init.", typeid(*m_runtimeModules[moduleIndex]).name());
                    bAllModuleReady = false;
                }
            }

            if (!bAllModuleReady)
            {
                LOG_ERROR("No all module init correctly, check prev logs to fix the bugs.");
                return false;
            }
        }

        // Everything init, no error so return true.
        return true;
    }

    bool Engine::tick(const EngineTickData& data)
    {
        bool bContinue = true;

        if (m_runtimeModules.size() > 0)
        {
            // Update engine timer.
            const bool bSmoothFpsUpdate = m_timer.tick();

            // Prepare module update tick data.
            RuntimeModuleTickData tickData { };
            tickData.windowWidth      = data.windowWidth;
            tickData.windowHeight     = data.windowHeight;
            tickData.bLoseFocus       = data.bLoseFocus;
            tickData.bIsMinimized     = data.bIsMinimized;
            tickData.deltaTime        = m_timer.getDt();
            tickData.smoothDeltaTime  = m_timer.getSmoothDt();
            tickData.fps              = m_timer.getFps();
            tickData.smoothFps        = m_timer.getSmoothFps();
            tickData.tickCount        = m_timer.getTickCount();
            tickData.bSmoothFpsUpdate = bSmoothFpsUpdate;
            tickData.runTime          = m_timer.getRuntime();


            if (m_bRuningGame)
            {
                m_gameTime += tickData.deltaTime;
            }

            tickData.gameTime =  m_gameTime;

            for (const auto& runtimeModule : m_runtimeModules)
            {
                bContinue &= runtimeModule->tick(tickData);
            }
        }

        return bContinue;
    }

    void Engine::release()
    {
        // Release runtime modules.
        if (m_runtimeModules.size() > 0)
        {
            // Release from end to start.
            for (size_t i = m_runtimeModules.size(); i > 0; i--)
            {
                m_runtimeModules[i - 1]->release();
            }

            // Manually destruct from end to start.
            for (size_t i = m_runtimeModules.size() - 1; i > 0; i--)
            {
                m_runtimeModules[i].reset();
            }

            m_runtimeModules.clear();
        }

        closedSoundEngine();
    }

    bool Engine::isConsoleApp() const
    {
        return m_framework->isConsole();
    }

    bool Engine::isWindowApp() const
    {
        return !m_framework->isConsole();
    }

    void Engine::setGameStart()
    {
        m_bRuningGame = true;
        m_gameTime = 0.0f;

        onGameStart.broadcast();
    }

    void Engine::setGameStop()
    {
        m_bRuningGame = false;
        m_gameTime = 0.0f;
        onGameStop.broadcast();
    }

    void Engine::setGamePause()
    {
        m_bRuningGame = false;
        onGamePause.broadcast();
    }

    void Engine::setGameContinue()
    {
        m_bRuningGame = true;
        onGameContinue.broadcast();
    }
}