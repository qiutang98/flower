#include "engine.h"
#include "framework.h"

namespace engine
{
    bool Engine::init(Framework* framework)
    {
        if (!framework)
        {
            LOG_ERROR("You must pass one valid framework pointer to init engine!");
            return false;

        }

        // Assign framework pointer.
        m_framework = framework;

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
    }

    bool Engine::isConsoleApp() const
    {
        return m_framework->isConsole();
    }

    bool Engine::isWindowApp() const
    {
        return !m_framework->isConsole();
    }
}