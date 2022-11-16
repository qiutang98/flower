#pragma once
#include "Pch.h"

namespace Flower
{
    class EngineTimer
    {
        friend class Engine;
    private:
        std::chrono::system_clock::time_point m_timePoint{ };
        std::chrono::system_clock::time_point m_startPoint{ };
        std::chrono::duration<float> m_deltaTime{ 0.0f };

        uint64_t m_tickCount = 0;

        float m_fps = 0.0f;
        float m_smoothFps = 0.0f;

        float m_dt = 0.0f; // ms.
        float m_smoothDt = 0.0f;

        uint32_t m_passFrameForSmoothFps = 0;
        float m_passTimeForSmoothFps = 0.0f;

    private:
        void init()
        {
            m_startPoint = std::chrono::system_clock::now();
            m_timePoint = std::chrono::system_clock::now();
        }

        // return true if smooth fps update.
        // false is non update.
        bool tick()
        {
            m_deltaTime = std::chrono::system_clock::now() - m_timePoint;
            m_timePoint = std::chrono::system_clock::now();

            m_dt = m_tickCount == 0 ? 0.0166f : m_deltaTime.count();
            m_smoothDt = computeSmoothDt(m_smoothDt, m_dt);

            m_tickCount ++;

            // Update smooth fps per-second.
            m_passFrameForSmoothFps++;
            m_passTimeForSmoothFps += m_dt;
            const bool bSmoothFpsUpdate = m_passTimeForSmoothFps >= 1.0f;

            if (bSmoothFpsUpdate)
            {
                m_smoothFps = m_passFrameForSmoothFps / m_passTimeForSmoothFps;
                m_passTimeForSmoothFps = 0;
                m_passFrameForSmoothFps = 0;
            }

            // Update fps every frame.
            m_fps = 1.0f / m_dt;

            return bSmoothFpsUpdate;
        }

        // Use lerp function to get smooth time dt.
        float computeSmoothDt(float oldDt, float dt)
        {
            constexpr double framesToAccumulate = 5.0;
            constexpr double deltaFeedback = 1.0 / framesToAccumulate;

            constexpr double fpsMin = 30.0;
            constexpr double deltaMax = 1.0f / fpsMin;

            const double deltaClamped = dt > deltaMax ? deltaMax : dt;
            return (float)(oldDt * (1.0 - deltaFeedback) + deltaClamped * deltaFeedback);
        }

    public:
        // Realtime fps.
        float getFps() const { return m_fps; }

        // Get smooth fps, update per-seconds.
        float getSmoothFps() const { return m_smoothFps; }

        // Realtime delta time from last frame use time.
        float getDt() const { return m_dt; }

        // Get smooth dt, use lerp 5 frames dt to get smooth result.
        float getSmoothDt() const { return m_smoothDt; }

        // Get tick count.
        auto  getTickCount() const { return m_tickCount; }

        float getRuntime() const 
        {
            std::chrono::duration<float> dd = std::chrono::system_clock::now() - m_startPoint;
            return dd.count(); 
        }
    };
}