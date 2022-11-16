#pragma once
#include "RHICommon.h"

namespace Flower
{
    struct TimeStamp
    {
        std::string label;
        float microseconds;
    };

    class GPUTimestamps
    {
    public:
        void init(uint32_t numberOfBackBuffers);
        void release();

        void getTimeStamp(VkCommandBuffer cmd, const char* label);
        void getTimeStampUser(TimeStamp ts);
        void onBeginFrame(VkCommandBuffer cmd, std::vector<TimeStamp>* pTimestamp);
        void onEndFrame();

    private:
        static const uint32_t m_maxValuesPerFrame = 128;
        VkQueryPool m_queryPool;

        uint32_t m_frame = 0;
        uint32_t m_numberOfBackBuffers = 0;

        std::vector<std::string> m_labels[5];
        std::vector<TimeStamp> m_cpuTimeStamps[5];
    };

    
}