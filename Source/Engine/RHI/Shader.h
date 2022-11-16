#pragma once
#include "RHICommon.h"

namespace Flower
{
    class ShaderCache
    {
    public:
        VkShaderModule getShader(const std::string& path, bool reload);

        void init();
        void release();

    private:
        std::unordered_map<std::string, VkShaderModule> m_moduleCache;

        void releaseModule(VkShaderModule shader);
    };
}