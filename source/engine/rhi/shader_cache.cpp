#include "shader_cache.h"
#include "rhi_log.h"
#include "rhi.h"

#include <filesystem>

namespace engine
{
    [[nodiscard]] VkShaderModule createShaderModule(const std::string& filename, VkDevice device)
    {
        auto file = std::ifstream(filename, std::ios::binary);
        if (file.bad())
        {
            LOG_RHI_FATAL("Open shader file: {} failed.", filename);
            return nullptr;
        }

        file.seekg(0, std::ios::end);
        int length = (int)file.tellg();

        VkShaderModule shaderModule;

        std::vector<uint32_t> opcodes = {};

        opcodes.resize((size_t)(length / 4));
        file.seekg(0, std::ios::beg);
        file.read((char*)opcodes.data(), opcodes.size() * 4);

        VkShaderModuleCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = opcodes.size() * 4;
        ci.pCode = opcodes.data();

        RHICheck(vkCreateShaderModule(device, &ci, nullptr, &shaderModule));

        return shaderModule;
    }

    VkShaderModule ShaderCache::getShader(const std::string& path, bool bReload)
    {
        CHECK(std::filesystem::exists(path));

        const bool bExist = m_moduleCache.contains(path);
        if (bExist && bReload)
        {
            releaseModule(m_moduleCache[path]);
        }

        const bool bLoad = (bReload) || (!bExist);

        if (bLoad)
        {
            m_moduleCache[path] = createShaderModule(path, m_context->getDevice());
        }

        return m_moduleCache[path];
    }

    void ShaderCache::init(const VulkanContext* context)
    {
        m_context = context;
    }

    void ShaderCache::release()
    {
        for (auto& shaders : m_moduleCache)
        {
            releaseModule(shaders.second);
        }
        m_moduleCache.clear();
    }

    void ShaderCache::releaseModule(VkShaderModule shader)
    {
        vkDestroyShaderModule(m_context->getDevice(), shader, nullptr);
    }
}