#pragma once

#include <util/framework.h>
#include <util/util.h>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

namespace engine
{
	class VulkanContext;

	class ShaderCache final : NonCopyable
	{
	public:
		void init(const VulkanContext* context);
		VkShaderModule getShader(const std::string& path, bool reload);
		void release();

	private:
		const VulkanContext* m_context;
		std::unordered_map<std::string, VkShaderModule> m_moduleCache;

		void releaseModule(VkShaderModule shader);
	};
}