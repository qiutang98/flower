#pragma once

#include <util/framework.h>
#include <util/util.h>

#include <vulkan/vulkan.h>
#include <sstream>
#include <cstdint>

namespace engine
{
	struct RHILogger
	{
	private:
		explicit RHILogger(LoggerSystem* loggerRegistry);

		std::shared_ptr<spdlog::logger> m_logger = nullptr;

	public:
		static std::shared_ptr<spdlog::logger> getDefaultLogger();
	};
}

#if ENABLE_LOG
	#define LOG_RHI_TRACE(...) { ::engine::RHILogger::getDefaultLogger()->trace   (__VA_ARGS__); }
	#define LOG_RHI_INFO(...)  { ::engine::RHILogger::getDefaultLogger()->info    (__VA_ARGS__); }
	#define LOG_RHI_WARN(...)  { ::engine::RHILogger::getDefaultLogger()->warn    (__VA_ARGS__); }
	#define LOG_RHI_ERROR(...) { ::engine::RHILogger::getDefaultLogger()->error   (__VA_ARGS__); }
	#define LOG_RHI_FATAL(...) { ::engine::RHILogger::getDefaultLogger()->critical(__VA_ARGS__); throw std::runtime_error("RHI fatal!"); }
#else
	#define LOG_RHI_TRACE(...)   
	#define LOG_RHI_INFO(...)    
	#define LOG_RHI_WARN(...)   
	#define LOG_RHI_ERROR(...)    
	#define LOG_RHI_FATAL(...) { throw std::runtime_error("RHI fatal!") }
#endif

namespace engine
{
	inline void RHICheck(VkResult err)
	{
		if (err != VK_SUCCESS)
		{
			LOG_RHI_FATAL("VkResult error: {0}.", int32_t(err));
		}
	}

	inline void RHICheck(decltype(VK_NULL_HANDLE) handle)
	{
		if (handle == VK_NULL_HANDLE)
		{
			LOG_RHI_FATAL("Handle is empty.");
		}
	}
}