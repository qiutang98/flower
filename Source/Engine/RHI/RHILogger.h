#pragma once

#include "../Core/Core.h"
#include <vma/vk_mem_alloc.h>

namespace Flower
{
	struct RHILogger
	{
		std::shared_ptr<spdlog::logger> logger = nullptr;

		RHILogger()
		{
			logger = LogSystem::get()->registerLogger("RHI");
			CHECK(logger && "Register RHI logger fail!");
		}
	};
	using RHILoggerSystem = Singleton<RHILogger>;
}

#ifdef ENABLE_LOG
#define LOG_RHI_TRACE(...) ::Flower::RHILoggerSystem::get()->logger->trace   (__VA_ARGS__)
#define LOG_RHI_INFO(...)  ::Flower::RHILoggerSystem::get()->logger->info    (__VA_ARGS__)
#define LOG_RHI_WARN(...)  ::Flower::RHILoggerSystem::get()->logger->warn    (__VA_ARGS__)
#define LOG_RHI_ERROR(...) ::Flower::RHILoggerSystem::get()->logger->error   (__VA_ARGS__)
#define LOG_RHI_FATAL(...) ::Flower::RHILoggerSystem::get()->logger->critical(__VA_ARGS__); throw std::runtime_error("RHI fatal!")
#else
#define LOG_RHI_TRACE(...)   
#define LOG_RHI_INFO(...)    
#define LOG_RHI_WARN(...)   
#define LOG_RHI_ERROR(...)    
#define LOG_RHI_FATAL(...) throw std::runtime_error("RHI fatal!")
#endif

namespace Flower
{
	inline void RHICheck(VkResult err)
	{
		if (err)
		{
			LOG_RHI_FATAL("check error: {}.", toString(err));
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