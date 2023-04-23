#include "rhi.h"

namespace engine
{
	RHILogger::RHILogger(LoggerSystem* loggerRegistry)
	{
		if(loggerRegistry == nullptr)
		{
			loggerRegistry = LoggerSystem::getDefaultLoggerSystem();
		}
		m_logger = loggerRegistry->registerLogger("RHI");
		CHECK(m_logger && "Register RHI logger fail!");
	}

	std::shared_ptr<spdlog::logger> RHILogger::getDefaultLogger()
	{
		static RHILogger defaultLogger(nullptr);
		return defaultLogger.m_logger;
	}
}