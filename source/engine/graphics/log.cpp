#include "log.h"

namespace engine
{
	RHILogger::RHILogger(LoggerSystem* loggerRegistry)
	{
		if (loggerRegistry == nullptr)
		{
			loggerRegistry = LoggerSystem::get();
		}
		m_logger = loggerRegistry->registerLogger("RHI");
		ASSERT(m_logger, "Register RHI logger fail!");
	}

	std::shared_ptr<spdlog::logger> RHILogger::get()
	{
		static RHILogger defaultLogger(nullptr);
		return defaultLogger.m_logger;
	}
}