#pragma once

#include "../Pch.h"

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

#include "NonCopyable.h"
#include "Singleton.h"

namespace Flower
{
	// Custom log cache sink.
	template<typename Mutex> class LogCacheSink;

	class DelegateHandle;

	enum class ELogType : uint8_t
	{
		Trace = 0,
		Info,
		Warn, 
		Error,
		Other,

		Max,
	};

	class Logger : private NonCopyable
	{
	private:
		std::vector<spdlog::sink_ptr> logSinks { };

		// Logger for common.
		std::shared_ptr<spdlog::logger> m_logger;

		// Logger cache for custom logger.
		std::shared_ptr<LogCacheSink<std::mutex>> m_loggerCache;

	public:
		Logger();

		inline auto& getLogger() { return m_logger; }

		// push callback to logger sink.
		[[nodiscard]] DelegateHandle pushCallback(std::function<void(std::string, ELogType)>&& callback);

		// pop callback from logger sink.
		void popCallback(DelegateHandle& name);

		// register a new logger.
		[[nodiscard]] std::shared_ptr<spdlog::logger> registerLogger(const char* name);
	};

	using LogSystem = Singleton<Logger>;
}
