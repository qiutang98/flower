#pragma once

#include "delegate.h"
#include "noncopyable.h"

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

namespace engine
{
	// Custom log cache sink.
	template<typename Mutex> class LogCacheSink;

	enum class ELogType : uint8_t
	{
		Trace = 0,
		Info,
		Warn,
		Error,
		Fatal,
		Other,

		Max,
	};

	class LoggerSystem : private NonCopyable
	{
	public:
		struct InitConfig
		{
			bool bOutputLog = false;
			std::string outputLogPath;
		};

	private:
		explicit LoggerSystem(bool bOutputFile, const std::string& saveFile);

		std::vector<spdlog::sink_ptr> logSinks { };

		// Logger for common.
		std::shared_ptr<spdlog::logger> m_defaultLogger;

		// Logger cache for custom logger.
		std::shared_ptr<LogCacheSink<std::mutex>> m_loggerCache;

		
		static InitConfig m_initConfigs;

	public:
		static void initBasicConfig(const InitConfig& in);

		auto& getDefaultLogger() noexcept { return m_defaultLogger; }

		// push callback to logger sink.
		[[nodiscard]] DelegateHandle pushCallback(std::function<void(const std::string&, ELogType)>&& callback);

		// pop callback from logger sink.
		void popCallback(DelegateHandle& name);

		// register a new logger.
		[[nodiscard]] std::shared_ptr<spdlog::logger> registerLogger(const char* name);

		static LoggerSystem* get();
	};
}