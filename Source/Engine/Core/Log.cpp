#include "Pch.h"
#pragma warning (disable: 4996)

#include <spdlog/sinks/stdout_color_sinks.h>
#include "Core.h"

namespace Flower
{
	static AutoCVarInt32 cVarLogFileOutEnable(
		"r.Log.FileOutEnable",
		"Enable log file output. 0 is off, 1 is on.",
		"Log",
		1,
		CVarFlags::ReadOnly
	);

	// Custom log cache sink, use for editor/hub/custom console output. etc.
	template<typename Mutex>
	class LogCacheSink : public spdlog::sinks::base_sink <Mutex>
	{
		friend Logger;
	private:
		MulticastDelegate<std::string, ELogType> m_callbacks;

		static ELogType toLogType(spdlog::level::level_enum level)
		{
			switch (level)
			{
			case spdlog::level::trace:
				return ELogType::Trace;
			case spdlog::level::info:
				return ELogType::Info;
			case spdlog::level::warn:
				return ELogType::Warn;
			case spdlog::level::err:
			case spdlog::level::critical:
				return ELogType::Error;
			default:
				return ELogType::Other;
			}
		}

	protected:
		void sink_it_(const spdlog::details::log_msg& msg) override
		{
			spdlog::memory_buf_t formatted;
			spdlog::sinks::base_sink<Mutex>::formatter_->format(msg, formatted);

			m_callbacks.broadcast(fmt::to_string(formatted), toLogType(msg.level));
		}

		void flush_() override
		{

		}
	};

	DelegateHandle Logger::pushCallback(std::function<void(std::string, ELogType)>&& callback)
	{
		return m_loggerCache->m_callbacks.addLambda(std::move(callback));
	}

	void Logger::popCallback(DelegateHandle& handle)
	{
		m_loggerCache->m_callbacks.remove(handle);
	}

	constexpr auto s_printFormat      = "%^[%H:%M:%S][%l] %n: %v%$";
	constexpr auto s_logFileFormat    =   "[%H:%M:%S][%l] %n: %v";
	constexpr auto s_cachePrintFormat = "%^[%H:%M:%S][%l] %n: %v%$";

	Logger::Logger()
	{
		using TimePoint = std::chrono::system_clock::time_point;
		auto serializeTimePoint = [](const TimePoint& time, const std::string& format)
		{
			std::time_t tt = std::chrono::system_clock::to_time_t(time);
			std::tm tm = *std::localtime(&tt);
			std::stringstream ss;
			ss << std::put_time(&tm, format.c_str());
			return ss.str();
		};

		// basic sinks.
		logSinks.emplace_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());

		// cache sinks.
		m_loggerCache = std::make_shared<LogCacheSink<std::mutex>>();
		logSinks.emplace_back(m_loggerCache);

		// set print format.
		logSinks[0]->set_pattern(s_printFormat);
		logSinks[1]->set_pattern(s_cachePrintFormat);

		if (cVarLogFileOutEnable.get() != 0)
		{
			TimePoint input = std::chrono::system_clock::now();

			std::filesystem::create_directory("Log");
			auto path = "Log/Flower" + serializeTimePoint(input, "%Y-%m-%d %H_%M_%S") + ".log";

			logSinks.emplace_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>(path, true));
			logSinks[logSinks.size() - 1]->set_pattern(s_logFileFormat);
		}

		m_logger = registerLogger("Utils");
	}

	std::shared_ptr<spdlog::logger> Logger::registerLogger(const char* name)
	{
		auto logger = std::make_shared<spdlog::logger>(name, begin(logSinks), end(logSinks));
		spdlog::register_logger(logger);

		logger->set_level(spdlog::level::trace);
		logger->flush_on(spdlog::level::trace);

		return logger;
	}
}
