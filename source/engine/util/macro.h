#pragma once

#define ENABLE_LOG    1

#if defined(_DEBUG) || defined(DEBUG)
	#define APP_DEBUG 1
#else
	#define APP_DEBUG 0
#endif

#if ENABLE_LOG
	#include "log.h"
	#define LOG_TRACE(...) { ::engine::LoggerSystem::getDefaultLoggerSystem()->getDefaultLogger()->trace   (__VA_ARGS__); }
	#define LOG_INFO(...)  { ::engine::LoggerSystem::getDefaultLoggerSystem()->getDefaultLogger()->info    (__VA_ARGS__); }
	#define LOG_WARN(...)  { ::engine::LoggerSystem::getDefaultLoggerSystem()->getDefaultLogger()->warn    (__VA_ARGS__); }
	#define LOG_ERROR(...) { ::engine::LoggerSystem::getDefaultLoggerSystem()->getDefaultLogger()->error   (__VA_ARGS__); }
	#define LOG_FATAL(...) { ::engine::LoggerSystem::getDefaultLoggerSystem()->getDefaultLogger()->critical(__VA_ARGS__); throw std::runtime_error("Utils fatal!"); }
#else
	#define LOG_TRACE(...)   
	#define LOG_INFO (...)    
	#define LOG_WARN(...)   
	#define LOG_ERROR(...)    
	#define LOG_FATAL(...) { throw std::runtime_error("Utils fatal!"); }
#endif

#if APP_DEBUG
	#define CHECK(x) { if(!(x)) { LOG_FATAL("Check error, at line {0} on file {1}.", __LINE__, __FILE__); __debugbreak(); } }
	#define ASSERT(x, ...) { if(!(x)) { LOG_FATAL("Assert failed: {2}, at line {0} on file {1}.", __LINE__, __FILE__, __VA_ARGS__); __debugbreak(); } }
#else
	#define CHECK(x) { if(!(x)) { LOG_FATAL("Check error."); } }
	#define ASSERT(x, ...) { if(!(x)) { LOG_FATAL("Assert failed: {0}.", __VA_ARGS__); } }
#endif

#define CHECK_ENTRY() ASSERT(false, "No entry implement here, fix me!")