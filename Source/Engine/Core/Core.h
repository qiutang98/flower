#pragma once

#define MIKU_MAGIC_NUMBER 39

#define MIKU_GREEN_R 57
#define MIKU_GREEN_G 197
#define MIKU_GREEN_B 187

#include "../Pch.h"
#include "Log.h"

#if defined(_DEBUG) || defined(DEBUG)
#define FLOWER_DEBUG
#endif

// Always enable log to help developer and user found hidden error and bug.
#define ENABLE_LOG

#ifdef ENABLE_LOG
#define LOG_TRACE(...) ::Flower::LogSystem::get()->getLogger()->trace   (__VA_ARGS__)
#define LOG_INFO(...)  ::Flower::LogSystem::get()->getLogger()->info    (__VA_ARGS__)
#define LOG_WARN(...)  ::Flower::LogSystem::get()->getLogger()->warn    (__VA_ARGS__)
#define LOG_ERROR(...) ::Flower::LogSystem::get()->getLogger()->error   (__VA_ARGS__)
#define LOG_FATAL(...) ::Flower::LogSystem::get()->getLogger()->critical(__VA_ARGS__); throw std::runtime_error("Utils fatal!")
#else
#define LOG_TRACE(...)   
#define LOG_INFO (...)    
#define LOG_WARN(...)   
#define LOG_ERROR(...)    
#define LOG_FATAL(...) throw std::runtime_error("Utils fatal!")
#endif

#define CHECK(x) { if(!(x)) { LOG_FATAL("Check error. {0}, {1}.", __LINE__, __FILE__); __debugbreak(); } }
#define ASSERT(x, ...) { if(!(x)) { LOG_FATAL("Assert failed: {0}, {1}, {2}.",__LINE__, __FILE__, __VA_ARGS__); __debugbreak(); } }
#define CHECK_ENTRY() CHECK(false && "No entry handle here, fix me!")

#include "CVar.h"
#include "NonCopyable.h"
#include "Singleton.h"
#include "Delegates.h"
#include "ThreadPool.h"
#include "UUID.h"