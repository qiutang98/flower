#pragma once

#include "Engine.h"
#include "../NativeFileDialog/include/nfd.h"

namespace Flower
{
	struct LauncherInfo
	{
		bool bResizeable = true;

		std::optional<uint32_t> initWidth;
		std::optional<uint32_t> initHeight;

		std::optional<std::string> titleName;
	};

	class Launcher
	{
	public:
		static MulticastDelegate<const LauncherInfo&> preInitHookFunction;
		static MulticastDelegate<> initHookFunction;
		static MulticastDelegate<const EngineTickData&> tickFunction;
		static MulticastDelegate<> releaseHookFunction;

	public:
		static bool preInit(const LauncherInfo& info = {});
		static bool init();
		static void guardedMain();
		static void release();

		static void setWindowTileName(const char* projectName, const char* sceneName);
	};
}