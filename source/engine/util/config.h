#pragma once
#include <string>

namespace engine
{
	struct Config
	{
		// Runing application name.
		std::string appName = "engine";

		// Run with console state.
		bool bConsole = false;

		// Init window infos.
		struct InitWindowInfo
		{
			// Init window show mode.
			enum class EWindowMode
			{
				FullScreenWithoutTile = 0,
				FullScreenWithTile,
				Free,
			} windowShowMode = EWindowMode::FullScreenWithTile;

			// Window resizeable state.
			bool bResizeable = true;

			// Init windows size.
			uint32_t initWidth  = 800;
			uint32_t initHeight = 480;
		} windowInfo = { };

		// Media path config.
		std::string iconPath = "image/icon.png";

		// Log folder config.
		bool bEnableLogFileOut = false;
		std::string logFolder = "log";

		// Config folder config.
		std::string configFolder = "config";
	};
}