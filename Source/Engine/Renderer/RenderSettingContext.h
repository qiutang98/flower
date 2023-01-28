#pragma once
#include "RendererCommon.h"
#include "Parameters.h"

namespace Flower
{
	struct RenderSetting
	{
		RHI::DisplayMode displayMode = RHI::DisplayMode::DISPLAYMODE_SDR;
	};

	using RenderSettingManager = Singleton<RenderSetting>;
}