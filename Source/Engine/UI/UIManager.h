#pragma once
#include "UICommon.h"

namespace Flower
{
	struct ImGuiContext
	{
		void init();
		void release();
		void newFrame();
		void updateAfterSubmit();
	};

	// UI manager host on renderer, lifetime same with renderer.
	// We alway keep one renderer module, so don't care about singleton problem.
	using UIManager = Singleton<ImGuiContext>;
}