#pragma once
#include "UICommon.h"

namespace Flower::UIHelper
{
	extern void drawVector3(
		const std::string& label,
		glm::vec3& values,
		const glm::vec3& resetValue,
		float labelWidth);

	extern void helpMarker(const char* desc);

	extern void hoverTip(const char* desc);
}