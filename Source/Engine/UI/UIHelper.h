#pragma once
#include "UICommon.h"

namespace Flower::UIHelper
{
	// return true if data change.
	extern bool drawVector3(
		const std::string& label,
		glm::vec3& values,
		const glm::vec3& resetValue,
		float labelWidth);

	extern bool drawVector4(
		const std::string& label,
		glm::vec4& values,
		const glm::vec4& resetValue,
		float labelWidth);

	extern bool drawFloat(
		const std::string& label,
		float& values,
		const float& resetValue);

	extern void helpMarker(const char* desc);

	extern void hoverTip(const char* desc);
}