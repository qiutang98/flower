#pragma once

#include <imgui/region_string.h>

inline std::string combineIcon(const engine::ui::RegionStringInit& name, const std::string& icon)
{
	return std::format("  {}  {}", icon, name.getActiveRegionValue());
}