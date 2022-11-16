#pragma once

#include "ImGuiGlfw.h"

namespace ImGui
{
	IMGUI_API void setWindowIcon(GLFWwindow* window);

	IMGUI_API void BeginGroupPanel(const char* name, const ImVec2& size = ImVec2(0.0f, 0.0f));
	IMGUI_API void EndGroupPanel();
}