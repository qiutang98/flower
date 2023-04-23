#pragma once

#include <util/util.h>
#include <rhi/rhi.h>

#include <iconFontcppHeaders/IconsFontAwesome6.h>
#include <iconFontcppHeaders/IconsFontAwesome6Brands.h>
#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>

#include <imgui/gizmo/ImGuizmo.h>
#include <imgui/node_editor/imgui_node_editor.h>

#define ICON_NONE "    "

namespace engine::ui
{
	inline void disableLambda(std::function<void()>&& lambda, bool bDisable)
	{
		if (bDisable)
		{
			ImGui::BeginDisabled();
		}

		lambda();

		if (bDisable)
		{
			ImGui::EndDisabled();
		}
	}

	inline void hoverTip(const char* desc)
	{
		if (ImGui::IsItemHovered())
		{
			ImGui::BeginTooltip();
			ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
			ImGui::TextUnformatted(desc);
			ImGui::PopTextWrapPos();
			ImGui::EndTooltip();
		}
	}

	inline bool treeNodeEx(const char* idLabel, const char* showlabel, ImGuiTreeNodeFlags flags)
	{
		ImGuiWindow* window = ImGui::GetCurrentWindow();
		if (window->SkipItems)
			return false;

		return ImGui::TreeNodeBehavior(window->GetID(idLabel), flags, showlabel, NULL);
	}

	inline void helpMarker(const char* desc)
	{
		ImGui::TextDisabled(" (?) ");
		if (ImGui::IsItemHovered())
		{
			ImGui::BeginTooltip();
			ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
			ImGui::TextUnformatted(desc);
			ImGui::PopTextWrapPos();
			ImGui::EndTooltip();
		}
	}

	extern bool drawVector3(const std::string& label, math::vec3& values, const math::vec3& resetValue, float labelWidth);

	extern void beginGroupPanel(const char* name, const ImVec2& size = ImVec2(0.0f, 0.0f));
	extern void endGroupPanel();
}