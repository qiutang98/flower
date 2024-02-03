#pragma once

#include "imgui/imgui.h"
#include "imgui/imgui_manager.h"
#include "imgui/imgui_internal.h"

#include "widget.h"
#include <nameof/nameof.hpp>
#include <iconFontcppHeaders/IconsFontAwesome6Brands.h>
#include <iconFontcppHeaders/IconsFontAwesome6.h>
#include "imgui/ImGuizmo.h"

namespace engine::ui
{
	extern const char* ICON_NONE;

	extern void disableLambda(std::function<void()>&& lambda, bool bDisable);
	extern void hoverTip(const char* desc);
	extern bool treeNodeEx(const char* idLabel, const char* showlabel, ImGuiTreeNodeFlags flags);
	extern void helpMarker(const char* desc);

	// No thread safe.
	extern void beginGroupPanel(const char* name, const ImVec2& size = ImVec2(0.0f, 0.0f));
	// No thread safe.
	extern void endGroupPanel();

	extern bool drawVector3(const std::string& label, math::vec3& values, const math::vec3& resetValue, float labelWidth);

	extern void drawCollapsingHeader(const std::string& name, std::function<void()>&& f);

	template<typename T>
	inline T drawComboEnumSelect(T enumValue, size_t maxCount, const char* name)
	{
		int formatValue = (int)enumValue;

		std::vector<std::string> formatList(maxCount);
		std::vector<const char*> formatListChar(maxCount);
		for (size_t i = 0; i < maxCount; i++)
		{
			std::string prefix = (formatValue == i) ? "  * " : "    ";
			formatList[i] = std::format("{0} {1}", prefix, nameof::nameof_enum(T(i)));
			formatListChar[i] = formatList[i].c_str();
		}
		ImGui::Combo(name, &formatValue, formatListChar.data(), formatListChar.size());

		return T(formatValue);
	}

	class ImGuiPopupSelfManagedOpenState
	{
	public:
		explicit ImGuiPopupSelfManagedOpenState(
			const std::string& titleName,
			ImGuiWindowFlags flags);

		void draw();

		bool open();

	protected:
		virtual void onDraw() { }
		virtual void onClosed() { }

	private:
		ImGuiWindowFlags m_flags;
		std::string m_popupName;
		bool m_bShouldOpenPopup = false;
		bool m_bPopupOpenState = true;

		UUID m_uuid = buildUUID();
	};
}