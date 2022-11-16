#include "Pch.h"
#include "UIHelper.h"
#include <ImGui/ImGuiInternal.h>
namespace Flower
{
	
	void UIHelper::drawVector3(
		const std::string& label,
		glm::vec3& values,
		const glm::vec3& resetValue,
		float labelWidth)
	{
		constexpr const char* RESET_ICON = ICON_FA_REPLY;

		ImGuiIO& io = ImGui::GetIO();
		auto boldFont = io.Fonts->Fonts[0];
		ImGui::PushID(label.c_str());

		float lineHeight = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
		const float sscale = 0.9f;
		ImVec2 buttonSize = { sscale * (lineHeight + 3.0f), sscale * lineHeight };
		ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{ 2.0f, 2.0f });
		if (ImGui::BeginTable("Vec3UI", 5, ImGuiTableFlags_Borders))
		{
			ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, labelWidth);
			ImGui::TableSetupColumn("", ImGuiTableColumnFlags_None);
			ImGui::TableSetupColumn("", ImGuiTableColumnFlags_None);
			ImGui::TableSetupColumn("", ImGuiTableColumnFlags_None);
			ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, ImGui::GetTextLineHeightWithSpacing() * 1.2f);

			ImGui::TableNextColumn(); // label
			ImGui::Text(label.c_str());

			ImGui::TableNextColumn();
			// X
			{
				ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.2f, 0.09f, 0.09f, 1.0f });
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.8f, 0.09f, 0.1f, 1.0f });
				ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.9f, 0.9f, 0.9f, 1.0f });
				ImGui::PushFont(boldFont);
				if (ImGui::Button("X", buttonSize))
					values.x = resetValue.x;
				ImGui::PopFont();
				ImGui::PopStyleColor(3);

				ImGui::SameLine();
				ImGui::DragFloat("##X", &values.x, 0.1f, 0.0f, 0.0f, "%.2f");
			}

			ImGui::TableNextColumn();
			// Y
			{
				ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.08f, 0.2f, 0.1f, 1.0f });
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.08f, 0.8f, 0.1f, 1.0f });
				ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.9f, 0.9f, 0.9f, 1.0f });
				ImGui::PushFont(boldFont);
				if (ImGui::Button("Y", buttonSize))
					values.y = resetValue.y;
				ImGui::PopFont();
				ImGui::PopStyleColor(3);

				ImGui::SameLine();
				ImGui::DragFloat("##Y", &values.y, 0.1f, 0.0f, 0.0f, "%.2f");
				ImGui::SameLine();
			}
			ImGui::TableNextColumn();
			// Z
			{
				ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.08f, 0.09f, 0.2f, 1.0f });
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.08f, 0.09f, 0.8f, 1.0f });
				ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.9f, 0.9f, 0.9f, 1.0f });
				ImGui::PushFont(boldFont);
				if (ImGui::Button("Z", buttonSize))
					values.z = resetValue.z;
				ImGui::PopFont();
				ImGui::PopStyleColor(3);

				ImGui::SameLine();
				ImGui::DragFloat("##Z", &values.z, 0.1f, 0.0f, 0.0f, "%.2f");
				ImGui::SameLine();
			}

			ImGui::TableNextColumn();
			// Reset.
			if (ImGui::Button(RESET_ICON))
			{
				values.x = resetValue.x;
				values.y = resetValue.y;
				values.z = resetValue.z;
			}

			ImGui::EndTable();
		}
		ImGui::PopStyleVar();
		ImGui::PopID();
		return;
	}

	void UIHelper::helpMarker(const char* desc)
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

	void UIHelper::hoverTip(const char* desc)
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
}


