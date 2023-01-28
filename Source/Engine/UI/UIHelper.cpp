#include "Pch.h"
#include "UIHelper.h"
#include <ImGui/ImGuiInternal.h>
namespace Flower
{
	
	bool UIHelper::drawVector3(
		const std::string& label,
		glm::vec3& values,
		const glm::vec3& resetValue,
		float labelWidth)
	{
		const auto srcData = values;

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
		return srcData != values;
	}

	bool UIHelper::drawVector4(
		const std::string& label, 
		glm::vec4& values, 
		const glm::vec4& resetValue, 
		float labelWidth)
	{
		const auto srcData = values;

		constexpr const char* RESET_ICON = ICON_FA_REPLY;

		ImGuiIO& io = ImGui::GetIO();
		auto boldFont = io.Fonts->Fonts[0];
		ImGui::PushID(label.c_str());

		float lineHeight = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
		const float sscale = 0.9f;
		ImVec2 buttonSize = { sscale * (lineHeight + 3.0f), sscale * lineHeight };
		ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{ 2.0f, 2.0f });
		if (ImGui::BeginTable("Vec4UI", 6, ImGuiTableFlags_Borders))
		{
			ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, labelWidth);
			ImGui::TableSetupColumn("", ImGuiTableColumnFlags_None);
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
			// W
			{
				ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4{ 0.08f, 0.2f, 0.3f, 1.0f });
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.08f, 0.4f, 0.6f, 1.0f });
				ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4{ 0.9f, 0.9f, 0.9f, 1.0f });
				ImGui::PushFont(boldFont);
				if (ImGui::Button("W", buttonSize))
					values.w = resetValue.w;
				ImGui::PopFont();
				ImGui::PopStyleColor(3);
				ImGui::SameLine();
				ImGui::DragFloat("##W", &values.w, 0.1f, 0.0f, 0.0f, "%.2f");
				ImGui::SameLine();
			}

			ImGui::TableNextColumn();
			// Reset.
			if (ImGui::Button(RESET_ICON))
			{
				values.x = resetValue.x;
				values.y = resetValue.y;
				values.z = resetValue.z;
				values.w = resetValue.w;
			}

			ImGui::EndTable();
		}
		ImGui::PopStyleVar();
		ImGui::PopID();
		return srcData != values;

	}

	bool UIHelper::drawFloat(const std::string& label, float& values, const float& resetValue)
	{
		float srcValue = values;

		ImGui::PushID(label.c_str()); 
	
		ImGui::DragFloat(label.c_str(), &values);

		ImGui::SameLine();

		constexpr const char* RESET_ICON = ICON_FA_REPLY;
		if (ImGui::Button(RESET_ICON))
		{
			values = resetValue;
		}
		ImGui::PopID();
		return srcValue != values;
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


