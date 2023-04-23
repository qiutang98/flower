#include "ui.h"

namespace engine
{
	bool ui::drawVector3(const std::string& label, math::vec3& values, const math::vec3& resetValue, float labelWidth)
	{
		constexpr const char* kResetIcon = ICON_FA_REPLY;

		const auto srcData = values;

		ImGuiIO& io = ImGui::GetIO();
		ImGui::PushID(label.c_str());



		float lineHeight = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
		ImVec2 buttonSize = ImVec2(lineHeight, lineHeight);

		ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{ 0.0f, 0.0f });
		ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2{ 0.0f, 0.0f });
		ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4{1.0f, 1.0f, 1.0f, 0.04f});
		if (ImGui::BeginTable("Vec3UI", 5, ImGuiTableFlags_Borders))
		{
			ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, labelWidth);
			ImGui::TableSetupColumn("", ImGuiTableColumnFlags_None);
			ImGui::TableSetupColumn("", ImGuiTableColumnFlags_None);
			ImGui::TableSetupColumn("", ImGuiTableColumnFlags_None);
			ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, lineHeight);

			ImGui::TableNextColumn(); // label

			// Center label.
			ImGui::SetCursorPosY(ImGui::GetCursorPosY() + GImGui->Style.FramePadding.y);
			ImGui::Text(label.c_str());

			
			ImGui::TableNextColumn();
			// X
			{
				ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.2f, 0.08f, 0.07f, 1.0f });
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.8f, 0.09f, 0.1f, 1.0f });
				ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.9f, 0.9f, 0.9f, 1.0f });
				if (ImGui::Button("X", buttonSize))
				{
					values.x = resetValue.x;
				}
					
				ImGui::PopStyleColor(3);

				ImGui::SameLine();
				ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x);
				ImGui::DragFloat("##X", &values.x, 0.1f, 0.0f, 0.0f, "%.2f");
				ImGui::PopItemWidth();

			}

			ImGui::TableNextColumn();
			// Y
			{
				ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.07f, 0.2f, 0.08f, 1.0f });
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.08f, 0.8f, 0.1f, 1.0f });
				ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.9f, 0.9f, 0.9f, 1.0f });
				if (ImGui::Button("Y", buttonSize))
				{
					values.y = resetValue.y;
				}

				ImGui::PopStyleColor(3);

				ImGui::SameLine();
				ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x);
				ImGui::DragFloat("##Y", &values.y, 0.1f, 0.0f, 0.0f, "%.2f");
				ImGui::PopItemWidth();
				ImGui::SameLine();
			}
			ImGui::TableNextColumn();
			// Z
			{
				ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.07f, 0.08f, 0.2f, 1.0f });
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.08f, 0.09f, 0.8f, 1.0f });
				ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.9f, 0.9f, 0.9f, 1.0f });
				if (ImGui::Button("Z", buttonSize))
				{
					values.z = resetValue.z;
				}
					
				ImGui::PopStyleColor(3);

				ImGui::SameLine();
				ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x);
				ImGui::DragFloat("##Z", &values.z, 0.1f, 0.0f, 0.0f, "%.2f");
				ImGui::PopItemWidth();
				ImGui::SameLine();
			}

			ImGui::TableNextColumn();
			// Reset.
			ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 1.0f, 1.0f, 1.0f, 0.075f });
			if (ImGui::Button(kResetIcon))
			{
				values.x = resetValue.x;
				values.y = resetValue.y;
				values.z = resetValue.z;
			}
			ImGui::PopStyleColor();

			ImGui::EndTable();
		}
		ImGui::PopStyleColor();
		ImGui::PopStyleVar(2);
		ImGui::PopID();
		return srcData != values;

		return false;
	}

	static ImVector<ImRect> s_GroupPanelLabelStack;

	void ui::beginGroupPanel(const char* name, const ImVec2& size)
	{
		ImGui::BeginGroup();


		auto cursorPos = ImGui::GetCursorScreenPos();
		auto itemSpacing = ImGui::GetStyle().ItemSpacing;
		ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 0.0f));
		ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0.0f));

		auto frameHeight = ImGui::GetFrameHeight();
		ImGui::BeginGroup();

		ImVec2 effectiveSize = size;
		if (size.x < 0.0f)
			effectiveSize.x = ImGui::GetContentRegionAvail().x;
		else
			effectiveSize.x = size.x;
		ImGui::Dummy(ImVec2(effectiveSize.x, 0.0f));

		ImGui::Dummy(ImVec2(frameHeight * 0.5f, 0.0f));
		ImGui::SameLine(0.0f, 0.0f);
		ImGui::BeginGroup();
		ImGui::Dummy(ImVec2(frameHeight * 0.5f, 0.0f));
		ImGui::SameLine(0.0f, 0.0f);
		ImGui::TextUnformatted(name);
		auto labelMin = ImGui::GetItemRectMin();
		auto labelMax = ImGui::GetItemRectMax();
		ImGui::SameLine(0.0f, 0.0f);
		ImGui::Dummy(ImVec2(0.0, frameHeight + itemSpacing.y));
		ImGui::BeginGroup();

		ImGui::PopStyleVar(2);

		ImGui::GetCurrentWindow()->ContentRegionRect.Max.x -= frameHeight * 0.5f;
		ImGui::GetCurrentWindow()->WorkRect.Max.x -= frameHeight * 0.5f;
		ImGui::GetCurrentWindow()->InnerRect.Max.x -= frameHeight * 0.5f;
		ImGui::GetCurrentWindow()->Size.x -= frameHeight;

		auto itemWidth = ImGui::CalcItemWidth();
		ImGui::PushItemWidth(ImMax(0.0f, itemWidth - frameHeight));

		s_GroupPanelLabelStack.push_back(ImRect(labelMin, labelMax));
	}

	void ui::endGroupPanel()
	{
		ImGui::PopItemWidth();

		auto itemSpacing = ImGui::GetStyle().ItemSpacing;

		ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 0.0f));
		ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0.0f));

		auto frameHeight = ImGui::GetFrameHeight();

		ImGui::EndGroup();

		ImGui::EndGroup();

		ImGui::SameLine(0.0f, 0.0f);
		ImGui::Dummy(ImVec2(frameHeight * 0.5f, 0.0f));
		ImGui::Dummy(ImVec2(0.0, frameHeight - frameHeight * 0.5f - itemSpacing.y));

		ImGui::EndGroup();

		auto itemMin = ImGui::GetItemRectMin();
		auto itemMax = ImGui::GetItemRectMax();

		auto labelRect = s_GroupPanelLabelStack.back();
		s_GroupPanelLabelStack.pop_back();

		ImVec2 halfFrame = ImVec2(frameHeight * 0.25f * 0.5f, frameHeight * 0.5f);
		ImRect frameRect = ImRect(
			ImVec2{ itemMin.x + halfFrame.x, itemMin.y + halfFrame.y },
			ImVec2{ itemMax.x - halfFrame.x, itemMax.y });
		labelRect.Min.x -= itemSpacing.x;
		labelRect.Max.x += itemSpacing.x;
		for (int i = 0; i < 4; ++i)
		{
			switch (i)
			{
			// left half-plane
			case 0: ImGui::PushClipRect(ImVec2(-FLT_MAX, -FLT_MAX), ImVec2(labelRect.Min.x, FLT_MAX), true); break;
			// right half-plane
			case 1: ImGui::PushClipRect(ImVec2(labelRect.Max.x, -FLT_MAX), ImVec2(FLT_MAX, FLT_MAX), true); break;
			// top
			case 2: ImGui::PushClipRect(ImVec2(labelRect.Min.x, -FLT_MAX), ImVec2(labelRect.Max.x, labelRect.Min.y), true); break;
			// bottom
			case 3: ImGui::PushClipRect(ImVec2(labelRect.Min.x, labelRect.Max.y), ImVec2(labelRect.Max.x, FLT_MAX), true); break;
			}

			ImGui::GetWindowDrawList()->AddRect(
				frameRect.Min, frameRect.Max,
				ImColor(ImGui::GetStyleColorVec4(ImGuiCol_Border)),
				halfFrame.x);

			ImGui::PopClipRect();
		}

		ImGui::PopStyleVar(2);

		ImGui::GetCurrentWindow()->ContentRegionRect.Max.x += frameHeight * 0.5f;
		ImGui::GetCurrentWindow()->WorkRect.Max.x += frameHeight * 0.5f;
		ImGui::GetCurrentWindow()->InnerRect.Max.x += frameHeight * 0.5f;
		ImGui::GetCurrentWindow()->Size.x += frameHeight;

		ImGui::Dummy(ImVec2(0.0f, 0.0f));

		ImGui::EndGroup();
	}
}