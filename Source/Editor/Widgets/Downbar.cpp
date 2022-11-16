#include "Pch.h"
#include "Downbar.h"

#pragma warning (disable: 4996)

using namespace Flower;
using namespace Flower::UI;

const std::string DOWNBAR_GCopyRightIcon = ICON_FA_COPYRIGHT;

WidgetDownbar::WidgetDownbar()
	: Widget("Downbar")
{
	m_bShow = false;
}

using TimePoint = std::chrono::system_clock::time_point;
inline std::string downbaarSerializeTimePoint(const TimePoint& time, const std::string& format)
{
	std::time_t tt = std::chrono::system_clock::to_time_t(time);
	std::tm tm = *std::localtime(&tt);
	std::stringstream ss;
	ss << std::put_time(&tm, format.c_str());
	return ss.str();
}

void WidgetDownbar::onTick(const RuntimeModuleTickData& tickData)
{
	bool hide = true;

	static ImGuiWindowFlags flag =
		ImGuiWindowFlags_NoCollapse |
		ImGuiWindowFlags_NoResize |
		ImGuiWindowFlags_NoMove |
		ImGuiWindowFlags_NoScrollbar |
		ImGuiWindowFlags_NoBackground |
		ImGuiWindowFlags_NoMouseInputs |
		ImGuiWindowFlags_NoFocusOnAppearing |
		ImGuiWindowFlags_NoScrollWithMouse |
		ImGuiWindowFlags_NoBringToFrontOnFocus |
		ImGuiWindowFlags_NoNavInputs |
		ImGuiWindowFlags_NoNavFocus |
		ImGuiWindowFlags_UnsavedDocument |
		ImGuiWindowFlags_NoTitleBar;

	if (!ImGui::Begin(getTile().c_str(), &hide, flag))
	{
		ImGui::End();
		return;
	}

	std::string fpsText;
	float fps;
	{
		fps = glm::clamp(tickData.smoothFps, 0.0f, 999.0f);
		std::stringstream ss;
		ss << std::setw(4) << std::left << std::setfill(' ') << std::fixed << std::setprecision(0) << fps;
		fpsText = "Editor Ticking  " + ss.str() + "FPS";
	}

	if (ImGui::BeginDownBar(1.1f))
	{
		// draw engine info
		{
			TimePoint input = std::chrono::system_clock::now();
			std::string name = downbaarSerializeTimePoint(input, "%Y/%m/%d %H:%M:%S");

			static const std::string sDevName = "FlowerEngine " + DOWNBAR_GCopyRightIcon + " Developing By Qiutang";
			ImGui::Text(sDevName.c_str());
			ImGui::Text(name.c_str());
		}

		const ImVec2 p = ImGui::GetCursorScreenPos();
		float textStartPositionX =
			ImGui::GetCursorPosX() + ImGui::GetColumnWidth() - ImGui::CalcTextSize(fpsText.c_str()).x
			- ImGui::GetScrollX() - 3 * ImGui::GetStyle().ItemSpacing.x;

		float sz = ImGui::GetTextLineHeight() * 0.40f;
		float fpsPointX = textStartPositionX - sz * 1.8f;

		glm::vec4 goodColor = { 0.1f, 0.9f, 0.05f, 1.0f };
		glm::vec4 badColor = { 0.9f, 0.1f, 0.1f,1.0f };

		{
			// prepare fps color state
			constexpr float lerpMax = 120.0f;
			constexpr float lerpMin = 30.0f;

			float lerpFps = (glm::clamp(fps, lerpMin, lerpMax) - lerpMin) / (lerpMax - lerpMin);
			glm::vec4 lerpColorfps = glm::lerp(badColor, goodColor, lerpFps);

			ImVec4 colffps;
			colffps.x = lerpColorfps.x;
			colffps.y = lerpColorfps.y;
			colffps.z = lerpColorfps.z;
			colffps.w = lerpColorfps.w;

			const ImU32 colfps = ImColor(colffps);

			// draw fps state.
			{
				float x = fpsPointX;
				float y = p.y + ImGui::GetFrameHeight() * 0.51f;
				
				ImGui::SetCursorPosX(textStartPositionX);
				ImGui::Text("%s", fpsText.c_str());
				ImGui::Spacing();
				ImGui::GetWindowDrawList()->AddNgonFilled(ImVec2(ImGui::GetCursorScreenPos().x, y), sz, colfps, 12);
			}
		}
		
		ImGui::EndDownBar();
	}

	ImGui::End();
}