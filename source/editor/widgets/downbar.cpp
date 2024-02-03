#include "downbar.h"

using namespace engine;
using namespace engine::ui;

#pragma warning(disable: 4996)

DownbarWidget::DownbarWidget()
	: WidgetBase("##Downbar", "##Downbar")
{
	m_bShow = false;
}

using TimePoint = std::chrono::system_clock::time_point;
inline std::string downbarSerializeTimePoint(const TimePoint& time, const std::string& format)
{
	std::time_t tt = std::chrono::system_clock::to_time_t(time);
	std::tm tm = *std::localtime(&tt);
	std::stringstream ss;
	ss << std::put_time(&tm, format.c_str());
	return ss.str();
}

bool beginDownBar(float heightScale, const char* name)
{
	ImGuiContext& g = *GImGui;

	ImGuiViewportP* viewport = (ImGuiViewportP*)(void*)ImGui::GetWindowViewport();

	g.NextWindowData.MenuBarOffsetMinVal = ImVec2(g.Style.DisplaySafeAreaPadding.x, ImMax(g.Style.DisplaySafeAreaPadding.y - g.Style.FramePadding.y, 0.0f));
	ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_MenuBar;
	float height = ImGui::GetFrameHeight() * heightScale;
	bool bOpen = ImGui::BeginViewportSideBar(name, viewport, ImGuiDir_Down, height, windowFlags);
	g.NextWindowData.MenuBarOffsetMinVal = ImVec2(0.0f, 0.0f);

	if (bOpen)
	{
		ImGui::BeginMenuBar();
	}	
	else
	{
		ImGui::End();
	}
	return bOpen;
}

void endDownBar()
{
	ImGui::EndMenuBar();

	ImGuiContext& g = *GImGui;
	if (g.CurrentWindow == g.NavWindow && g.NavLayer == ImGuiNavLayer_Main && !g.NavAnyRequest)
	{
		ImGui::FocusTopMostWindowUnderOne(g.NavWindow, NULL);
	}
		
	ImGui::End();
}


void DownbarWidget::draw(bool bNewWindow, const engine::RuntimeModuleTickData& tickData, engine::VulkanContext* context, const std::string& name, ImGuiID ID)
{
	ZoneScoped;

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
	
	ImGui::SetNextWindowViewport(ID);

	if (bNewWindow)
	{
		if (!ImGui::Begin(name.c_str(), &hide, flag))
		{
			ImGui::End();
			return;
		}
	}


	std::string fpsText;
	float fps;
	{
		fps = glm::clamp(tickData.smoothFps, 0.0f, 999.0f);
		std::stringstream ss;
		ss << std::setw(4) << std::left << std::setfill(' ') << std::fixed << std::setprecision(0) << fps;
		fpsText = "Studio Ticking: " + ss.str() + "FPS";
	}

	ImGui::PushStyleColor(ImGuiCol_FrameBg, ImGui::GetStyleColorVec4(ImGuiCol_MenuBarBg));
	ImGui::PushStyleColor(ImGuiCol_Border, ImGui::GetStyleColorVec4(ImGuiCol_MenuBarBg));
	ImGui::PushStyleColor(ImGuiCol_BorderShadow, ImGui::GetStyleColorVec4(ImGuiCol_MenuBarBg));
	if (beginDownBar(0.98f, std::format("##ViewportName: {}", name).c_str()))
	{
		// draw engine info
		{
			TimePoint input = std::chrono::system_clock::now();
			std::string name = downbarSerializeTimePoint(input, "%Y/%m/%d %H:%M:%S");

			static const std::string sDevName = "DarkEngine-Alpha-Ver0.0.0";
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

		ImGui::PopStyleColor(3);
		endDownBar();
	}


	if (bNewWindow)
	{
		ImGui::End();
	}
}

void DownbarWidget::onTick(const RuntimeModuleTickData& tickData, VulkanContext* context)
{
	draw(true, tickData, context, getName(), ImGui::GetMainViewport()->ID);
}