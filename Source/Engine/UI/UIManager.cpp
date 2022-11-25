#include "Pch.h"
#include "UIManager.h"
#include "../RHI/RHI.h"

namespace Flower
{
	constexpr const char* CFontName = "lato-semi-bold.ttf";
	static AutoCVarFloat cVarUIFontScale(
		"r.ImGui.FontScale",
		"ImGui font scale.",
		"ImGui",
		1.0f,
		CVarFlags::ReadAndWrite
	);

	static AutoCVarFloat cVarUIFontSize(
		"r.ImGui.FontSize",
		"ImGui font size.",
		"ImGui",
		16.0f,
		CVarFlags::ReadOnly | CVarFlags::InitOnce
	);

	static AutoCVarFloat cVarUIIconSize(
		"r.ImGui.IconSize",
		"ImGui icon size.",
		"ImGui",
		13.0f,
		CVarFlags::ReadOnly | CVarFlags::InitOnce
	);

	static AutoCVarString cVarFontFolderPath(
		"r.ImGui.FontFolderPath",
		"ImGui font folder path.",
		"ImGui",
		"./Font/",
		CVarFlags::ReadOnly | CVarFlags::InitOnce
	);

	void setupStyle()
	{
		ImGui::StyleColorsDark();

		ImGuiStyle& style = ImGui::GetStyle();
		ImVec4* colors = ImGui::GetStyle().Colors;

		colors[ImGuiCol_BorderShadow] = ImVec4(0.1f, 0.1f, 0.0f, 0.39f);
		style.WindowBorderSize = 1;
		style.ChildBorderSize = 1;
		style.PopupBorderSize = 1;
		style.FrameBorderSize = 1;
		style.TabBorderSize = 1;
		style.WindowRounding = 0;
		style.ChildRounding = 0;
		style.FrameRounding = 3;
		style.PopupRounding = 0;
		style.ScrollbarRounding = 0;
		style.GrabRounding = 0;
		style.LogSliderDeadzone = 0;
		style.TabRounding = 0;

		{
			style.Colors[ImGuiCol_Text] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
			style.Colors[ImGuiCol_TextDisabled] = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
			//style.Colors[ImGuiCol_WindowBg] = ImVec4(0.13f, 0.14f, 0.15f, 1.00f);
			//style.Colors[ImGuiCol_ChildBg] = ImVec4(0.13f, 0.14f, 0.15f, 1.00f);
			//style.Colors[ImGuiCol_PopupBg] = ImVec4(0.13f, 0.14f, 0.15f, 1.00f);
			style.Colors[ImGuiCol_Border] = ImVec4(0.43f, 0.43f, 0.50f, 0.50f);
			style.Colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
			style.Colors[ImGuiCol_FrameBg] = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
			style.Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.38f, 0.38f, 0.38f, 1.00f);
			style.Colors[ImGuiCol_FrameBgActive] = ImVec4(0.67f, 0.67f, 0.67f, 0.39f);
			style.Colors[ImGuiCol_TitleBg] = ImVec4(0.08f, 0.08f, 0.09f, 1.00f);
			style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.08f, 0.08f, 0.09f, 1.00f);
			style.Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.00f, 0.00f, 0.00f, 0.51f);
			style.Colors[ImGuiCol_MenuBarBg] = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
			style.Colors[ImGuiCol_ScrollbarBg] = ImVec4(0.02f, 0.02f, 0.02f, 0.53f);
			style.Colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.31f, 0.31f, 0.31f, 1.00f);
			style.Colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.41f, 0.41f, 0.41f, 1.00f);
			style.Colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.51f, 0.51f, 0.51f, 1.00f);
			style.Colors[ImGuiCol_CheckMark] = ImVec4(0.11f, 0.64f, 0.92f, 1.00f);
			style.Colors[ImGuiCol_SliderGrab] = ImVec4(0.11f, 0.64f, 0.92f, 1.00f);
			style.Colors[ImGuiCol_SliderGrabActive] = ImVec4(0.08f, 0.50f, 0.72f, 1.00f);
			style.Colors[ImGuiCol_Button] = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
			style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.38f, 0.38f, 0.38f, 1.00f);
			style.Colors[ImGuiCol_ButtonActive] = ImVec4(0.67f, 0.67f, 0.67f, 0.39f);
			style.Colors[ImGuiCol_Header] = ImVec4(0.22f, 0.22f, 0.22f, 1.00f);
			style.Colors[ImGuiCol_HeaderHovered] = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
			style.Colors[ImGuiCol_HeaderActive] = ImVec4(0.67f, 0.67f, 0.67f, 0.39f);
			style.Colors[ImGuiCol_Separator] = style.Colors[ImGuiCol_Border];
			style.Colors[ImGuiCol_SeparatorHovered] = ImVec4(0.41f, 0.42f, 0.44f, 1.00f);
			style.Colors[ImGuiCol_SeparatorActive] = ImVec4(0.26f, 0.59f, 0.98f, 0.95f);
			style.Colors[ImGuiCol_ResizeGrip] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
			style.Colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.29f, 0.30f, 0.31f, 0.67f);
			style.Colors[ImGuiCol_ResizeGripActive] = ImVec4(0.26f, 0.59f, 0.98f, 0.95f);
			style.Colors[ImGuiCol_Tab] = ImVec4(0.08f, 0.08f, 0.09f, 0.83f);
			style.Colors[ImGuiCol_TabHovered] = ImVec4(0.33f, 0.34f, 0.36f, 0.83f);
			style.Colors[ImGuiCol_TabActive] = ImVec4(0.23f, 0.23f, 0.24f, 1.00f);
			style.Colors[ImGuiCol_TabUnfocused] = ImVec4(0.08f, 0.08f, 0.09f, 1.00f);
			style.Colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.13f, 0.14f, 0.15f, 1.00f);
			style.Colors[ImGuiCol_DockingPreview] = ImVec4(0.26f, 0.59f, 0.98f, 0.70f);
			style.Colors[ImGuiCol_DockingEmptyBg] = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
			style.Colors[ImGuiCol_PlotLines] = ImVec4(0.61f, 0.61f, 0.61f, 1.00f);
			style.Colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
			style.Colors[ImGuiCol_PlotHistogram] = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
			style.Colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
			style.Colors[ImGuiCol_TextSelectedBg] = ImVec4(0.26f, 0.59f, 0.98f, 0.35f);
			style.Colors[ImGuiCol_DragDropTarget] = ImVec4(0.11f, 0.64f, 0.92f, 1.00f);
			style.Colors[ImGuiCol_NavHighlight] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
			style.Colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
			style.Colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
			style.Colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.35f);

		}

		style.WindowMenuButtonPosition = ImGuiDir_Left;
	}

	void ImGuiContext::init()
	{
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();

		ImGuiIO& io = ImGui::GetIO();
		(void)io;

		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
		// 
		io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
		

		// See static void ImGui_ImplVulkan_CreateWindow(ImGuiViewport* viewport).
		// in ImGuiVulkan.cpp
		// Bad implement :(
		// Exist a lot of warning when surface is R10G10B10A2, maybe i will fix it if i have free time.
		// io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

		CHAR szExeFileName[MAX_PATH];
		GetModuleFileNameA(NULL, szExeFileName, MAX_PATH);

		std::filesystem::path exePath = szExeFileName;

		std::stringstream iniFilePath;
		iniFilePath << "Config/";
		iniFilePath << exePath.filename().replace_extension().string().c_str();
		iniFilePath << "-ImGui.ini";

		static std::string name = iniFilePath.str();
		io.IniFilename = name.c_str();

		setupStyle();
		ImGuiStyle& style = ImGui::GetStyle();

		if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
		{
			style.WindowRounding = 0.0f;
			style.Colors[ImGuiCol_WindowBg].w = 1.0f;
		}

		ImGui_ImplGlfw_InitForVulkan(RHI::get()->getWindow(), true);


		// load font data to memory.
		io.Fonts->AddFontFromFileTTF(
			(cVarFontFolderPath.get() + CFontName).c_str(),
			cVarUIFontSize.get(),
			NULL,
			io.Fonts->GetGlyphRangesChineseFull()
		);

		// load font data to memory.
		{
			// merge in icons from Font Awesome
			static const ImWchar icons_ranges[] = { ICON_MIN_FA, ICON_MAX_FA, 0 };
			ImFontConfig icons_config;
			icons_config.MergeMode = true;
			icons_config.PixelSnapH = true;

			io.Fonts->AddFontFromFileTTF(
				(cVarFontFolderPath.get() + FONT_ICON_FILE_NAME_FAS).c_str(),
				cVarUIIconSize.get(),
				&icons_config,
				icons_ranges
			);
		}
		{
			// merge in icons from Font Awesome bands.
			static const ImWchar icons_ranges[] = { ICON_MIN_FAB, ICON_MAX_FAB, 0 };
			ImFontConfig icons_config;
			icons_config.MergeMode = true;
			icons_config.PixelSnapH = true;

			io.Fonts->AddFontFromFileTTF(
				(cVarFontFolderPath.get() + FONT_ICON_FILE_NAME_FAB).c_str(),
				cVarUIIconSize.get(),
				&icons_config,
				icons_ranges
			);
		}
		io.FontGlobalScale = cVarUIFontScale.get();
	}

	void ImGuiContext::release()
	{
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
	}

	void ImGuiContext::newFrame()
	{
		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
	}

	void ImGuiContext::updateAfterSubmit()
	{
		ImGuiIO& io = ImGui::GetIO();
		(void)io;

		io.FontGlobalScale = cVarUIFontScale.get();

		// Update and Render additional Platform Windows
		if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
		{
			ImGui::UpdatePlatformWindows();
			ImGui::RenderPlatformWindowsDefault();
		}
	}
}