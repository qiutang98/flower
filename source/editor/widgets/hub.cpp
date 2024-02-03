#include "hub.h"
#include "../editor.h"
#include <asset/asset_manager.h>

#include <fstream>
#include <nfd.h>
#include <scene/scene_manager.h>
using namespace engine;

HubWidget::HubWidget() : WidgetBase("dark projects", "dark projects")
{
	m_bShow = false;
}

void HubWidget::onInit()
{
	m_recentProjectList.load();
}

void HubWidget::onRelease()
{
	m_recentProjectList.save();
}

void HubWidget::onTick(const engine::RuntimeModuleTickData& tickData, engine::VulkanContext* context)
{
	ZoneScoped;

	bool bProjectSelectReady = false;
	std::u16string projectPath;

	ImGui::DockSpaceOverViewport();

	static bool bUseWorkArea = true;
	static ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings;
	const ImGuiViewport* viewport = ImGui::GetMainViewport();

	ImGui::SetNextWindowPos(bUseWorkArea ? viewport->WorkPos : viewport->Pos);
	ImGui::SetNextWindowSize(bUseWorkArea ? viewport->WorkSize : viewport->Size);

	const float footerHeightToReserve =
		ImGui::GetStyle().ItemSpacing.y +
		ImGui::GetFrameHeightWithSpacing();

	ImGui::Begin("ProjectSelectedWindow", &bUseWorkArea, flags);
	{
		ImGui::Indent();

		ImGui::Spacing();
		ImGui::TextDisabled("Develop by qiutang under MIT copyright, feedback and contact with qiutanguu@gmail.com.");

		ImGui::Spacing();
		ImGui::TextDisabled("Dark game engine version 0.0.1 alpha test, project select or create...");

		ImGui::NewLine();

		if (ImGui::Button("Load", { 4.0f * ImGui::GetFontSize(), 0.0f }))
		{
			loadProject();
		}
		ImGui::SameLine();

		if (ImGui::Button("New", { 4.0f * ImGui::GetFontSize(), 0.0f }))
		{
			newProject();
		}
		ImGui::SameLine();

		ImGui::BeginDisabled();
		ImGui::InputText(" ", m_projectPath, kProjectPathSize); ImGui::SameLine();
		ImGui::EndDisabled();

		ui::disableLambda([&]()
		{
			if (ImGui::Button("OK", { 3.0f * ImGui::GetFontSize(), 0.0f }))
			{
				projectPath = utf8::utf8to16(m_projectPath);
				bProjectSelectReady = createOrOpenProject(projectPath);
			}
		}, !m_bProjectPathReady);

		ImGui::NewLine();
		ImGui::Separator();

		ImGui::Spacing();
		ImGui::TextDisabled("History project list create by current engine, we read from ini file in the install path.");
		ImGui::NewLine();

		ImGui::Indent();

		for (size_t i = 0; i < m_recentProjectList.validId; i++)
		{
			const std::string& pathName = m_recentProjectList.utf8Strings[i];
			if (ImGui::Selectable(pathName.c_str()))
			{
				projectPath = m_recentProjectList.recentOpenProjects[i];
				bProjectSelectReady = createOrOpenProject(projectPath);
			}
			ImGui::Spacing();
			ImGui::Spacing();
		}
		ImGui::Unindent();

		ImGui::SetCursorPosY(ImGui::GetCursorPosY() + ImGui::GetContentRegionAvail().y - ImGui::GetTextLineHeightWithSpacing());

		ImGui::Separator();
		ImGui::TextDisabled("Recent open project num: %d.", m_recentProjectList.validId);
	}
	ImGui::End();

	// Project ready, close hub and resizable windows.
	if (bProjectSelectReady)
	{
		setupEditorProject(projectPath);
	}
}

bool HubWidget::loadProject()
{
	std::string readPathString;
	nfdchar_t* readPath = NULL;
	nfdresult_t result = NFD_OpenDialog("dark", NULL, &readPath);
	if (result == NFD_OKAY)
	{
		readPathString = readPath;
		free(readPath);
	}
	else if (result == NFD_CANCEL)
	{
		return false;
	}

	auto u16PathString = utf8::utf8to16(readPathString);
	std::filesystem::path fp{ u16PathString };

	if (!fp.empty())
	{
		if (std::filesystem::exists(fp))
		{
			auto path = utf8::utf16to8(fp.u16string());
			std::copy(path.begin(), path.end(), m_projectPath);
			m_projectPath[readPathString.size()] = '\0';

			m_bProjectPathReady = true;
			return true;
		}
	}

	return false;
}

bool HubWidget::newProject()
{
	std::string path;
	nfdchar_t* outPathChars = NULL;
	nfdresult_t result = NFD_PickFolder(NULL, &outPathChars);

	if (result == NFD_CANCEL)
	{
		return false;
	}
	else if (result == NFD_OKAY)
	{
		path = outPathChars;
		free(outPathChars);
	}

	auto u16PathString = utf8::utf8to16(path);
	std::filesystem::path fp(u16PathString);
	if (!path.empty())
	{
		std::filesystem::path projectName = fp.filename();
		fp /= (projectName.string() + ".dark");

		path = utf8::utf16to8(fp.u16string());
		std::copy(path.begin(), path.end(), m_projectPath);
		m_projectPath[path.size()] = '\0';

		m_bProjectPathReady = true;
		return true;
	}

	return false;
}

bool HubWidget::createOrOpenProject(const std::filesystem::path& projectPath)
{
	// Create project if no exist.
	if (!std::filesystem::exists(projectPath))
	{
		std::ofstream os(projectPath);
	}

	auto projectRoot = projectPath.parent_path();

	std::filesystem::create_directory(projectRoot / "asset");
	std::filesystem::create_directory(projectRoot / "config");
	std::filesystem::create_directory(projectRoot / "log");
	std::filesystem::create_directory(projectRoot / "cache");

	m_recentProjectList.update(projectPath.u16string());
	return true;
}

void HubWidget::setupEditorProject(const std::filesystem::path& path)
{
    ZoneScoped;
	getAssetManager()->setupProject(path.u16string());
	const auto& projectConfig = getAssetManager()->getProjectConfig();

	LOG_TRACE("Start editor with project {}.", utf8::utf16to8(projectConfig.projectName));
	LOG_TRACE("Active project path {}, ative root project path {}.", utf8::utf16to8(projectConfig.projectFilePath), utf8::utf16to8(projectConfig.rootPath));

	Editor::get()->closedHubWidget();

	Editor::get()->onceEventAfterTick.add([]() 
	{
		ZoneScopedN("Maximum Windows");

		// Window resizable.
		glfwSetWindowAttrib(Engine::get()->getGLFWWindows()->getGLFWWindowHandle(), GLFW_RESIZABLE, GL_TRUE);

		// Maximized window.
		glfwMaximizeWindow(Engine::get()->getGLFWWindows()->getGLFWWindowHandle());
	});
}

void RecentOpenProjects::update(const std::u16string& path)
{
    ZoneScoped;
	size_t existPos = recentOpenProjects.size() - 1;

	for (size_t i = 0; i < recentOpenProjects.size(); i++)
	{
		if (recentOpenProjects[i] == path)
		{
			existPos = i;
			break;
		}
	}

	// Move back.
	for (size_t i = existPos; i > 0; i--)
	{
		recentOpenProjects[i] = recentOpenProjects[i - 1];
	}

	// Insert first.
	recentOpenProjects[0] = path;

	updatePathForView();
}

void RecentOpenProjects::updatePathForView()
{
    ZoneScoped;
	auto copyArray = recentOpenProjects;
	validId = 0;
	for (size_t i = 0; i < recentOpenProjects.size(); i++)
	{
		if (std::filesystem::exists(recentOpenProjects[i]))
		{
			copyArray[validId] = recentOpenProjects[i];
			utf8Strings[validId] = utf8::utf16to8(copyArray[validId]);

			validId++;
		}
	}
	recentOpenProjects = copyArray;
}

std::filesystem::path getRecentProjectsPath()
{
	static const std::string kRecentProjectPath = kConfigCacheFolder + "recent-projects.ini";
	return kRecentProjectPath;
}

void RecentOpenProjects::save()
{
	updatePathForView();

	std::ofstream os(getRecentProjectsPath());

	for (size_t i = 0; i < recentOpenProjects.size(); i++)
	{
		os << utf8::utf16to8(recentOpenProjects[i]) << std::endl;
	}
}

void RecentOpenProjects::load()
{
    ZoneScopedN("RecentOpenProjects::load()");
	if (std::filesystem::exists(getRecentProjectsPath()))
	{
		std::ifstream is(getRecentProjectsPath());
		std::string u8string;
		for (size_t i = 0; i < recentOpenProjects.size(); i++)
		{
			std::getline(is, u8string);

			recentOpenProjects[i] = utf8::utf8to16(u8string);
		}
	}

	updatePathForView();
}