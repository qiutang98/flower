#pragma once

#include <ui/ui.h>
#include <graphics/graphics.h>

#include <filesystem>

class Editor;

struct RecentOpenProjects
{
	size_t validId = 0;
	std::array<std::u16string, 10> recentOpenProjects;
	std::array<std::string, 10> utf8Strings;

	void update(const std::u16string& path);
	void updatePathForView();

	void save();
	void load();
};

class HubWidget : public engine::WidgetBase
{
public:
	explicit HubWidget();

protected:
	virtual void onInit() override;
	virtual void onTick(const engine::RuntimeModuleTickData& tickData, engine::VulkanContext* context) override;
	virtual void onRelease() override;

	bool loadProject();
	bool newProject();
	bool createOrOpenProject(const std::filesystem::path& path);
	void setupEditorProject(const std::filesystem::path& path);



protected:
	// Is project path ready.
	bool m_bProjectPathReady = false;

	// Input of project path.
	static const auto kProjectPathSize = 256;
	char m_projectPath[kProjectPathSize] = "";

	// Recent project lists.
	RecentOpenProjects m_recentProjectList;
};