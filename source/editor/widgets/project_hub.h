#pragma once

#include <util/framework.h>
#include <rhi/rhi.h>
#include <renderer/renderer.h>
#include <imgui/ui.h>
#include "../widget.h"

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

class HubWidget : public Widget
{
public:
	HubWidget(Editor* editor);

protected:
	virtual void onInit() override;
	virtual void onTick(const engine::RuntimeModuleTickData& tickData, engine::VulkanContext* context) override;
	virtual void onRelease() override;

	bool loadProject();
	bool newProject();
	bool createOrOpenProject(const std::filesystem::path& path);
	void setupEditorProject(const std::filesystem::path& path);

	bool m_bProjectPathReady = false;

	static const auto kProjectPathSize = 256;
	char m_projectPath[kProjectPathSize] = "";


	RecentOpenProjects m_recentProjectList;
};