#pragma once
#include "Pch.h"
#include "Widgets/WidgetsHeader.h"

class Editor
{
private:
	// Collections.
	std::vector<std::unique_ptr<Widget>> m_widgets;

	// Widgets.
	DockSpace*      m_dockSpace;
	WidgetDownbar*  m_downbar;
	WidgetConsole*  m_console;
	WidgetViewport* m_viewport;
	WidgetSceneOutliner* m_outliner;
	WidgetDetail*  m_detail;
	WidgetProjectSelect* m_projectSelect;
	WidgetContentViewer* m_contentViewer;
	WidgetRenderSetting* m_renderSetting;
public:
	void run();

	auto* getDockSpace() { return m_dockSpace; }
	auto* getWidgetDownbar() { return m_downbar; }
	auto* getWidgetConsole() { return m_console; }
	auto* getWidgetViewport() { return m_viewport; }
	auto* getSceneOutliner() { return m_outliner; }
	auto* getWidgetDetail() { return m_detail; }
	auto* getProjectSelect() { return m_projectSelect; }
	auto* getContentViewer() { return m_contentViewer; }

	auto* getRenderSetting() { return m_renderSetting; }
private:
	void preInit(const Flower::LauncherInfo& info);
	void init();
	void tick(const Flower::EngineTickData& tickData);
	void release();

public:
	bool setProjectPath(const std::filesystem::path& in);
};

extern Editor* const GEditor;