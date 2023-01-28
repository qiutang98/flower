#include "Pch.h"
#include "Editor.h"

using namespace Flower;
using namespace Flower::UI;

Editor* const GEditor = new Editor();

void Editor::preInit(const LauncherInfo& info)
{
	GEngine->registerRuntimeModule<AssetSystem>();
	GEngine->registerRuntimeModule<SceneManager>();
	GEngine->registerRuntimeModule<Renderer>();
	

	{
		auto newWidget = std::make_unique<std::remove_pointer_t<decltype(m_dockSpace)>>();
		m_dockSpace = newWidget.get();
		m_widgets.push_back(std::move(newWidget));
	}
	{
		auto newWidget = std::make_unique<std::remove_pointer_t<decltype(m_downbar)>>();
		m_downbar = newWidget.get();
		m_widgets.push_back(std::move(newWidget));
	}
	{
		auto newWidget = std::make_unique<std::remove_pointer_t<decltype(m_console)>>();
		m_console = newWidget.get();
		m_widgets.push_back(std::move(newWidget));
	}
	{
		auto newWidget = std::make_unique<std::remove_pointer_t<decltype(m_viewport)>>();
		m_viewport = newWidget.get();
		m_widgets.push_back(std::move(newWidget));
	}
	{
		auto newWidget = std::make_unique<std::remove_pointer_t<decltype(m_outliner)>>();
		m_outliner = newWidget.get();
		m_widgets.push_back(std::move(newWidget));
	}
	{
		auto newWidget = std::make_unique<std::remove_pointer_t<decltype(m_detail)>>();
		m_detail = newWidget.get();
		m_widgets.push_back(std::move(newWidget));
	}
	{
		auto newWidget = std::make_unique<std::remove_pointer_t<decltype(m_projectSelect)>>();
		m_projectSelect = newWidget.get();
		m_widgets.push_back(std::move(newWidget));
	}
	{
		auto newWidget = std::make_unique<std::remove_pointer_t<decltype(m_contentViewer)>>();
		m_contentViewer = newWidget.get();
		m_widgets.push_back(std::move(newWidget));
	}
	{
		auto newWidget = std::make_unique<std::remove_pointer_t<decltype(m_renderSetting)>>();
		m_renderSetting = newWidget.get();
		m_widgets.push_back(std::move(newWidget));
	}
	{
		auto newWidget = std::make_unique<std::remove_pointer_t<decltype(m_assetInspector)>>();
		m_assetInspector = newWidget.get();
		m_widgets.push_back(std::move(newWidget));
	}
}

void Editor::init()
{
	for (const auto& ptr : m_widgets)
	{
		ptr->init();
	}
}

void Editor::tick(const EngineTickData& tickData)
{

}

void Editor::release()
{
	for (const auto& ptr : m_widgets)
	{
		ptr->release();
	}
}

bool Editor::setProjectPath(const std::filesystem::path& in)
{
	// In path with format like "C://user/d/Ck/Ck.flower
	// Two points require:
	// 1. project file end with .flower.
	// 2. project file parent is same name with project file name. eg. Ck.flower under folder Ck.
	if (!(in.string().ends_with(".flower")))
	{
		return false;
	}
	if (!(in.parent_path().filename() == in.stem()))
	{
		return false;
	}

	CHECK(ProjectContext::get()->project.isValid());

	auto projectPath = in.parent_path();
	auto projectName = projectPath.filename().string();

	ProjectContext::get()->path = projectPath;

	LOG_TRACE("Set folder {0} as working path to it.", projectPath.string());

	GEngine->getRuntimeModule<AssetSystem>()->setupProject(projectPath);

	ASSERT(ProjectContext::get()->project.getName() == projectName, 
		"Project name {0} should same with path {1}", ProjectContext::get()->project.getName(), projectName);

	Launcher::setWindowTileName(
		projectName.c_str(),
		GEngine->getRuntimeModule<SceneManager>()->getScenes()->getName().c_str());

	GEditor->getContentViewer()->markContentSnapshotDirty();

	return true;
}

void Editor::run()
{
	Launcher::preInitHookFunction.addRaw(this, &Editor::preInit);
	Launcher::initHookFunction.addRaw(this, &Editor::init);
	Launcher::tickFunction.addRaw(this, &Editor::tick);
	Launcher::releaseHookFunction.addRaw(this, &Editor::release);

	CHECK(Launcher::preInit());
	CHECK(Launcher::init());

	Launcher::guardedMain();
	Launcher::release();
}
