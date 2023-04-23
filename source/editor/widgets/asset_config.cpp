#include "asset_config.h"
#include "imgui/ui.h"
#include "imgui/region_string.h"
#include "../editor.h"
#include "downbar.h"
#include "dockspace.h"

using namespace engine;
using namespace engine::ui;

static inline ImGuiID getAssetConfigViewportID()
{
	static const auto kAssetConfigViewportID = ImGui::GetID("WidgetAssetConfigViewport");
	return kAssetConfigViewportID;
}

static inline void prepareAssetConfigWindow()
{
	// Set window class.
	static ImGuiWindowClass windClass;
	windClass.ClassId = ImGui::GetID("WidgetAssetConfig");
	windClass.DockingAllowUnclassed = false;
	windClass.DockNodeFlagsOverrideSet = ImGuiDockNodeFlags_NoDockingSplitMe;
	windClass.ViewportFlagsOverrideSet = ImGuiViewportFlags_NoAutoMerge;
	ImGui::SetNextWindowClass(&windClass);
}


WidgetAssetConfig::WidgetAssetConfig(Editor* editor, AssetConfigWidgetManager* manager, const std::filesystem::path& path)
	: m_editor(editor), m_manager(manager), m_bRun(true), m_nameUTF8(utf8::utf16to8(path.u16string()))
{

}

WidgetAssetConfig::~WidgetAssetConfig() noexcept
{

}

void WidgetAssetConfig::tick(const RuntimeModuleTickData& tickData, VulkanContext* context)
{

	prepareAssetConfigWindow();

	// Default window size set to 400x300
	ImGui::SetNextWindowSize(ImVec2(800, 400), ImGuiCond_FirstUseEver);
	if (ImGui::Begin(m_nameUTF8.c_str(), &m_bRun, ImGuiWindowFlags_NoSavedSettings))
	{
		ImGui::PushID(m_nameUTF8.c_str());
		{


		}
		ImGui::PopID();
	}

	ImGui::End();

}

AssetConfigWidgetManager::AssetConfigWidgetManager(Editor* editor)
	: m_editor(editor)
{

}

WidgetAssetConfig* AssetConfigWidgetManager::openWidget(const std::filesystem::path& path)
{
	if (!m_widgets.contains(path))
	{
		m_widgets[path] = std::make_unique<WidgetAssetConfig>(m_editor, this, path);
	}

	m_editor->setFocusWindow(m_widgets[path]->getNameUtf8());
	return m_widgets[path].get();
}

void AssetConfigWidgetManager::tick(const RuntimeModuleTickData& tickData, VulkanContext* context)
{
	// Clear cache viewport.
	m_tickDrawCtx.viewport = nullptr;

	if (!m_widgets.empty())
	{
		for (auto& widget : m_widgets)
		{
			widget.second->tick(tickData, context);
		}

		std::erase_if(m_widgets, [](const auto& item) { auto const& [path, widget] = item; return widget->shouldClosed(); });
	}
}

void AssetConfigWidgetManager::tickCmd(const RuntimeModuleTickData& tickData, VkCommandBuffer cmd, VulkanContext* context)
{


	for (auto& widget : m_widgets)
	{
		widget.second->tickWithCmd(tickData, cmd, context);
	}
}

void AssetConfigWidgetManager::release()
{
	m_widgets.clear();
}