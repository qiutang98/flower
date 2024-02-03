#pragma once

#include <engine.h>
#include <filesystem>

#include <renderer/renderer.h>
#include <ui/ui.h>
#include <utfcpp/utf8.h>
#include <utfcpp/utf8/cpp17.h>

#include <reflection/reflection.h>

#include "selection.h"

#include <scene/scene.h>
#include <renderer/render_scene.h>

#include <profile/profile.h>

class WidgetConsole;
class HubWidget;
class WidgetContent;
class MainViewportDockspaceAndMenu;
class ProjectContentModel;
struct EditorBuiltinResource;
class SceneOutlinerWidget;
struct SceneNodeSelctor;
class WidgetDetail;
class ViewportWidget;

class Editor
{
private:
	Editor() = default;

	bool init();
	bool release();

	void tick(const engine::RuntimeModuleTickData& tickData, engine::VulkanContext* context);
	void tickWithCmd(const engine::RuntimeModuleTickData& tickData, VkCommandBuffer cmd, engine::VulkanContext* context);

	bool onWindowRequireClosed(const engine::GLFWWindows* windows);

public:
	static const size_t kMultiWidgetMaxNum = 4;

	// Get editor instance.
	static Editor* get();

	// Editor run function.
	int run(int argc, char** argv);

	void setTitleName(const std::u16string& name) const;
	void closedHubWidget();

	engine::WidgetManager& getWidgetManager() { return m_widgetManager; }

	engine::MulticastDelegate<const engine::RuntimeModuleTickData&, engine::VulkanContext*> tickFunctions; // Tick without command buffer, this tick first.
	engine::MulticastDelegate<const engine::RuntimeModuleTickData&, VkCommandBuffer, engine::VulkanContext*> tickCmdFunctions; // Tick with command buffer, this tick second.

	engine::CallOnceEvents<> onceEventAfterTick;

	SceneOutlinerWidget* getSceneOutlineWidegt() const { return m_outliner; }
	WidgetConsole* getConsole() const { return m_consoleHandle; }
	MainViewportDockspaceAndMenu* getDockSpaceAndMenu() const { return m_dockSpace; }

	EditorBuiltinResource* getBuiltinResources() const { return m_builtinResources.get(); }

	ImTextureID getImGuiTexture(engine::VulkanImage* image, const VkSamplerCreateInfo& sampler);
	ImTextureID getClampToTransparentBorderImGuiTexture(engine::VulkanImage* image);


	engine::MulticastDelegate<Selection<SceneNodeSelctor>&> onOutlinerSelectionChange;

	static const char* getDragDropAssetsName() { return "dark_ContentAssetDragDrops"; }
	auto& getDragDropAssets() { return m_dragDropAssets; }
	const auto& getDragDropAssets() const { return m_dragDropAssets; }
	void clearDragDropAssets() { m_dragDropAssets.clear(); }

	engine::math::vec3 getActiveViewportCameraPos() const { return m_activeViewportCameraPosition; }
	void setActiveViewportCameraPos(engine::math::vec3 in) { m_activeViewportCameraPosition = in; }

private:
	void updateApplicationTitle();
	void shortcutHandle();

private:
	engine::GLFWWindows m_windows;
	engine::RendererManager* m_renderer;

	// Builtin resource.
	std::unique_ptr<EditorBuiltinResource> m_builtinResources = nullptr;

	engine::WidgetManager m_widgetManager;
	HubWidget* m_hubHandle = nullptr;

	// Tick function handle.
	engine::DelegateHandle m_tickFunctionHandle;
	engine::DelegateHandle m_tickCmdFunctionHandle;

	engine::DelegateHandle m_onWindowRequireColosedHandle;

	MainViewportDockspaceAndMenu* m_dockSpace = nullptr;
	WidgetConsole* m_consoleHandle = nullptr;

	// Model of project content.
	std::unique_ptr<ProjectContentModel> m_projectContentModel = nullptr;
	// Views of project content.
	std::array<WidgetContent*, kMultiWidgetMaxNum> m_contents;
	SceneOutlinerWidget* m_outliner;
	std::array<WidgetDetail*, kMultiWidgetMaxNum> m_details;
	std::array<ViewportWidget*, kMultiWidgetMaxNum> m_viewports;

	// Cache image set for editor.
	std::unordered_map<engine::UUID64u, ImGuiTexture> m_cacheImGuiImage;

	// Draging and droping assets.
	DragAndDropAssets m_dragDropAssets;

	engine::math::vec3 m_activeViewportCameraPosition;
};