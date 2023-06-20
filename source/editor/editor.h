#pragma once

#include "widgets/dockspace.h"
#include "ui_string.h"
#include "widgets/console.h"
#include "widgets/downbar.h"
#include "widgets/project_hub.h"
#include "widgets/project_content.h"
#include "widgets/scene_outliner.h"
#include "undo.h"
#include <imgui/region_string.h>
#include <asset/asset_system.h>
#include "widgets/viewport.h"
#include "widgets/detail.h"
#include "widgets/asset_config.h"
#include "selection.h"
#include "widgets/render_manager.h"

class Editor : engine::NonCopyable
{
public:
	static Editor* get();

	// Editor run function.
	int run(int argc, char** argv);

	// Tick without command buffer, this tick first.
	engine::MulticastDelegate<const engine::RuntimeModuleTickData&, engine::VulkanContext*> tickFunctions; 

	// Tick with command buffer, this tick second.
	engine::MulticastDelegate<const engine::RuntimeModuleTickData&, VkCommandBuffer, engine::VulkanContext*> tickCmdFunctions; 

	// Cache renderer module.
	engine::Renderer* getRenderer() const {	return m_renderer; }

	// Cache engine handle.
	engine::Engine* getEngine() const { return m_engine; }

	engine::SceneManager* getSceneManager() const { return m_sceneManager; }

	// Cache vulkan context.
	engine::VulkanContext* getContext() const { return m_context; }

	engine::AssetSystem* getAssetSystem() const { return m_assetSystem; }

	bool isHubWidgetActive() const { return m_bHubWidgetActive; }
	void setHubWidgetActiveState(bool bState) { m_bHubWidgetActive = bState; }

	ViewportWidget* getViewportWidget() { return m_viewport.get(); }

	// Project file absolute path.
	const std::string& getProjectFilePathUtf8() const { return m_projectFilePathUtf8; }
	const std::u16string& getProjectFilePathUtf16() const { return m_projectFilePathUtf16; }

	// Project root absolute path.
	const std::string& getProjectRootPathUtf8() const { return m_projectRootPathUtf8; }
	const std::u16string& getProjectRootPathUtf16() const { return m_projectRootPathUtf16; }

	// Project name, no included period.
	const std::string& getProjectNameUtf8() const { return m_projectNameUtf8; }
	const std::u16string& getProjectNameUtf16() const { return m_projectNameUtf16; }

	const std::string& getProjectAssetPathUtf8() const { return m_projectAssetPathUtf8; }
	const std::u16string& getProjectAssetPathUtf16() const { return m_projectAssetPathUtf16; }

	void setupProjectDirectory(const std::filesystem::path& inProjectPath);
	void setTitleName() const;

	struct BuiltinAssets
	{
		std::unique_ptr<engine::VulkanImage> folderImage;
		std::unique_ptr<engine::VulkanImage> fileImage;

		std::unique_ptr<engine::VulkanImage> materialImage;
		std::unique_ptr<engine::VulkanImage> sceneImage;
		std::unique_ptr<engine::VulkanImage> meshImage;

		std::unique_ptr<engine::VulkanImage> sunImage;
		std::unique_ptr<engine::VulkanImage> userImage;
		std::unique_ptr<engine::VulkanImage> effectImage;
		std::unique_ptr<engine::VulkanImage> postImage;
	};

	engine::VulkanImage* getFolderImage() const { return m_builtinResources.folderImage.get(); }
	engine::VulkanImage* getFileImage()   const { return m_builtinResources.fileImage.get(); }

	engine::VulkanImage* getStaticMeshImage()   const { return m_builtinResources.meshImage.get(); }
	engine::VulkanImage* getMaterialImage()   const { return m_builtinResources.materialImage.get(); }
	engine::VulkanImage* getSceneImage()   const { return m_builtinResources.sceneImage.get(); }


	const auto* getBuiltinAssets() const {return &m_builtinResources;}

	VkDescriptorSet getSet(engine::VulkanImage* image, const VkSamplerCreateInfo& sampler);
	VkDescriptorSet getClampToTransparentBorderSet(engine::VulkanImage* image);

	Undo& getUndo() { return *m_undo; }
	const Undo& getUndo() const { return *m_undo; }

	AssetConfigWidgetManager* getAssetConfigManager() { return m_assetConfigs.get(); }

	void setFocusWindow(const std::string& name) { m_focusWindow = name; m_bShouldSetFocus = true; }

	const auto& getSceneNodeSelections() const { return m_selectedSceneNodes; }
	const auto& getSceneNodeSelected() const { return m_selectedSceneNodes.getSelections(); }
	auto& getSceneNodeSelections() { return m_selectedSceneNodes; }
	auto& getSceneNodeSelected() { return m_selectedSceneNodes.getSelections(); }

	const auto& getAssetSelections() const { return m_selectedAssets; }
	const auto& getAssetSelected() const { return m_selectedAssets.getSelections(); }
	auto& getAssetSelections() { return m_selectedAssets; }
	auto& getAssetSelected() { return m_selectedAssets.getSelections(); }

	ProjectContentWidget* getContentWidget() { return m_projectContent.get(); }

	const auto& getDirtyAssets() const { return m_dirtyAssets; }
	void saveDirtyAssetActions();
private:
	Editor() = default;

	void init();
	void release();


	void tick(const engine::RuntimeModuleTickData& tickData, engine::VulkanContext* context);
	void tickWithCmd(const engine::RuntimeModuleTickData& tickData, VkCommandBuffer cmd, engine::VulkanContext* context);

	void initBuiltinResources();

	void releaseBuiltinResources();

	void shortcutHandle();


	
	void onAssetDirty(std::shared_ptr<engine::AssetInterface> asset);

protected:
	// Project stem name, not include period.
	std::string m_projectNameUtf8;
	std::u16string m_projectNameUtf16;

	// Project file absolute path file in this computer, this is runtime generate value.
	std::string m_projectFilePathUtf8;
	std::u16string m_projectFilePathUtf16;

	// Project root path where project file exist, this is runtime generate value.
	std::string m_projectRootPathUtf8;
	std::u16string m_projectRootPathUtf16;

	std::string m_projectAssetPathUtf8;
	std::u16string m_projectAssetPathUtf16;

private:
	// Cache renderer module.
	engine::Renderer* m_renderer;

	// Cache engine handle.
	engine::Engine* m_engine;

	// Cache vulkan context.
	engine::VulkanContext* m_context;

	engine::SceneManager* m_sceneManager;

	engine::AssetSystem* m_assetSystem;

	// Tick function handle.
	engine::DelegateHandle m_tickFunctionHandle;
	engine::DelegateHandle m_tickCmdFunctionHandle;

	// Asset delegates.
	engine::DelegateHandle m_onAssetDirtyHandle;

	std::unique_ptr<Undo> m_undo;

	// Is hub widget active?
	bool m_bHubWidgetActive = true;

	std::unique_ptr<MainViewportDockspaceAndMenu> m_mainDockspace;
	std::unique_ptr<DownbarWidget> m_downbar;
	std::unique_ptr<WidgetConsole> m_console;
	std::unique_ptr<HubWidget> m_hubWidget;
	std::unique_ptr<ProjectContentWidget> m_projectContent;
	std::unique_ptr<SceneOutlinerWidget> m_outlinerWidget;
	std::unique_ptr<ViewportWidget> m_viewport;
	std::unique_ptr<WidgetDetail> m_detail;
	std::unique_ptr<RenderManagerWidget> m_renderManager;

	std::unique_ptr<AssetConfigWidgetManager> m_assetConfigs;

	BuiltinAssets m_builtinResources;

	// Cache set for editor image.
	std::unordered_map<engine::UUID64u, VkDescriptorSet> m_cacheImageSet;

	Selection<SceneNodeSelctor> m_selectedSceneNodes;
	Selection<std::filesystem::path> m_selectedAssets;

	// Foucus window.
	bool m_bShouldSetFocus = false;
	std::string m_focusWindow;

	// Asset dirty.
	std::unordered_map<engine::UUID, std::weak_ptr<engine::AssetInterface>> m_dirtyAssets;
};


#define SceneGraphUndoRecordEditor(Name) SceneGraphUndoScope __scenegraph_undo_editor(&Editor::get()->getUndo(), Editor::get()->getSceneManager(), Name)