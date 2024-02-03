#pragma once

#include "../editor.h"

struct WidgetInView
{
	bool bMultiWindow;
	std::array<engine::WidgetBase*, Editor::kMultiWidgetMaxNum> widgets;
};

class SceneAssetSaveWidget : public engine::ui::ImGuiPopupSelfManagedOpenState
{
public:
	explicit SceneAssetSaveWidget(const std::string& titleName);

	std::function<void()> afterEventAccept = nullptr;

protected:
	virtual void onDraw() override;
	virtual void onClosed() override
	{
		afterEventAccept = nullptr;
		m_bSelected = true;
		m_processingAsset = {};
	}

private:
	bool m_bSelected = true;
	engine::AssetSaveInfo m_processingAsset = {};
};

class ContentAssetImportWidget : public engine::ui::ImGuiPopupSelfManagedOpenState
{
public:
	explicit ContentAssetImportWidget(const std::string& titleName);

	rttr::string_view typeName = {};
	std::function<void()> afterEventAccept = nullptr;

	std::vector<std::shared_ptr<engine::AssetImportConfigInterface>> importConfigs = {};


protected:
	virtual void onDraw() override;
	virtual void onClosed() override
	{
		typeName = {};
		afterEventAccept = nullptr;
		importConfigs = {};
		m_bImporting = false;
	}

	void onDrawState();
	void onDrawImporting();

	// Import progress handle.
	struct ImportProgress
	{
		engine::DelegateHandle logHandle{ };
		std::deque<std::pair<engine::ELogType,std::string>> logItems{ };
	} m_importProgress{ };

	// Import execut futures.
	engine::FutureCollection<void> m_executeFutures = {};

	// The assets is importing?
	bool m_bImporting = false;
	
};

// Control main viewport dockspace of the windows.
class MainViewportDockspaceAndMenu : public engine::WidgetBase
{
public:
	MainViewportDockspaceAndMenu();

	static void dockspace(
		bool bNewWindow, 
		const std::string& name, 
		const engine::RuntimeModuleTickData& tickData, 
		engine::VulkanContext* context, 
		ImGuiViewport* viewport, 
		std::function<void()>&& menu);

	engine::RegisterManager<WidgetInView> widgetInView;

	SceneAssetSaveWidget sceneAssetSave;
	ContentAssetImportWidget contentAssetImport;

protected:
	virtual void onTick(const engine::RuntimeModuleTickData& tickData, engine::VulkanContext* context) override;

	void drawDockspaceMenu();


};