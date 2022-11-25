#pragma once
#include "Pch.h"
#include "Widget.h"


class WidgetContentViewer;

class AssetTypeDrawer
{
private:
	size_t m_counter = 0;
	WidgetContentViewer* m_viewer;

public:
	explicit AssetTypeDrawer(WidgetContentViewer* inViewer)
		: m_viewer(inViewer)
	{

	}

	void drawAssetImport();
	void drawAssetNew();

	void flushLazyCallFunctions();

	std::vector<std::function<void()>> lazyCallFunctions;
};

struct AssetSnapShotDrawer
{
	class WidgetContentViewer* viewer;
	Flower::RegistryUUID entry;

	std::shared_ptr<Flower::GPUImageAsset> cacheAsset = nullptr;

	explicit AssetSnapShotDrawer(WidgetContentViewer* viewerIn)
		: viewer(viewerIn)
	{

	}

	void draw(float drawDimSize);
};


class WidgetContentViewer : public Widget
{
	friend AssetSnapShotDrawer;
public:
	WidgetContentViewer();
	virtual ~WidgetContentViewer() noexcept;

	void importAssetAction(Flower::EAssetType type, std::shared_ptr<Flower::RegistryEntry> entry = nullptr);

	void markContentSnapshotDirty()
	{
		m_bCacheSnapShotDirty = true;
	}

	void setWorkingEntry(std::shared_ptr<Flower::RegistryEntry> entry)
	{
		if (m_workingEntry.lock() != entry)
		{
			m_bCacheSnapShotDirty = true;
			m_workingEntry = entry;
		}
	}

protected:
	// event init.
	virtual void onInit() override;

	// event always tick.
	virtual void onTick(const Flower::RuntimeModuleTickData& tickData) override;

	// event release.
	virtual void onRelease() override;

	// event when widget visible tick.
	virtual void onVisibleTick(const Flower::RuntimeModuleTickData&) override;

	void drawMenu();

	void drawContent();

	void drawContentTreeView(const std::shared_ptr<Flower::RegistryEntry>& entry);

	void drawContentSnapshot();

private:
	ImGuiTextFilter m_filter;

	std::weak_ptr<Flower::RegistryEntry> m_selectedEntryInTreeView;

	bool m_bCacheSnapShotDirty = true;
	float m_inspectorItemIconSize = 5.0f;
	std::vector<AssetSnapShotDrawer> m_snapshotDrawers;
	std::unique_ptr<struct DragAndDropAssets> m_dragDropObjects;

	std::unique_ptr<AssetTypeDrawer> m_assetTypeDrawer;

	std::weak_ptr<Flower::RegistryEntry> m_workingEntry;


	bool m_bNeedSaveAsset = false;

	// Save scene info.
	static const auto GCreateScenePathSize = 256;
	char m_createSceneName[GCreateScenePathSize] = "";

	bool m_bSceneSelect = false;
};