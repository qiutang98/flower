#pragma once

#include "../editor.h"
#include "../selection.h"

class ProjectAssetTreeEntry 
	: public std::enable_shared_from_this<ProjectAssetTreeEntry>, engine::NonCopyable
{
	friend class ProjectAssetTree;

public:
	ProjectAssetTreeEntry(engine::u8str name, bool bFolder, const std::filesystem::path& path, std::shared_ptr<ProjectAssetTreeEntry> parent);

	// Getter functions.
	bool isFoleder() const { return m_bFolder; }
	bool isChildrenEmpty() const { return m_children.empty(); }
	const auto& getChildren() const { return m_children; }
	auto& getChildren() { return m_children; }
	const auto& getParent() { return m_parent; }
	const engine::u8str& getName() const { return m_name; }
	bool isFolderOpen() const { return m_bFolderOpen; }
	const auto& getPath() const { return m_path; }
	void setFolderOpenState(bool bState) { m_bFolderOpen = bState; }
	const auto& getRuntimeUUID() const { return m_runtimeUUID; }

	ImTextureID getSet(ImVec2& uv0, ImVec2& uv1);

private:
	void buildTreeRecursive(ProjectAssetTree* tree);

private:
	// Entry is folder or not.
	bool m_bFolder;

	// Folder is open.
	bool m_bFolderOpen;

	// Entry disk path.
	std::filesystem::path m_path;

	// Entry name in utf8 encode.
	engine::u8str m_name;

	// Hierarchy structure.
	std::weak_ptr<ProjectAssetTreeEntry> m_parent;
	std::vector<std::shared_ptr<ProjectAssetTreeEntry>> m_children;

	engine::UUID64u m_runtimeUUID;
};

struct AssetSelector
{
	std::weak_ptr<ProjectAssetTreeEntry> entry;

	explicit AssetSelector(std::shared_ptr<ProjectAssetTreeEntry> inEntry)
		: entry(inEntry)
	{

	}

	bool isValid() const
	{
		return entry.lock() != nullptr;
	}

	operator bool() const
	{
		return isValid();
	}

	bool operator==(const AssetSelector& rhs) const
	{
		return entry.lock() == rhs.entry.lock();
	}

	bool operator!=(const AssetSelector& rhs) const
	{
		return !(*this == rhs);
	}

	bool operator<(const AssetSelector& rhs) const
	{
		if (entry.lock() && rhs.entry.lock())
		{
			return entry.lock()->getPath() < rhs.entry.lock()->getPath();
		}
		
		return false;
	}
};

class ProjectAssetTree : engine::NonCopyable
{
	friend class ProjectAssetTreeEntry;
	friend class ProjectContentModel;
public:
	ProjectAssetTree() = default;

	std::shared_ptr<ProjectAssetTreeEntry> getRoot() { return m_projectAssetTreeRoot; }
	std::weak_ptr<ProjectAssetTreeEntry> getEntry(const std::filesystem::path& path)
	{
		return m_pathEntryMap[path];
	}
protected:
	void build();


private:
	std::shared_ptr<ProjectAssetTreeEntry> m_projectAssetTreeRoot;

	std::unordered_map<std::filesystem::path, std::weak_ptr<ProjectAssetTreeEntry>> m_pathEntryMap;
};

class ProjectContentModel : engine::NonCopyable
{
public:
	ProjectContentModel();
	~ProjectContentModel();

	void release();

	void tick();

	void rebuild();

	const ProjectAssetTree& getTree() const { return m_projectAssetTree; }
	ProjectAssetTree& getTree() { return m_projectAssetTree; }

	engine::MulticastDelegate<> onProjectTreeRebuild;

protected:
	// Project asset tree.
	ProjectAssetTree m_projectAssetTree;

	std::atomic<bool> m_bDirty = false;

	engine::DelegateHandle m_onAssetNewlySaveToDisk = {};
};

class WidgetContent : public engine::WidgetBase
{
public:
	explicit WidgetContent(size_t index, ProjectContentModel* model);

protected:
	// event init.
	virtual void onInit() override;

	// event always tick.
	virtual void onTick(const engine::RuntimeModuleTickData& tickData, engine::VulkanContext* context) override;

	// event release.
	virtual void onRelease() override;

	// event when widget visible tick.
	virtual void onVisibleTick(const engine::RuntimeModuleTickData& tickData) override;

	const auto& getSelections() const { return m_selections; }
	auto& getSelections() { return m_selections; }

	void onContentTreeRebuild();

private:
	void setupProject();

	void drawMenu(const engine::RuntimeModuleTickData& tickData);
	void drawContent(const engine::RuntimeModuleTickData& tickData);


	void drawContentTreeView(std::shared_ptr<ProjectAssetTreeEntry> entry);
	void drawContentSnapShot(std::shared_ptr<ProjectAssetTreeEntry> entry);
	void drawItemSnapshot(float drawDimSize, std::shared_ptr<ProjectAssetTreeEntry> entry);

	void setActiveEntry(std::shared_ptr<ProjectAssetTreeEntry> entry);

	void drawRightClickedMenu();

	void drawAssetImport();


protected:
	ProjectContentModel* m_model;
	engine::DelegateHandle m_onTreeViewRebuildHandle = {};

	size_t m_index;

	std::filesystem::path m_activeFolder = {};

	ImGuiTextFilter m_filter;

	// Tree view hovering entry.
	std::weak_ptr<ProjectAssetTreeEntry> m_treeviewHoverEntry;
	// Asset selection state.
	Selection<AssetSelector> m_selections;


	float m_snapshotItemIconSize = 6.0f;
};