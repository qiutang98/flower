#pragma once
#include "../widget.h"
#include <imgui/imgui.h>
#include <asset/asset.h>
#include <utf8/cpp17.h>
#include <asset/asset_texture.h>
#include <asset/asset_system.h>
#include <asset/asset_staticmesh.h>

class ProjectAssetTree;
class ProjectAssetTreeEntry;

// Asset registry is an asset tree.
// We also cache some asset info map for quick search.

class ProjectAssetTreeEntry : public std::enable_shared_from_this<ProjectAssetTreeEntry>
{
	friend class ProjectAssetTree;

public:
	ProjectAssetTreeEntry(std::string nameUtf8, bool bFolder, const std::filesystem::path& path, std::shared_ptr<ProjectAssetTreeEntry> parent)
		: m_nameUtf8(nameUtf8)
		, m_bFolder(bFolder)
		, m_path(path)
		, m_parent(parent)
		, m_runtimeUUID(engine::buildRuntimeUUID64u())
		, m_bFolderOpen(false)
	{

	}

	// Getter functions.
	bool isFoleder() const { return m_bFolder; }
	bool isChildrenEmpty() const { return m_children.empty(); }
	const auto& getChildren() const { return m_children; }
	auto& getChildren() { return m_children; }
	const auto& getParent() { return m_parent; }
	const std::string& getNameUtf8() const { return m_nameUtf8; }
	bool isFolderOpen() const { return m_bFolderOpen; }
	const auto& getPath() const { return m_path; }
	void setFolderOpenState(bool bState) { m_bFolderOpen = bState; }

	const auto& getRuntimeUUID() const { return m_runtimeUUID; }

	VkDescriptorSet getSet(class Editor* editor, ImVec2& uv0, ImVec2& uv1);

private:
	void buildTreeRecursive();

private:
	// Entry is folder or not.
	bool m_bFolder;

	// Folder is open.
	bool m_bFolderOpen;

	// Entry disk path.
	std::filesystem::path m_path;

	// Entry name in utf8 encode.
	std::string m_nameUtf8;

	// Hierarchy structure.
	std::weak_ptr<ProjectAssetTreeEntry> m_parent;
	std::vector<std::shared_ptr<ProjectAssetTreeEntry>> m_children;

	engine::UUID64u m_runtimeUUID;

	// Show snapshot set cache.
	struct
	{
		VkDescriptorSet set = VK_NULL_HANDLE;
		ImVec2 uv0 = { 0.0f, 0.0f };
		ImVec2 uv1 = { 1.0f, 1.0f };
	} m_drawDetail;

};

class ProjectAssetTree
{
public:
	ProjectAssetTree() = default;

	void setupProject(const std::filesystem::path& projectFilePath)
	{
		// First update root path.
		m_assetRootPath = projectFilePath.parent_path() / "asset";

		// Update asset tree.
		updateWholeAsset();
	}

	void updateWholeAsset()
	{
		m_projectAssetTreeRoot = std::make_shared<ProjectAssetTreeEntry>("asset", true, m_assetRootPath, nullptr);

		m_projectAssetTreeRoot->buildTreeRecursive();
	}

	auto getRoot() { return m_projectAssetTreeRoot; }

private:
	std::filesystem::path m_assetRootPath;

	std::shared_ptr<ProjectAssetTreeEntry> m_projectAssetTreeRoot;
};

struct DragAndDropAssets
{
	void clear()
	{
		selectAssets.clear();
	}

	std::unordered_set<std::filesystem::path> selectAssets;
};

class ProjectContentWidget : public Widget
{
public:
	ProjectContentWidget(Editor* editor);

	// event init.
	virtual void onInit() override;

	// event always tick.
	virtual void onTick(const engine::RuntimeModuleTickData& tickData, engine::VulkanContext* context) override;

	// event release.
	virtual void onRelease() override;

	// event when widget visible tick.
	virtual void onVisibleTick(const engine::RuntimeModuleTickData& tickData) override;

	// Setup project.
	void setupProject(const std::filesystem::path& path);

	const auto& getDragDropAssets() const { return m_dragDropAssets; }
	void clearDragDropAssets() { m_dragDropAssets.clear(); }

private:
	void drawMenu();

	void drawAssetImport();

	void drawAssetCreate();

	void drawContent();

	// Draw content tree view.
	void drawContentTreeView(std::shared_ptr<ProjectAssetTreeEntry> entry);

	void setActiveEntry(std::shared_ptr<ProjectAssetTreeEntry> entry);
	
	bool importAssetAction(engine::EAssetType type);

	ProjectAssetTree* getProjectAssetTree();

	void drawContentSnapshot();
	void drawItemSnapshot(float drawDimSize, std::shared_ptr<ProjectAssetTreeEntry> entry);


	void drawAssetImportModal();
	void drawImageImportModalContent();
	void drawStaticMeshImportModalContent();
	void executeImport();

public:
	static const std::string kAssetDragDropName;

private:
	// Import progress handle.
	struct ImportProgress
	{
		engine::DelegateHandle logHandle { };
		std::deque<std::string> logItems { };
	} m_importProgress { };


	ImGuiTextFilter m_filter;

	// Project asset tree.
	std::unique_ptr<ProjectAssetTree> m_projectAssetTree = nullptr;

	// Active asset folder.
	std::filesystem::path m_activeFolder;

	// Active asset entry or selected entry.
	std::weak_ptr<ProjectAssetTreeEntry> m_workingEntry;
	std::weak_ptr<ProjectAssetTreeEntry> m_treeviewHoverEntry;

	// Content snapshot item icon size.
	float m_snapshotItemIconSize = 5.0f;


	DragAndDropAssets m_dragDropAssets;

	struct
	{
		engine::EAssetType type;

		// Import execut futures.
		engine::FutureCollection<void> executeFutures = {};

		// Show modal state.
		bool bShouldDrawImportModal = false;

		// Execute import configs.
		bool bExecuteImportConfigs = false;

		// Import file path.
		std::vector<std::filesystem::path> srcPaths;
		std::vector<std::filesystem::path> savePaths;

		// Asset import config init or no.
		bool bConfigInit = false;

		// Image import configs.
		std::vector<engine::AssetTexture::ImportConfig> imageConfigs;
		std::vector<engine::AssetStaticMesh::ImportConfig> staticmeshConfigs;

		void cleanState()
		{
			srcPaths.clear();
			savePaths.clear();

			bConfigInit = false;
			bExecuteImportConfigs = false;
			bShouldDrawImportModal = false;
			executeFutures = {};

			imageConfigs.clear();
			staticmeshConfigs.clear();
		}
		
	} m_assetImportPayload;
};