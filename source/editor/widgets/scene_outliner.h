#pragma once

#include "../editor.h"
#include "../selection.h"

#include <scene/scene.h>
#include <scene/scene_manager.h>

struct SceneNodeSelctor
{
	std::weak_ptr<engine::SceneNode> node;
	size_t nodeId = ~0;

	explicit SceneNodeSelctor(const std::shared_ptr<engine::SceneNode> inNode)
		: node(inNode)
	{
		if (inNode)
		{
			nodeId = inNode->getId();
		}
	}

	bool operator==(const SceneNodeSelctor& rhs) const
	{
		return nodeId == rhs.nodeId;
	}

	bool operator!=(const SceneNodeSelctor& rhs) const { return !(*this == rhs); }
	bool operator<(const SceneNodeSelctor& rhs) const { return nodeId < rhs.nodeId; }

	operator bool() const { return isValid(); }
	bool isValid() const { return node.lock() != nullptr; }
};

class SceneOutlinerWidget : public engine::WidgetBase
{
public:
	SceneOutlinerWidget();

	// event init.
	virtual void onInit() override;

	// event always tick.
	virtual void onTick(const engine::RuntimeModuleTickData& tickData, engine::VulkanContext* context) override;

	// event release.
	virtual void onRelease() override;

	// event when widget visible tick.
	virtual void onVisibleTick(const engine::RuntimeModuleTickData& tickData) override;

	const auto& getSelection() const { return m_sceneSelections; }
	auto& getSelection() { return m_sceneSelections; }
	void clearSelection() { m_sceneSelections.clear(); }

private:
	void drawSceneNode(std::shared_ptr<engine::SceneNode> node);

	void handleEvent();

	void popupMenu();

	void beginDragDrop(std::shared_ptr<engine::SceneNode> node);

	void acceptDragdrop(bool bRoot);

	void handleDrawState(std::shared_ptr<engine::SceneNode> node);

	void rebuildSceneNodeNameMap();

	[[nodiscard]] std::string addUniqueIdForName(const std::string& name);

	void sortChildren(std::shared_ptr<engine::SceneNode> node);

	void onActiveSceneChange(engine::Scene* old, engine::Scene* newScene);



private:
	engine::UUID m_activeSceneUUID { };

	Selection<SceneNodeSelctor> m_sceneSelections;

	struct
	{
		// Padding in y.
		float scenePaddingItemY = 5.0f;

		// Draw index for each draw loop.
		size_t drawIndex = 0;

		std::unordered_set<size_t> expandNodeInTreeView = {};

		// Rename input buffer.
		char inputBuffer[32];

		// Is renameing?
		bool bRenameing = false;

		// Mouse hover/left click/right click node.
		std::weak_ptr<engine::SceneNode> hoverNode = { };

		engine::UUID64u dragDropUUID;
		std::vector<std::weak_ptr<engine::SceneNode>> dragingNodes = { };

		// Cache node string map avoid rename with same name.
		std::unordered_map<std::string, size_t> cacheNodeNameMap;

		inline static const char* kPopupMenuName = "##OutlinerContextMenu";
		inline static const char* kOutlinerDragDropName = "##OutlinerDragDropName";
	} m_drawContext;


	engine::DelegateHandle m_onActiveSceneChange;
};