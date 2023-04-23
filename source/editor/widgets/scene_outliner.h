#pragma once

#include "../widget.h"
#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>
#include <asset/asset.h>
#include <utf8/cpp17.h>
#include "../selection.h"
#include <scene/scene.h>
#include <scene/scene_graph.h>

class SceneOutlinerWidget : public Widget
{
public:
	SceneOutlinerWidget(Editor* editor);

	// event init.
	virtual void onInit() override;

	// event always tick.
	virtual void onTick(const engine::RuntimeModuleTickData& tickData, engine::VulkanContext* context) override;

	// event release.
	virtual void onRelease() override;

	// event when widget visible tick.
	virtual void onVisibleTick(const engine::RuntimeModuleTickData& tickData) override;

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

private:
	engine::UUID m_activeSceneUUID { };

	Selection<SceneNodeSelctor>* m_sceneSelections = nullptr;

	struct
	{
		// Padding in y.
		float scenePaddingItemY = 5.0f;

		// Draw index for each draw loop.
		size_t drawIndex = 0;

		// Rename input buffer.
		char inputBuffer[32];

		// Is renameing?
		bool bRenameing = false;

		// Mouse hover/left click/right click node.
		std::weak_ptr<engine::SceneNode> hoverNode      = { };

		inline static const char* kPopupMenuName = "##OutlinerContextMenu";
	
		engine::UUID64u dragDropUUID;
		std::vector<std::weak_ptr<engine::SceneNode>> dragingNodes = { };
		inline static const char* kOutlinerDragDropName = "##OutlinerDragDropName";

		// Cache node string map avoid rename with same name.
		std::unordered_map<std::string, size_t> cacheNodeNameMap;

	} m_drawContext;
};

// Snapshot basic undo for scene graph. TODO: XOR compression to save some memory.
struct SceneGraphUndo : public Undo::Entry
{
public:
	SceneGraphUndo(Undo* undo, engine::SceneManager* manager, const char* name) : m_manager(manager), m_name(name)
	{
		m_dataCopy = manager->getActiveScene()->saveToStream();
	}

	virtual const char* getType() const { return m_name; }

	virtual void undo()
	{
		// Save current state.
		auto oldData = m_manager->getActiveScene()->saveToStream();

		// Load prev state.
		m_manager->getActiveScene()->loadFromStream(std::move(m_dataCopy));

		// Update data.
		m_dataCopy = std::move(oldData);
	}

	virtual void redo() { undo(); }

private:
	const char* m_name;

	// Scene manager.
	engine::SceneManager* m_manager;

	// Copy data.
	std::stringstream m_dataCopy;
};

class SceneGraphUndoScope
{
public:
	explicit SceneGraphUndoScope(Undo* undo, engine::SceneManager* manager, const char* name)
	{
		m_undo = undo;
		m_entry = Undo::createEntry<SceneGraphUndo>(undo, manager, name);
	}

	~SceneGraphUndoScope()
	{
		m_undo->done(std::move(m_entry));
	}
private:
	std::unique_ptr<SceneGraphUndo> m_entry;
	Undo* m_undo;
};

#define SceneGraphUndoRecord(Name) SceneGraphUndoScope __scenegraph_undo(m_undo, m_sceneManager, Name)