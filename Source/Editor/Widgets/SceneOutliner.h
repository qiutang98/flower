#pragma once
#include "Pch.h"
#include "Widget.h"
#include <ImGui/ImGuiInternal.h>

class WidgetSceneOutliner : public Widget
{
public:
	WidgetSceneOutliner();
	virtual ~WidgetSceneOutliner() noexcept;

protected:
	// event init.
	virtual void onInit() override;

	// event always tick.
	virtual void onTick(const Flower::RuntimeModuleTickData& tickData) override;

	// event release.
	virtual void onRelease() override;

	// event when widget visible tick.
	virtual void onVisibleTick(const Flower::RuntimeModuleTickData&) override;

	void drawSceneNode(std::shared_ptr<Flower::SceneNode> node);

	void setSelectedNode(std::weak_ptr<Flower::SceneNode> node);

	void handleEvent();

	void popupMenu();

	void beginDragDrop(std::shared_ptr<Flower::SceneNode> node);

	void acceptDragdrop(bool bRoot);

	void handleDrawState(std::shared_ptr<Flower::SceneNode> node);

public:
	std::shared_ptr<Flower::SceneNode> getSelectedNode() 
	{ 
		return m_selectedNode.lock(); 
	}

	Flower::Scene* getScene()
	{
		return m_scene;
	}

private:
	Flower::SceneManager* m_manager = nullptr;
	Flower::Scene* m_scene = nullptr;

	const char* m_popupMenuName = "##OutlinerContextMenu";
	const char* m_outlinerDragDropName = "##OutlinerDragDropName";

	size_t m_drawIndex = 0;

	// Scene outliner select node.
	std::weak_ptr<Flower::SceneNode> m_selectedNode;

	// Rename input buffer.
	char m_inputBuffer[32];

	bool m_bExpandToSelection = false;
	bool m_bExpandedToSelection = false;
	bool m_bRenameing = false;

	std::weak_ptr<Flower::SceneNode>      m_hoverNode;
	std::weak_ptr<Flower::SceneNode>  m_leftClickNode;
	std::weak_ptr<Flower::SceneNode> m_rightClickNode;

	struct DragDropWrapper
	{
		size_t id;
		static const size_t UNVALID_ID = ~0;
		std::weak_ptr<Flower::SceneNode> dragingNode;
	};
	DragDropWrapper m_dragingNode;

	ImRect m_selectedNodeRect;
};