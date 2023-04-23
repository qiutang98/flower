#include "scene_outliner.h"
#include "imgui/ui.h"
#include "imgui/region_string.h"
#include "../editor.h"
#include <regex>
#include "component_draw/component_draw.h"

using namespace engine;
using namespace engine::ui;

RegionStringInit SceneOutliner_Title("SceneOutliner_Title", "Outliner", "SceneOutliner");
const static std::string ICON_OUTLINER_CONTENT = ICON_FA_CHESS_QUEEN;

SceneOutlinerWidget::SceneOutlinerWidget(Editor* editor)
	: Widget(editor, "Outliner")
{

}


void SceneOutlinerWidget::onInit()
{
	m_name = combineIcon(SceneOutliner_Title, ICON_OUTLINER_CONTENT);
	m_sceneSelections = &m_editor->getSceneNodeSelections();
}

void SceneOutlinerWidget::onRelease()
{

}

void SceneOutlinerWidget::onTick(const RuntimeModuleTickData& tickData, VulkanContext* context)
{
	// Check wether active scene change.
	auto activeScene = m_sceneManager->getActiveScene();
	if (m_activeSceneUUID != activeScene->getUUID())
	{
		m_activeSceneUUID = activeScene->getUUID();
		rebuildSceneNodeNameMap();
	}
}

void SceneOutlinerWidget::onVisibleTick(const RuntimeModuleTickData& tickData)
{
	auto activeScene = m_sceneManager->getActiveScene();

	// Reset draw index.
	m_drawContext.drawIndex = 0;
	m_drawContext.hoverNode = { };

	// Header text.
	ImGui::Spacing();
	ImGui::TextDisabled("%s  Active scene:  %s.", ICON_FA_FAN, activeScene->getName().c_str());
	ImGui::Spacing();
	ImGui::Separator();

	// 
	const float footerHeightToReserve = ImGui::GetStyle().ItemSpacing.y + ImGui::GetFrameHeightWithSpacing();
	ImGui::BeginChild("HierarchyScrollingRegion", ImVec2(0, -footerHeightToReserve), true, ImGuiWindowFlags_HorizontalScrollbar);
	ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(ImGui::GetStyle().ItemSpacing.x, m_drawContext.scenePaddingItemY));
	{
		for (auto& child : activeScene->getRootNode()->getChildren())
		{
			drawSceneNode(child);
		}

		handleEvent();
		popupMenu();
	}
	ImGui::PopStyleVar(1);
	ImGui::EndChild();

	acceptDragdrop(true);

	// End decorated text.
	ImGui::Separator(); ImGui::Spacing();
	ImGui::Text("  %d scene nodes.", activeScene->getNodeCount());
}

void SceneOutlinerWidget::drawSceneNode(std::shared_ptr<SceneNode> node)
{
	// This is an event draw or not.
	const bool bEvenDrawIndex = m_drawContext.drawIndex % 2 == 0;
	m_drawContext.drawIndex++;

	// This is a tree node or not.
	const bool bTreeNode = node->getChildren().size() > 0;

	const bool bVisibilityNodePrev = node->getVisibility();
	const bool bStaticNodePrev = node->getStatic();

	bool bEditingName = false;
	bool bPushEditNameDisable = false;
	bool bNodeOpen;
	bool bSelectedNode = false;

	// Visible and static style prepare.
	if (!bVisibilityNodePrev) ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.39f);
	if (!bStaticNodePrev) ImGui::PushStyleColor(ImGuiCol_Text, (ImVec4)ImColor::HSV(0.15f, 0.6f, 1.0f));
	{
		ImGuiTreeNodeFlags nodeFlags = ImGuiTreeNodeFlags_AllowItemOverlap | ImGuiTreeNodeFlags_SpanFullWidth;
		nodeFlags |= bTreeNode ? ImGuiTreeNodeFlags_OpenOnArrow : ImGuiTreeNodeFlags_Leaf;

		const bool bThisNodeSelected = m_sceneSelections->isSelected(SceneNodeSelctor(node));

		if (bThisNodeSelected)
		{
			nodeFlags |= ImGuiTreeNodeFlags_Selected;
		}

		auto& nodeNameUtf8 = node->getName();

		if (m_drawContext.bRenameing && bThisNodeSelected)
		{
			bNodeOpen = true;
			bEditingName = true;

			bPushEditNameDisable = true;
			ImGui::Indent();

			ImGui::Text("  %s  ", ICON_FA_ELLIPSIS);
			ImGui::SameLine();

			strcpy_s(m_drawContext.inputBuffer, nodeNameUtf8.c_str());
			if (ImGui::InputText(" ", m_drawContext.inputBuffer, IM_ARRAYSIZE(m_drawContext.inputBuffer), ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll))
			{
				std::string newName = m_drawContext.inputBuffer;
				if (!newName.empty())
				{
					SceneGraphUndoRecord("Change scene node name.");
					node->setName(addUniqueIdForName(newName));
				}
				m_drawContext.bRenameing = false;
			}

			if (!ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
			{
				m_drawContext.bRenameing = false;
			}
		}
		else
		{
			bNodeOpen = ImGui::TreeNodeEx(
				reinterpret_cast<void*>(static_cast<intptr_t>(node->getId())),
				nodeFlags, 
				" %s    %s", bTreeNode ? ICON_FA_FOLDER : ICON_FA_FAN, 
				nodeNameUtf8.c_str());
		}

		// update hover node.
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_RectOnly))
		{
			m_drawContext.hoverNode = node;
		}

		if(ImGui::IsItemClicked())
		{
			if (ImGui::GetIO().KeyCtrl)
			{
				if (m_sceneSelections->isSelected(SceneNodeSelctor(m_drawContext.hoverNode.lock())))
				{
					m_sceneSelections->removeSelect(SceneNodeSelctor(m_drawContext.hoverNode.lock()));
				}
				else
				{
					m_sceneSelections->addSelected(SceneNodeSelctor(m_drawContext.hoverNode.lock()));
				}
			}
			else
			{
				m_sceneSelections->clearSelections();
				m_sceneSelections->addSelected(SceneNodeSelctor(m_drawContext.hoverNode.lock()));
			}
		}

		// start drag drop.
		beginDragDrop(node);
		acceptDragdrop(false);

		auto colorBg = bEvenDrawIndex ?
			ImGui::GetStyleColorVec4(ImGuiCol_TableRowBg) :
			ImGui::GetStyleColorVec4(ImGuiCol_TableRowBgAlt);

		auto itemEndPosX = ImGui::GetCursorPosX();
		itemEndPosX += ImGui::GetItemRectSize().x;

		ImGui::GetWindowDrawList()->AddRectFilled(
			{ ImGui::GetItemRectMin().x - ImGui::GetStyle().ItemSpacing.x * 0.5f, ImGui::GetItemRectMin().y - 0.5f * m_drawContext.scenePaddingItemY },
			{ ImGui::GetItemRectMax().x + ImGui::GetStyle().ItemSpacing.x * 0.5f, ImGui::GetItemRectMax().y + 0.5f * m_drawContext.scenePaddingItemY },
			IM_COL32(colorBg.x * 255, colorBg.y * 255, colorBg.z * 255, colorBg.w * 255));

		if (!bEditingName)
		{
			ImGui::SameLine();
			handleDrawState(node);
		}

	}

	// Visible and static style pop.
	if (!bVisibilityNodePrev) ImGui::PopStyleVar();
	if (!bStaticNodePrev) ImGui::PopStyleColor();

	if (bNodeOpen)
	{
		if (bTreeNode)
		{
			for (const auto& child : node->getChildren())
			{
				drawSceneNode(child);
			}
		}

		if (!bEditingName)
		{
			ImGui::TreePop();
		}
	}

	if (bPushEditNameDisable)
	{
		ImGui::Unindent();
	}
}

void SceneOutlinerWidget::handleEvent()
{
	const auto bWindowHovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByPopup | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);
	if (!bWindowHovered)
	{
		return;
	}

	// Prepare seleted one node state.
	std::shared_ptr<SceneNode> selectedOneNode = nullptr;
	if (m_sceneSelections->getNum() == 1)
	{
		selectedOneNode = m_sceneSelections->getSelections()[0].node.lock();
	}

	const auto bLeftClick = ImGui::IsMouseClicked(0);
	const auto bRightClick = ImGui::IsMouseClicked(1);
	const auto bDoubleClick = ImGui::IsMouseDoubleClicked(0);

	// Click empty state.
	if (!ImGui::GetIO().KeyCtrl)
	{
		// Update selected node to root if no hover node.
		if ((bRightClick || bLeftClick) && !m_drawContext.hoverNode.lock())
		{
			m_sceneSelections->clearSelections();
		}
	}

	// Upadte rename state, only true when only one node selected.
	if (bDoubleClick && selectedOneNode)
	{
		m_drawContext.bRenameing = true;
	}

	if (bRightClick)
	{
		ImGui::OpenPopup(m_drawContext.kPopupMenuName);
	}
}

void SceneOutlinerWidget::popupMenu()
{
	auto activeScene = m_sceneManager->getActiveScene();

	if (!ImGui::BeginPopup(m_drawContext.kPopupMenuName))
	{
		return;
	}

	const bool bSelectedLessEqualOne = m_sceneSelections->getNum() <= 1;
	std::shared_ptr<SceneNode> selectedOneNode = nullptr;
	if (m_sceneSelections->getNum() == 1)
	{
		selectedOneNode = m_sceneSelections->getSelections()[0].node.lock();
	}

	if (selectedOneNode)
	{
		if (ImGui::MenuItem(ICON_NONE"   Rename"))
		{
			m_drawContext.bRenameing = true;
		}
	}

	if (m_sceneSelections->getNum() > 0)
	{
		if (ImGui::MenuItem(ICON_NONE"   Delete"))
		{
			SceneGraphUndoRecord("Delete scene node.");

			for (const auto& node : m_sceneSelections->getSelections())
			{
				auto nodePtr = node.node.lock();
				if (nodePtr && !nodePtr->isRoot())
				{
					activeScene->deleteNode(node.node.lock());
				}
			}
			m_sceneSelections->clearSelections();

			ImGui::EndPopup();
			return;
		}
	}

	if (bSelectedLessEqualOne)
	{
		static const std::string kEmptyNodeStr = std::string("  ") + ICON_FA_FAN + std::string("  Empty Scene Node");
		if (ImGui::MenuItem(kEmptyNodeStr.c_str()))
		{
			SceneGraphUndoRecord("Create scene node.");
			activeScene->createNode(addUniqueIdForName("Untitled"), selectedOneNode);

			ImGui::EndPopup();
			return;
		}

		static const std::string kSkyName = std::string("  ") + kIconSky;
		if (ImGui::MenuItem(kSkyName.c_str()))
		{
			SceneGraphUndoRecord("Create scene node with sky.");
			auto newNode = activeScene->createNode(addUniqueIdForName("Sky"), selectedOneNode);

			newNode->getTransform()->setRotation(glm::vec3(-0.7854f, 0.0f, 0.0f));
			newNode->getScene()->addComponent<SkyComponent>(std::make_shared<SkyComponent>(), newNode);
			newNode->setType(SceneNode::EType::Sky);
		}

		static const std::string kPostprocessName = std::string("  ") + kIconPostprocess;
		if (ImGui::MenuItem(kPostprocessName.c_str()))
		{
			SceneGraphUndoRecord("Create scene node with postprocess.");
			auto newNode = activeScene->createNode(addUniqueIdForName("Postprocess"), selectedOneNode);

			newNode->getScene()->addComponent<PostprocessVolumeComponent>(std::make_shared<PostprocessVolumeComponent>(), newNode);
			newNode->setType(SceneNode::EType::Postprocess);
		}
	}

	ImGui::EndPopup();
}

void SceneOutlinerWidget::beginDragDrop(std::shared_ptr<SceneNode> node)
{
	if (ImGui::BeginDragDropSource())
	{
		m_drawContext.dragingNodes.reserve(m_sceneSelections->getNum());
		for (const auto& s : m_sceneSelections->getSelections())
		{
			m_drawContext.dragingNodes.push_back(s.node);
		}
		m_drawContext.dragDropUUID = buildRuntimeUUID64u();

		ImGui::SetDragDropPayload(m_drawContext.kOutlinerDragDropName, &m_drawContext.dragDropUUID, sizeof(m_drawContext.dragDropUUID));
		ImGui::Text(node->getName().c_str());
		ImGui::EndDragDropSource();
	}
}

void SceneOutlinerWidget::acceptDragdrop(bool bRoot)
{
	auto activeScene = m_sceneManager->getActiveScene();

	if (bRoot)
	{
		if (ImGui::BeginDragDropTarget())
		{
			if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload(m_drawContext.kOutlinerDragDropName))
			{
				if (m_drawContext.dragingNodes.size() > 0)
				{
					SceneGraphUndoRecord("Change scene node relationship.");

					for (const auto& node : m_drawContext.dragingNodes)
					{
						if (auto nodePtr = node.lock())
						{
							activeScene->setParent(activeScene->getRootNode(), nodePtr);
						}
					}

					// Reset draging node.
					m_drawContext.dragingNodes.clear();
					m_drawContext.dragDropUUID = ~0;

					sortChildren(activeScene->getRootNode());
				}
			}
			ImGui::EndDragDropTarget();
		}
	}
	else
	{
		// Accept drag drop.
		if (auto hoverNode = m_drawContext.hoverNode.lock())
		{
			if (ImGui::BeginDragDropTarget())
			{
				if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload(m_drawContext.kOutlinerDragDropName))
				{
					if (m_drawContext.dragingNodes.size() > 0)
					{
						SceneGraphUndoRecord("Change scene node relationship.");

						for (auto& node : m_drawContext.dragingNodes)
						{
							if (auto nodePtr = node.lock())
							{
								if (activeScene->setParent(hoverNode, nodePtr))
								{
									activeScene->setDirty(true);
								}
							}
							// Reset draging node.
							m_drawContext.dragingNodes.clear();
							m_drawContext.dragDropUUID = ~0;
						}
	
						sortChildren(activeScene->getRootNode());
					}
				}
				ImGui::EndDragDropTarget();
			}
		}
	}
}

void SceneOutlinerWidget::handleDrawState(std::shared_ptr<SceneNode> node)
{
	const char* visibilityIcon = node->getVisibility() ? ICON_FA_EYE : ICON_FA_EYE_SLASH;
	const char* staticIcon = node->getStatic() ? ICON_FA_PERSON : ICON_FA_PERSON_WALKING;

	auto iconSizeEye = ImGui::CalcTextSize(ICON_FA_EYE_SLASH);
	auto iconSizeStatic = ImGui::CalcTextSize(ICON_FA_PERSON_WALKING);

	ImGui::BeginGroup();
	ImGui::PushID(node->getRuntimeIdName().c_str());

	auto eyePosX = ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - iconSizeEye.x - 1.0f;
	ImGui::SetCursorPosX(eyePosX);
	if (ImGui::Selectable(visibilityIcon))
	{
		SceneGraphUndoRecord("Change scene node visibility.");
		node->setVisibility(!node->getVisibility());
	}
	ui::hoverTip("Set scene node visibility.");

	ImGui::SameLine();
	ImGui::SetCursorPosX(eyePosX - iconSizeEye.x);

	bool bSeletcted = false;
	if (ImGui::Selectable(staticIcon, &bSeletcted, ImGuiSelectableFlags_None, { iconSizeStatic.x, iconSizeEye.y }))
	{
		SceneGraphUndoRecord("Change scene node static state.");
		node->setStatic(!node->getStatic());
	}
	ui::hoverTip("Set scene node static state.");

	ImGui::PopID();
	ImGui::EndGroup();
}

void SceneOutlinerWidget::rebuildSceneNodeNameMap()
{
	auto activeScene = m_sceneManager->getActiveScene();
	m_drawContext.cacheNodeNameMap.clear();

	activeScene->loopNodeTopToDown([&](std::shared_ptr<SceneNode> node)
	{
		m_drawContext.cacheNodeNameMap.insert({ node->getName(), 0 });
	}, activeScene->getRootNode());
}

std::string SceneOutlinerWidget::addUniqueIdForName(const std::string& name)
{
	std::string removePrefix = name.substr(0, name.find_last_of("("));
	removePrefix = removePrefix.substr(0, removePrefix.find_last_not_of(" \t\f\v\n\r") + 1);

	size_t& id = m_drawContext.cacheNodeNameMap[removePrefix];

	if (id == 0)
	{
		id++;
		return std::move(removePrefix);
	}
	else
	{
		std::string uniqueName = std::format("{} ({})", removePrefix, id);
		id++;
		return std::move(uniqueName);
	}
}

void SceneOutlinerWidget::sortChildren(std::shared_ptr<SceneNode> node)
{
	if (node)
	{
		auto& children = node->getChildren();

		std::sort(std::begin(children), std::end(children), [&](const auto& a, const auto& b)
		{
			return a->getName() < b->getName();
		});

		for (auto& child : children)
		{
			sortChildren(child);
		}
	}
}
