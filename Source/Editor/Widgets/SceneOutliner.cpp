#include "Pch.h"
#include "SceneOutliner.h"
#include <ImGui/ImGuiInternal.h>
#include "DrawComponent/DrawComponent.h"

using namespace Flower;
using namespace Flower::UI;

const std::string SCENEOUTLINER_GTileIcon = ICON_FA_LAYER_GROUP;

constexpr float cScenePaddingItemY = 5.0f;

WidgetSceneOutliner::WidgetSceneOutliner()
	: Widget("  " + SCENEOUTLINER_GTileIcon + "  Hierarchy")
{

}

WidgetSceneOutliner::~WidgetSceneOutliner() noexcept
{

}

void WidgetSceneOutliner::onInit()
{
	m_manager = GEngine->getRuntimeModule<SceneManager>();
	CHECK(m_manager != nullptr && "You must register one scene manager!");
}

void WidgetSceneOutliner::onTick(const Flower::RuntimeModuleTickData& tickData)
{
	m_scene = m_manager->getScenes();
	CHECK(m_scene != nullptr);
}

void WidgetSceneOutliner::onRelease()
{

}

void WidgetSceneOutliner::onVisibleTick(const Flower::RuntimeModuleTickData& tickData)
{
	m_hoverNode.reset();

	ImGui::Spacing();
	ImGui::TextDisabled("%s  Active flower scene:  %s.", ICON_FA_FAN, m_scene->getName().c_str());
	ImGui::Separator();

	const float footerHeightToReserve = ImGui::GetStyle().ItemSpacing.y + ImGui::GetFrameHeightWithSpacing();
	ImGui::BeginChild("HierarchyScrollingRegion", ImVec2(0, -footerHeightToReserve), true, ImGuiWindowFlags_HorizontalScrollbar);

	ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(ImGui::GetStyle().ItemSpacing.x, cScenePaddingItemY));

	m_drawIndex = 0;
	for (auto& child : m_scene->getRootNode()->getChildren())
	{
		drawSceneNode(child);
	}

	

	handleEvent();

	popupMenu();

	ImGui::PopStyleVar(1);
	
	ImGui::EndChild();
	acceptDragdrop(true);
	ImGui::Separator();
	ImGui::Text("  %d scene nodes.", m_scene->getNodeCount());
}

void WidgetSceneOutliner::drawSceneNode(std::shared_ptr<Flower::SceneNode> node)
{
	const bool bVisibilityNodePrev = node->getVisibility();
	const bool bStaticNodePrev = node->getStatic();

	if (!bVisibilityNodePrev)
	{
		ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.39f);
	}
	if (!bStaticNodePrev)
	{
		ImGui::PushStyleColor(ImGuiCol_Text, (ImVec4)ImColor::HSV(0.15f, 0.6f, 1.0f));
	}
	const bool bEvenDrawIndex = m_drawIndex % 2 == 0;
	m_drawIndex ++;

	const bool bRootNode = node->getDepth() == 0;
	const bool bTreeNode = node->getChildren().size() > 0;

	m_bExpandedToSelection = false;
	bool bSelectedNode = false;

	ImGuiTreeNodeFlags nodeFlags = ImGuiTreeNodeFlags_AllowItemOverlap | ImGuiTreeNodeFlags_SpanFullWidth;
	nodeFlags |= bTreeNode ? ImGuiTreeNodeFlags_OpenOnArrow : ImGuiTreeNodeFlags_Leaf;

	if (const auto selectedNode = m_selectedNode.lock())
	{
		nodeFlags |= selectedNode->getId() == node->getId() ? ImGuiTreeNodeFlags_Selected : nodeFlags;

		// Expanded whole scene tree to select item.
		if (m_bExpandToSelection)
		{
			if (selectedNode->getParent() && selectedNode->getParent()->getId() == node->getId())
			{
				ImGui::SetNextItemOpen(true);
				m_bExpandedToSelection = true;
			}
		}
	}

	bool bEditingName = false;
	bool bPushEditNameDisable = false;
	bool bNodeOpen;
	if (m_bRenameing && m_selectedNode.lock()->getId() == node->getId())
	{
		bNodeOpen = true;
		bEditingName = true;

		bPushEditNameDisable = true;
		ImGui::Indent();

		ImGui::Text("  %s  ", ICON_FA_ELLIPSIS);
		ImGui::SameLine();

		strcpy_s(m_inputBuffer, node->getName().c_str());
		if (ImGui::InputText(" ", m_inputBuffer, IM_ARRAYSIZE(m_inputBuffer), ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll))
		{
			node->setName(m_inputBuffer);
			m_bRenameing = false;
		}

		if (!ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
		{
			m_bRenameing = false;
		}
	}
	else
	{
		bNodeOpen = ImGui::TreeNodeEx(reinterpret_cast<void*>(static_cast<intptr_t>(node->getId())), nodeFlags, " %s    %s", bTreeNode ? ICON_FA_FOLDER : ICON_FA_FAN, node->getName().c_str());
	}

	if ((nodeFlags & ImGuiTreeNodeFlags_Selected) && m_bExpandToSelection)
	{
		m_selectedNodeRect = ImGui::GetCurrentContext()->LastItemData.Rect;
	}

	// update hover node.
	if (ImGui::IsItemHovered(ImGuiHoveredFlags_RectOnly))
	{
		m_hoverNode = node;
	}

	// start drag drop.
	beginDragDrop(node);
	acceptDragdrop(false);

	auto colorBg = bEvenDrawIndex ? 
		ImGui::GetStyleColorVec4(ImGuiCol_TableRowBg):
		ImGui::GetStyleColorVec4(ImGuiCol_TableRowBgAlt);

	auto itemEndPosX = ImGui::GetCursorPosX();
	itemEndPosX += ImGui::GetItemRectSize().x;

	ImGui::GetWindowDrawList()->AddRectFilled(
		{ ImGui::GetItemRectMin().x - ImGui::GetStyle().ItemSpacing.x * 0.5f, ImGui::GetItemRectMin().y - 0.5f * cScenePaddingItemY },
		{ ImGui::GetItemRectMax().x + ImGui::GetStyle().ItemSpacing.x * 0.5f, ImGui::GetItemRectMax().y + 0.5f * cScenePaddingItemY },
		IM_COL32(colorBg.x * 255, colorBg.y * 255, colorBg.z * 255, colorBg.w * 255));

	if (!bEditingName)
	{
		ImGui::SameLine();
		handleDrawState(node);
	}

	if (!bVisibilityNodePrev)
	{
		ImGui::PopStyleVar();
	}
	if (!bStaticNodePrev)
	{
		ImGui::PopStyleColor();
	}
	
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

void WidgetSceneOutliner::setSelectedNode(std::weak_ptr<Flower::SceneNode> node)
{
	m_bExpandToSelection = true;
	m_selectedNode = node;
}

void WidgetSceneOutliner::handleEvent()
{
	if (m_bExpandToSelection && !m_bExpandedToSelection)
	{
		ImGui::ScrollToBringRectIntoView(ImGui::GetCurrentWindow(), m_selectedNodeRect);
		m_bExpandToSelection = false;
	}

	const auto bWindowHovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByPopup | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);
	if (!bWindowHovered)
	{
		return;
	}

	const auto bLeftClick = ImGui::IsMouseClicked(0);
	const auto bRightClick = ImGui::IsMouseClicked(1);
	const auto bDoubleClick = ImGui::IsMouseDoubleClicked(0);

	// Update left click node.
	if (bLeftClick && m_hoverNode.lock())
	{
		m_leftClickNode = m_hoverNode;
	}

	// Update right click node.
	if (bRightClick && m_hoverNode.lock())
	{
		m_rightClickNode = m_hoverNode;
	}

	// Update selected node.
	if (bLeftClick || bDoubleClick || bRightClick)
	{
		if (m_hoverNode.lock())
		{
			setSelectedNode(m_hoverNode);
		}
	}

	if (bDoubleClick && !m_selectedNode.lock()->isRoot())
	{
		m_bRenameing = true;
	}

	if (bRightClick)
	{
		ImGui::OpenPopup(m_popupMenuName);
	}

	// Update selected node to root if no hover node.
	if ((bRightClick || bLeftClick) && !m_hoverNode.lock())
	{
		setSelectedNode(m_scene->getRootNode());
	}

	if (ImGui::IsMouseReleased(0) && m_leftClickNode.lock() && !m_leftClickNode.lock()->isRoot())
	{
		if (m_hoverNode.lock() && m_hoverNode.lock()->getId() == m_leftClickNode.lock()->getId())
		{
			setSelectedNode(m_leftClickNode);
		}
		m_leftClickNode = m_scene->getRootNode();
	}
	m_bExpandToSelection = true;
}

void WidgetSceneOutliner::popupMenu()
{
	if (auto node = m_selectedNode.lock())
	{
		if (!ImGui::BeginPopup(m_popupMenuName))
			return;

		if (!node->isRoot())
		{
			if (ImGui::MenuItem("Rename"))
			{
				m_bRenameing = true;
			}

			if (ImGui::MenuItem("Delete"))
			{
				if (auto scene = node->getScene())
				{
					if (!node->isRoot())
					{
						scene->deleteNode(node);
					}
				}
			}

			ImGui::Separator();
		}

		static const std::string  GFANEmptyNode = std::string("  ") + ICON_FA_FAN + std::string("  Empty Scene Node");
		if (ImGui::MenuItem(GFANEmptyNode.c_str()))
		{
			m_scene->createNode("SceneNode", node);
		}

		if (ImGui::MenuItem(GIconLandscape.c_str()))
		{
			auto newNode = m_scene->createNode("Landscape", node);
			newNode->getScene()->addComponent<LandscapeComponent>(std::make_shared<LandscapeComponent>(), newNode);
		}

		if (ImGui::MenuItem(GIconSunSky.c_str()))
		{
			auto newNode = m_scene->createNode("SunSky", node);
			newNode->getTransform()->setRotation(glm::quat(glm::radians(glm::vec3(45, 45, 0))));
			newNode->getScene()->addComponent<SunSkyComponent>(std::make_shared<SunSkyComponent>(), newNode);
		}
		if (ImGui::MenuItem(GIconSpotLight.c_str()))
		{
			auto newNode = m_scene->createNode("SpotLight", node);
			newNode->getTransform()->setRotation(glm::quat(glm::radians(glm::vec3(45, 45, 0))));
			newNode->getScene()->addComponent<SpotLightComponent>(std::make_shared<SpotLightComponent>(), newNode);
		}
		if (ImGui::MenuItem(GIconPostprocessVolume.c_str()))
		{
			auto newNode = m_scene->createNode("PostprocessVolume", node);
			newNode->getScene()->addComponent<PostprocessVolumeComponent>(std::make_shared<PostprocessVolumeComponent>(), newNode);
		}

		ImGui::EndPopup();
	}
}

void WidgetSceneOutliner::beginDragDrop(std::shared_ptr<Flower::SceneNode> node)
{
	if (ImGui::BeginDragDropSource())
	{
		m_dragingNode.dragingNode = node;
		m_dragingNode.id = node->getId();

		ImGui::SetDragDropPayload(m_outlinerDragDropName, &m_dragingNode.id, sizeof(size_t));

		ImGui::Text(node->getName().c_str());

		ImGui::EndDragDropSource();
	}
}

void WidgetSceneOutliner::acceptDragdrop(bool bRoot)
{
	if (bRoot)
	{
		if (ImGui::BeginDragDropTarget())
		{
			if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload(m_outlinerDragDropName))
			{
				if (const auto payloadNode = m_dragingNode.dragingNode.lock())
				{
					if (m_dragingNode.id != DragDropWrapper::UNVALID_ID)
					{
						if (m_scene->setParent(m_scene->getRootNode(), payloadNode))
						{
							// Reset draging node.
							m_dragingNode.dragingNode.reset();
							m_dragingNode.id = DragDropWrapper::UNVALID_ID;
						}
					}
				}
			}
			ImGui::EndDragDropTarget();
		}
	}
	else
	{
		// Accept drag drop.
		if (const auto hoverActiveNode = m_hoverNode.lock())
		{
			if (ImGui::BeginDragDropTarget())
			{
				if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload(m_outlinerDragDropName))
				{
					if (const auto payloadNode = m_dragingNode.dragingNode.lock())
					{
						if (m_dragingNode.id != DragDropWrapper::UNVALID_ID)
						{
							if (m_scene->setParent(hoverActiveNode, payloadNode))
							{
								// Reset draging node.
								m_dragingNode.dragingNode.reset();
								m_dragingNode.id = DragDropWrapper::UNVALID_ID;
							}
						}
					}
				}
				ImGui::EndDragDropTarget();
			}
		}
	}
}

void WidgetSceneOutliner::handleDrawState(std::shared_ptr<Flower::SceneNode> node)
{
	const char* visibilityIcon = node->getVisibility() ? ICON_FA_EYE : ICON_FA_EYE_SLASH;
	const char* staticIcon = node->getStatic() ? ICON_FA_PERSON : ICON_FA_PERSON_WALKING;

	auto iconSizeEye = ImGui::CalcTextSize(ICON_FA_EYE_SLASH);
	auto iconSizeStatic = ImGui::CalcTextSize(ICON_FA_PERSON_WALKING);

	ImGui::BeginGroup();

	auto eyePosX = ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - iconSizeEye.x - 1.0f;
	ImGui::SetCursorPosX(eyePosX);
	if (ImGui::Selectable(visibilityIcon))
	{
		node->setVisibility(!node->getVisibility());
	}
	UIHelper::hoverTip("Set scene node visibility.");

	ImGui::SameLine();


	ImGui::SetCursorPosX(eyePosX - iconSizeEye.x);

	bool bSeletcted = false;

	if (ImGui::Selectable(staticIcon, &bSeletcted, ImGuiSelectableFlags_None, { iconSizeStatic.x, iconSizeEye.y }))
	{
		node->setStatic(!node->getStatic());
	}
	UIHelper::hoverTip("Set scene node static state.");
	ImGui::EndGroup();
}
