#include "detail.h"
#include "imgui/ui.h"
#include "imgui/region_string.h"
#include "../editor.h"
#include "scene_outliner.h"
#include <scene/component.h>
#include "component_draw/component_draw.h"
#include <scene/component/static_mesh.h>
#include <scene/component/terrain.h>
using namespace engine;
using namespace engine::ui;

RegionStringInit Detail_Title("Detail_Title", "Detail", "Detail");
const static std::string ICON_DETAIL = ICON_FA_LIST;

static const std::string DETAIL_SearchIcon = ICON_FA_MAGNIFYING_GLASS;
static const std::string DETAIL_AddIcon    = std::string("  ") + ICON_FA_SQUARE_PLUS + std::string("  ADD  ");

WidgetDetail::WidgetDetail(Editor* editor)
	: Widget(editor, "Detail")
{

}

WidgetDetail::~WidgetDetail() noexcept 
{ 

}

void WidgetDetail::onInit() 
{
	m_name = combineIcon(Detail_Title, ICON_DETAIL);
}

void WidgetDetail::onRelease() 
{

}

void WidgetDetail::onTick(const RuntimeModuleTickData& tickData, VulkanContext* context)
{

}


void WidgetDetail::onVisibleTick(const RuntimeModuleTickData& tickData)
{
	ImGui::Spacing();

	if (m_editor->getSceneNodeSelected().empty())
	{
		ImGui::TextDisabled("No selected node to inspect.");
		return;
	}

	if (m_editor->getSceneNodeSelected().size() > 1)
	{
		ImGui::TextDisabled("Multi node detail inspect still no support.");
		return;
	}

	std::shared_ptr<SceneNode> selectedNode;
	for (auto node : m_editor->getSceneNodeSelected()) 
	{
		if (node)
		{
			selectedNode = node.node.lock();
			break;
		} 
	}

	if (!selectedNode)
	{
		// No valid scene node.
		return;
	}

	// Print detail info.
	ImGui::TextDisabled("%s with runtime ID %d and depth %d.", selectedNode->getName().c_str(), selectedNode->getId(), selectedNode->getDepth());

	ImGui::Separator();
	ImGui::Spacing();

	 
	auto transform = selectedNode->getTransform();

	const float sizeLable = ImGui::GetFontSize() * 1.5f;

	bool bChangeTransform = false;
	math::vec3 anglesRotate = math::degrees(transform->getRotation());

	bChangeTransform |= ui::drawVector3("  P  ", transform->getTranslation(), math::vec3(0.0f), sizeLable);
	bChangeTransform |= ui::drawVector3("  R  ", anglesRotate, math::vec3(0.0f), sizeLable);
	bChangeTransform |= ui::drawVector3("  S  ", transform->getScale(), math::vec3(1.0f), sizeLable);


	if (bChangeTransform)
	{
		transform->getRotation() = math::radians(anglesRotate);
		transform->invalidateWorldMatrix();
	}

	ImGui::Spacing();

	ui::helpMarker(
		"Scene node state can use for accelerate engine speed.\n"
		"When invisible, renderer can cull this entity before render and save render time\n"
		"But still can simulate or tick logic on entity.\n"
		"When un movable, renderer can do some cache for mesh, skip too much dynamic objects."); ImGui::SameLine();

	const bool bCanSetVisiblity = selectedNode->canSetNewVisibility();
	const bool bCanSetStatic = selectedNode->canSetNewStatic();

	bool bVisibleState = selectedNode->getVisibility();
	bool bMovableState = !selectedNode->getStatic();

	ui::disableLambda([&]() 
	{
		if (ImGui::Checkbox("Show", &bVisibleState))
		{
			SceneGraphUndoRecord("Change scene node visibility.");
			selectedNode->setVisibility(!selectedNode->getVisibility());
		}
		ui::hoverTip("Scene node visibility state.");
	}, !bCanSetVisiblity);

	ImGui::SameLine();

	ui::disableLambda([&]()
	{
		if (ImGui::Checkbox("Movable", &bMovableState))
		{
			SceneGraphUndoRecord("Change scene node static state.");
			selectedNode->setStatic(!selectedNode->getStatic());
		}
		ui::hoverTip("Entity movable state.");
	}, !bCanSetStatic);

	ImGui::Separator();

	drawComponent(selectedNode);
}

void WidgetDetail::drawComponent(std::shared_ptr<SceneNode> node)
{
	if (ImGui::BeginTable("Add UIC##", 2))
	{
		const float sizeLable = ImGui::GetFontSize();
		ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, sizeLable * 4.0f);
		ImGui::TableSetupColumn("", ImGuiTableColumnFlags_None);

		ImGui::TableNextColumn();

		if (ImGui::Button((DETAIL_AddIcon).c_str()))
		{
			ImGui::OpenPopup("##XComponentContextMenu_Add");
		}

		if (ImGui::BeginPopup("##XComponentContextMenu_Add"))
		{
			ImGui::TextDisabled("New  Components");
			ImGui::Separator();

			bool bExistOneNewComponent = false;

			auto drawAddNode = [&]<typename T>(const std::string & showName)
			{
				const bool bShouldAdd = !node->hasComponent<T>();

				if (bShouldAdd)
				{
					bExistOneNewComponent = true;
					ImGui::PushID(typeid(T).name());
					if (ImGui::Selectable(showName.c_str()))
					{
						node->getScene()->addComponent<T>(std::make_shared<T>(), node);
					}
					ImGui::PopID();
				}
			};

			drawAddNode.template operator()<PMXComponent>(kIconPMX);
			drawAddNode.template operator()<MMDCameraComponent>(kIconMMDCamera);
			drawAddNode.template operator()<StaticMeshComponent>(kIconStaticMesh);
			drawAddNode.template operator()<SkyComponent>(kIconSky);
			drawAddNode.template operator()<PostprocessVolumeComponent>(kIconPostprocess);
			drawAddNode.template operator()<TerrainComponent>(kIconTerrain);

			if (!bExistOneNewComponent)
			{
				ImGui::TextDisabled("Non-Component");
			}
			ImGui::EndPopup();
		}

		ui::hoverTip("Add new component for entity.");

		ImGui::TableNextColumn();
		m_filter.Draw((DETAIL_SearchIcon).c_str());

		ImGui::EndTable();
	}

	ImGui::TextDisabled("Additional components.");

	const ImGuiTreeNodeFlags treeNodeFlags =
		ImGuiTreeNodeFlags_DefaultOpen |
		ImGuiTreeNodeFlags_Framed |
		ImGuiTreeNodeFlags_SpanAvailWidth |
		ImGuiTreeNodeFlags_AllowItemOverlap |
		ImGuiTreeNodeFlags_FramePadding;

	for (auto& [showName, drawer] : kDrawComponentMap)
	{
		if (node->hasComponent(drawer.typeName))
		{
			ImGui::PushID(drawer.typeName);

			ImVec2 contentRegionAvailable = ImGui::GetContentRegionAvail();
			float lineHeight = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
			ImGui::Spacing();
			ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2{ 4.0f, 4.0f });
			bool open = ImGui::TreeNodeEx("TreeNodeForComp", treeNodeFlags, showName.c_str());
			ImGui::PopStyleVar();

			ImGui::SameLine(contentRegionAvailable.x - lineHeight + GImGui->Style.FramePadding.x);
			if (ImGui::Button(ICON_FA_XMARK, ImVec2{ lineHeight, lineHeight }))
			{
				node->getScene()->removeComponent(node, drawer.typeName);

				if (open)
				{
					ImGui::TreePop();
				}
				ImGui::PopID();

				continue;
			}
			ui::hoverTip("Remove component.");

			if (open)
			{
				ImGui::PushID("Widget");
				ImGui::Spacing();


				// Draw callback.
				drawer.drawFunc(node);

				ImGui::Spacing();
				ImGui::Separator();
				ImGui::PopID();

				ImGui::TreePop();
			}

			ImGui::PopID();
		}
	}
}
