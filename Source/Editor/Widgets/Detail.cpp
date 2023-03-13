#include "Pch.h"
#include "Detail.h"
#include "../Editor.h"
#include "DrawComponent/DrawComponent.h"

using namespace Flower;
using namespace Flower::UI;

static const std::string DETAIL_DetailIcon = ICON_FA_LIST;
static const std::string DETAIL_SearchIcon = ICON_FA_MAGNIFYING_GLASS;
static const std::string DETAIL_AddIcon    = std::string("  ") + ICON_FA_SQUARE_PLUS + std::string("  ADD  ");

WidgetDetail::WidgetDetail()
	: Widget("  " + DETAIL_DetailIcon + "  Detail")
{

}

WidgetDetail::~WidgetDetail() noexcept 
{ 

}

void WidgetDetail::onInit() 
{
	m_outliner = GEditor->getSceneOutliner();

}

void WidgetDetail::onRelease() 
{

}

void WidgetDetail::onTick(const RuntimeModuleTickData& tickData)
{

}


void WidgetDetail::onVisibleTick(const RuntimeModuleTickData& tickData)
{
	ImGui::Spacing();

	Scene* scene = m_outliner->getScene();
	if (!scene)
	{
		ImGui::TextDisabled("No active scene to inspect.");
		return;
	}

	std::shared_ptr<SceneNode> selectedNode = m_outliner->getSelectedNode();
	if (!selectedNode || selectedNode->isRoot())
	{
		ImGui::TextDisabled("No selected node to inspect.");
		return;
	}

	if (selectedNode)
	{
		ImGui::TextDisabled("%s with runtime ID %d", selectedNode->getName().c_str(), selectedNode->getId());
	}
	ImGui::Separator();
	ImGui::Spacing();

	auto transform = selectedNode->getTransform();

	const float sizeLable = ImGui::GetFontSize();
	const auto cacheTranslation = transform->getTranslation();
	const auto cacheScale = transform->getTranslation();

	glm::vec3 rotation = glm::degrees(glm::eulerAngles(transform->getRotation()));
	UIHelper::drawVector3(" P ", transform->getTranslation(), glm::vec3(0.0f), sizeLable);
	UIHelper::drawVector3(" R ", rotation, glm::vec3(0.0f), sizeLable);
	UIHelper::drawVector3(" S ", transform->getScale(), glm::vec3(1.0f), sizeLable);
	glm::quat rotationQ = glm::quat(glm::radians(rotation));
	if (rotationQ != transform->getRotation())
	{
		transform->setRotation(rotationQ);
	}

	if (cacheTranslation != transform->getTranslation() || cacheScale != transform->getScale())
	{
		transform->invalidateWorldMatrix();
	}

	ImGui::Spacing();

	UIHelper::helpMarker(
		"Scene node state can use for accelerate engine speed.\n"
		"When invisible, renderer can cull this entity before render and save render time\n"
		"But still can simulate or tick logic on entity.\n"
		"When un movable, renderer can do some cache for mesh, skip too much dynamic objects."); ImGui::SameLine();

	const bool bCanSetVisiblity = selectedNode->canSetNewVisibility();
	
	if (!bCanSetVisiblity)
	{
		ImGui::BeginDisabled();
	}
	bool bVisibleState = selectedNode->getVisibility();
	if (ImGui::Checkbox("Show", &bVisibleState))
	{
		selectedNode->setVisibility(!selectedNode->getVisibility());
	}
	UIHelper::hoverTip("Scene node visibility state.");
	if (!bCanSetVisiblity)
	{
		ImGui::EndDisabled();
	}

	ImGui::SameLine();

	const bool bCanSetStatic = selectedNode->canSetNewStatic();
	if (!bCanSetStatic)
	{
		ImGui::BeginDisabled();
	}
	bool bStatic = selectedNode->getStatic();
	if (ImGui::Checkbox("Movable", &bStatic))
	{
		selectedNode->setStatic(!selectedNode->getStatic());
	}
	UIHelper::hoverTip("Entity movable state.");
	if (!bCanSetStatic)
	{
		ImGui::EndDisabled();
	}

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
			ImGui::TextDisabled("New  Components:");
			ImGui::Separator();

			bool bExistOneNewComponent = false;

			auto drawAddNode = [&]<typename T>(const std::string& showName)
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

			drawAddNode.template operator()<PMXComponent>(GIconPMX);
			drawAddNode.template operator()<StaticMeshComponent>(GIconStaticMesh);
			drawAddNode.template operator()<LandscapeComponent>(GIconLandscape);
			drawAddNode.template operator()<SunSkyComponent>(GIconSunSky);
			drawAddNode.template operator()<SpotLightComponent>(GIconSpotLight);
			drawAddNode.template operator()<ReflectionCaptureComponent>(GIconReflectionCapture);
			drawAddNode.template operator()<PostprocessVolumeComponent>(GIconPostprocessVolume);
			if (!bExistOneNewComponent)
			{
				ImGui::TextDisabled("Non-Component");
			}
			ImGui::EndPopup();
		}

		UIHelper::hoverTip("Add new component for entity.");

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

	for (auto& [showName, drawer] : GDrawComponentMap)
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

			ImGui::SameLine(contentRegionAvailable.x - lineHeight * 0.5f);
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
			UIHelper::hoverTip("Remove component.");

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