#include "detail.h"

#include <scene/component.h>

using namespace engine;
using namespace engine::ui;

const static std::string kIconDetail = ICON_FA_LIST;

static const std::string DETAIL_SearchIcon = ICON_FA_MAGNIFYING_GLASS;
static const std::string DETAIL_AddIcon    = std::string("  ") + ICON_FA_SQUARE_PLUS + std::string("  ADD  ");

WidgetDetail::WidgetDetail(size_t index)
	: WidgetBase(
		combineIcon("Detail", kIconDetail).c_str(),
		combineIcon(combineIndex("Detail", index), kIconDetail).c_str())
{

}

WidgetDetail::~WidgetDetail() noexcept 
{ 

}

void WidgetDetail::onInit() 
{
	m_onSelectorChange = Editor::get()->onOutlinerSelectionChange.addRaw(this, &WidgetDetail::onOutlinerSelectionChange);
}

void WidgetDetail::onRelease() 
{
	Editor::get()->onOutlinerSelectionChange.remove(m_onSelectorChange);
}

void WidgetDetail::onTick(const RuntimeModuleTickData& tickData, VulkanContext* context)
{

}


void WidgetDetail::onVisibleTick(const RuntimeModuleTickData& tickData)
{
	ZoneScoped;

	ImGui::Spacing();

	if (!m_selector || m_selector->empty())
	{
		ImGui::TextDisabled("No selected node to inspect.");
		return;
	}

	if (m_selector->getNum() > 1)
	{
		ImGui::TextDisabled("Multi node detail inspect still no support.");
		return;
	}

	std::shared_ptr<SceneNode> selectedNode;
	for (const auto& node : m_selector->getSelections())
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
	ImGui::TextDisabled("Inspecting %s with runtime ID %d.", 
		selectedNode->getName().c_str(), selectedNode->getId());

	ImGui::Separator();
	ImGui::Spacing();

	selectedNode->getTransform()->uiDrawComponent();
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
			selectedNode->setVisibility(!selectedNode->getVisibility());
		}
		ui::hoverTip("Scene node visibility state.");
	}, !bCanSetVisiblity);

	ImGui::SameLine();

	ui::disableLambda([&]()
	{
		if (ImGui::Checkbox("Movable", &bMovableState))
		{
			selectedNode->setStatic(!selectedNode->getStatic());
		}
		ui::hoverTip("Entity movable state.");
	}, !bCanSetStatic);

	ImGui::Separator();

	drawComponent(selectedNode);
}

void WidgetDetail::drawComponent(std::shared_ptr<SceneNode> node)
{
	auto assetList = rttr::type::get<Component>().get_derived_classes();

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
			for (auto& assetType : assetList)
			{
				const auto& method = assetType.get_method("uiComponentReflection");
				if (method.is_static() && method.is_valid())
				{
					rttr::variant returnValue = method.invoke({});
					if (returnValue.is_valid() && returnValue.is_type<UIComponentReflectionDetailed>())
					{
						const auto& meta = returnValue.get_value<UIComponentReflectionDetailed>();
						if (meta.bOptionalCreated)
						{
							std::string typeName = assetType.get_name().data();
							const bool bShouldAdd = !node->hasComponent(typeName);
							if (bShouldAdd)
							{
								bExistOneNewComponent = true;
								ImGui::PushID(typeName.c_str());
								if (ImGui::Selectable(meta.iconCreated.c_str()))
								{
									auto newComp = assetType.create();
									std::shared_ptr<Component> comp = newComp.get_value<std::shared_ptr<Component>>();
									node->getScene()->addComponent(typeName, comp, node);

									CHECK(comp->isValid());
								}
								ImGui::PopID();
							}
						}
					}
				}
			}

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

	for (auto& assetType : assetList)
	{
		const auto& method = assetType.get_method("uiComponentReflection");
		if (method.is_static() && method.is_valid())
		{
			rttr::variant returnValue = method.invoke({});
			if (returnValue.is_valid() && returnValue.is_type<UIComponentReflectionDetailed>())
			{
				const auto& meta = returnValue.get_value<UIComponentReflectionDetailed>();
				std::string typeName = assetType.get_name().data();
				if (meta.bOptionalCreated && node->hasComponent(typeName))
				{
					ImGui::PushID(meta.iconCreated.c_str());

					ImVec2 contentRegionAvailable = ImGui::GetContentRegionAvail();
					float lineHeight = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
					ImGui::Spacing();
					ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2{ 4.0f, 4.0f });
					bool open = ImGui::TreeNodeEx("TreeNodeForComp", treeNodeFlags, meta.iconCreated.c_str());
					ImGui::PopStyleVar();

					ImGui::SameLine(contentRegionAvailable.x - lineHeight + GImGui->Style.FramePadding.x);
					if (ImGui::Button(ICON_FA_XMARK, ImVec2{ lineHeight, lineHeight }))
					{
						node->getScene()->removeComponent(node, typeName);

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

						node->getComponent(typeName)->uiDrawComponent();

						ImGui::Spacing();
						ImGui::Separator();
						ImGui::PopID();

						ImGui::TreePop();
					}

					ImGui::PopID();
				}
			}
		}
	}
}

void WidgetDetail::onOutlinerSelectionChange(Selection<SceneNodeSelctor>& selector)
{
	m_selector = &selector;
}
