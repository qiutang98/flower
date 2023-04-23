#include "component_draw.h"
#include <imgui/ui.h>
#include <scene/scene.h>
#include "../editor/editor.h"
#include "../editor/widgets/project_content.h"
#include <scene/component/terrain.h>

using namespace engine;
using namespace engine::ui;

void drawHeightFieldSelect(std::shared_ptr<SceneNode> node, std::shared_ptr<TerrainComponent> comp, bool bSync, std::function<void(const UUID& id)>&& func)
{
	auto* assetSystem = Editor::get()->getAssetSystem();
	auto* context = Editor::get()->getContext();

	if (ImGui::BeginMenu("Project"))
	{
		const auto& map = assetSystem->getAssetMap(EAssetType::Texture);
		for (const auto& texd : map)
		{
			const auto& asset = assetSystem->getAsset(texd);
			if (ImGui::MenuItem((std::string("  ") + ICON_FA_IMAGE"   " + asset->getRelativePathUtf8()).c_str()))
			{
				func(texd);
				comp->loadTexturesByUUID(bSync);
			}
		}
		ImGui::EndMenu();
	}
}

void ComponentDrawer::drawTerrain(std::shared_ptr<SceneNode> node)
{
	std::shared_ptr<TerrainComponent> comp = node->getComponent<TerrainComponent>();

	ProjectContentWidget* contentWidget = Editor::get()->getContentWidget();
	auto* assetSystem = Editor::get()->getAssetSystem();

	ImGui::Unindent();
	ui::beginGroupPanel("Basic");

	static const std::string selectButtonName = kIconTerrain + " Chose ";
	if (ImGui::Button(selectButtonName.c_str()))
		ImGui::OpenPopup("TerrainHeightFieldSelectPopUp");
	if (ImGui::BeginPopup("TerrainHeightFieldSelectPopUp"))
	{
		ImGui::TextDisabled("Select HeightField...");
		ImGui::Spacing();

		drawHeightFieldSelect(node, comp, true, [&](const UUID& id){ comp->setHeightField(id); });
		ImGui::EndPopup();
	}

	if (ImGui::Button(" Mask Chose "))
		ImGui::OpenPopup("TerrainMaskSelectPopUp");
	if (ImGui::BeginPopup("TerrainMaskSelectPopUp"))
	{
		ImGui::TextDisabled("Select Mask...");
		ImGui::Spacing();

		drawHeightFieldSelect(node, comp, true, [&](const UUID& id) { comp->setMask(id); });
		ImGui::EndPopup();
	}

	ui::endGroupPanel();

	auto copySetting = comp->getSetting();
	ImGui::DragFloat("Accurate", &copySetting.primitivePixelLengthTarget, 1.0f, 1.0f, 30.0f);
	ImGui::InputFloat("Terrain max height", &copySetting.dumpFactor);
	comp->changeSetting(copySetting);

	ImGui::Indent();
}