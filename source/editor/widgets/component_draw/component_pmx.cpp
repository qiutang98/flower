#include "component_draw.h"
#include <imgui/ui.h>
#include <scene/scene.h>
#include "../editor/editor.h"
#include "../editor/widgets/project_content.h"
#include <scene/component/pmx.h>

using namespace engine;
using namespace engine::ui;

void drawPMXSelect(
	std::shared_ptr<SceneNode> node,
	std::shared_ptr<PMXComponent> comp)
{
	auto* assetSystem = Editor::get()->getAssetSystem();
	auto* context = Editor::get()->getContext();

	if (ImGui::BeginMenu("Project"))
	{
		const auto& map = assetSystem->getAssetMap(EAssetType::PMX);
		for (const auto& id : map)
		{
			const auto& asset = assetSystem->getAsset(id);
			if (ImGui::MenuItem((std::string("  ") + ICON_FA_CHESS_QUEEN"   " + asset->getRelativePathUtf8()).c_str()))
			{
				comp->setPMX(id);
			}
		}
		ImGui::EndMenu();
	}

}

void ComponentDrawer::drawPMX(std::shared_ptr<engine::SceneNode> node)
{
	std::shared_ptr<PMXComponent> comp = node->getComponent<PMXComponent>();

	ProjectContentWidget* contentWidget = Editor::get()->getContentWidget();
	auto* assetSystem = Editor::get()->getAssetSystem();

	ImGui::Unindent();
	ui::beginGroupPanel("Basic");
	{
		static const std::string selectButtonName = " PMX Chose ";
		if (ImGui::Button(selectButtonName.c_str()))
			ImGui::OpenPopup("PMXSelectPopUp");
		if (ImGui::BeginPopup("PMXSelectPopUp"))
		{
			ImGui::TextDisabled("Select PMX...");
			ImGui::Spacing();

			drawPMXSelect(node, comp);
			ImGui::EndPopup();
		}
	}
	ui::endGroupPanel();
	ImGui::Indent();
}