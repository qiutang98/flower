#include "component_draw.h"
#include <imgui/ui.h>
#include <scene/scene.h>
#include "../editor/editor.h"
#include "../editor/widgets/project_content.h"
#include <scene/component/static_mesh.h>

using namespace engine;
using namespace engine::ui;

void drawStaticMeshSelect(std::shared_ptr<SceneNode> node, std::shared_ptr<StaticMeshComponent> comp)
{
	auto* assetSystem = Editor::get()->getAssetSystem();
	auto* context = Editor::get()->getContext();

	if (ImGui::BeginMenu("Engine"))
	{
		auto meshItemShow = [&](const UUID& info, const std::string& showName)
		{
			const char* name = showName.c_str();
			if (ImGui::MenuItem(name))
			{
				comp->setMesh(info, showName, true);
			}
		};

		meshItemShow(context->getBuiltEngineAssetUUID(EBuiltinEngineAsset::StaticMesh_Box), std::string("  ") + ICON_FA_CHESS + "   /Engine/Box");
		meshItemShow(context->getBuiltEngineAssetUUID(EBuiltinEngineAsset::StaticMesh_Sphere), std::string("  ") + ICON_FA_CHESS + "   /Engine/Sphere");
		ImGui::EndMenu();
	}

	ImGui::Spacing();
	if (ImGui::BeginMenu("Project"))
	{
		const auto& map = assetSystem->getAssetMap(EAssetType::StaticMesh);
		for (const auto& meshId : map)
		{
			const auto& asset = assetSystem->getAsset(meshId);
			if (ImGui::MenuItem((std::string("  ") + ICON_FA_CHESS_PAWN"   " + asset->getRelativePathUtf8()).c_str()))
			{
				comp->setMesh(meshId, asset->getRelativePathUtf8(), false);
			}
		}
		ImGui::EndMenu();
	}
}


void ComponentDrawer::drawStaticMesh(std::shared_ptr<SceneNode> node)
{
	std::shared_ptr<StaticMeshComponent> comp = node->getComponent<StaticMeshComponent>();

	ProjectContentWidget* contentWidget = Editor::get()->getContentWidget();
	auto* assetSystem = Editor::get()->getAssetSystem();

	ImGui::Unindent();
	ui::beginGroupPanel("Basic");

	if(comp->getMeshUUID().empty())
	{
		ImGui::TextDisabled("Non-mesh set on the mesh component.");
		ImGui::TextDisabled("Please select one mesh asset for this component.");
	}
	else
	{
		ImGui::TextDisabled("Mesh asset: %s.", comp->getMeshAssetRelativeRoot().c_str());
		ImGui::TextDisabled("Asset uuid: %s.", comp->getMeshUUID().c_str());
	}

	ImGui::Spacing();

	VkDescriptorSet set = Editor::get()->getClampToTransparentBorderSet(&Editor::get()->getContext()->getEngineTextureTranslucent()->getImage());
	
	const float kItemDim = ImGui::GetTextLineHeightWithSpacing() * 5.0f;

	ImGui::Image(set, { kItemDim , kItemDim });
	ImGui::GetWindowDrawList()->AddRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), IM_COL32(255, 255, 255, 80));

    if (ImGui::BeginDragDropTarget())
    {
        if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload(ProjectContentWidget::kAssetDragDropName.c_str()))
        {
			const auto& dragAssets = contentWidget->getDragDropAssets();
			if (dragAssets.selectAssets.size() == 1)
			{
				const std::filesystem::path& assetPath = *dragAssets.selectAssets.begin();
				if (isAssetStaticMeshMeta(assetPath.extension().string()))
				{
					// Is static mesh meta asset, can assign.
					auto copyPath = assetPath;
					copyPath.replace_extension();

					const auto relativeAssetPath = buildRelativePathUtf8(Editor::get()->getProjectRootPathUtf16(), copyPath);
					comp->setMesh(assetSystem->getAssetByRelativeMap(relativeAssetPath)->getUUID(), relativeAssetPath, false);
				}
			}
        }
        ImGui::EndDragDropTarget();
    }

	ImGui::SameLine();
	ImGui::BeginGroup();
	static const std::string selectButtonName = kIconStaticMesh + " Chose ";
	if (ImGui::Button(selectButtonName.c_str()))
		ImGui::OpenPopup("StaticMeshSelectPopUp");
	if (ImGui::BeginPopup("StaticMeshSelectPopUp"))
	{
		ImGui::TextDisabled("Select StaticMesh...");
		ImGui::Spacing();

		drawStaticMeshSelect(node, comp);
		ImGui::EndPopup();
	}


	ImGui::TextDisabled("Submesh  count: %d.", comp->isGPUMeshAssetExist() ? comp->getSubmeshCount() : 0);
	ImGui::TextDisabled("Vertices count: %d.", comp->isGPUMeshAssetExist() ? comp->getVerticesCount() : 0);
	ImGui::TextDisabled("Indices  count: %d.", comp->isGPUMeshAssetExist() ? comp->getIndicesCount() : 0);

	ImGui::EndGroup();

	ui::endGroupPanel();
	ImGui::Indent();
}