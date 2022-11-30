#include "Pch.h"
#include "../Detail.h"
#include "DrawComponent.h"

using namespace Flower;
using namespace Flower::UI;

void drawStaticMeshSelect(std::shared_ptr<SceneNode> node, std::shared_ptr<StaticMeshComponent> comp)
{
	if (ImGui::BeginMenu("Engine"))
	{
		auto meshItemShow = [&](const Flower::UUID& info, const std::string& showName)
		{
			const char* name = showName.c_str();
			if (ImGui::MenuItem(name))
			{
				comp->setMeshUUID(info);
			}
		};

		meshItemShow(EngineMeshes::GBoxUUID, ICON_FA_CHESS"   Box");
		meshItemShow(EngineMeshes::GSphereUUID, ICON_FA_CHESS"   Sphere");
		ImGui::EndMenu();
	}

	if (ImGui::BeginMenu("Project"))
	{
		const auto& map = AssetRegistryManager::get()->getTypeAssetSetMap();
		if (map.contains(size_t(EAssetType::StaticMesh)))
		{
			const auto& meshMap = map.at(size_t(EAssetType::StaticMesh));
			for (const auto& meshId : meshMap)
			{
				if (ImGui::MenuItem(AssetRegistryManager::get()->getHeaderMap().at(meshId)->getName().c_str()))
				{
					comp->setMeshUUID(meshId);
				}
			}

		}
		

		ImGui::EndMenu();
	}
}

void ComponentDrawer::drawStaticMesh(std::shared_ptr<SceneNode> node)
{
	std::shared_ptr<StaticMeshComponent> comp = node->getComponent<StaticMeshComponent>();

	if (comp->isMeshAlreadySet())
	{
		ImGui::TextDisabled("Mesh asset: %s aleady set for this component.", comp->getMeshAssetName().c_str());
		ImGui::TextDisabled("Asset uuid: %s.", comp->getUUID().c_str());
	}
	else
	{
		ImGui::TextDisabled("Non-mesh set on the mesh component.");
		ImGui::TextDisabled("Please select one mesh asset for this component.");
	}

	static const std::string selectButtonName = GIconStaticMesh + "  Select...";
	if (ImGui::Button(selectButtonName.c_str()))
		ImGui::OpenPopup("StaticMeshSelectPopUp");
	if (ImGui::BeginPopup("StaticMeshSelectPopUp"))
	{
		drawStaticMeshSelect(node, comp);
		ImGui::EndPopup();
	}

	ImGui::TextDisabled("Submesh  count: %d.", comp->isMeshAlreadySet() ? comp->getSubmeshCount() : 0);
	ImGui::TextDisabled("Vertices count: %d.", comp->isMeshAlreadySet() ? comp->getVerticesCount() : 0);
	ImGui::TextDisabled("Indices  count: %d.", comp->isMeshAlreadySet() ? comp->getIndicesCount() : 0);
}