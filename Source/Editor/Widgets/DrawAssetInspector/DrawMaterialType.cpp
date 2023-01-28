#include "Pch.h"
#include "DrawAssetInspector.h"

using namespace Flower;
using namespace Flower::UI;

static inline bool drawStandardMaterial(std::shared_ptr<StandardPBRMaterialHeader> asset)
{
	bool bDataChange = false;

	bDataChange |= UIHelper::drawVector4("BaseColor Mul", asset->baseColorMul, glm::vec4{ 1.0f }, ImGui::GetFontSize() * 6.0f);
	bDataChange |= UIHelper::drawVector4("BaseColor Add", asset->baseColorAdd, glm::vec4{ 0.0f }, ImGui::GetFontSize() * 6.0f);

	bDataChange |= UIHelper::drawVector4("Emissive Mul", asset->emissiveMul, glm::vec4{ 1.0f }, ImGui::GetFontSize() * 6.0f);
	bDataChange |= UIHelper::drawVector4("Emissive Add", asset->emissiveAdd, glm::vec4{ 0.0f }, ImGui::GetFontSize() * 6.0f);

	bDataChange |= UIHelper::drawFloat("Metallic Mul", asset->metalMul, 1.0f);
	bDataChange |= UIHelper::drawFloat("Metallic Add", asset->metalAdd, 0.0f);
	bDataChange |= UIHelper::drawFloat("Roughness Mul", asset->roughnessMul, 1.0f);
	bDataChange |= UIHelper::drawFloat("Roughness Add", asset->roughnessAdd, 0.0f);

	if (bDataChange)
	{
		asset->buildCache();
	}

	return bDataChange;
}

void AssetInspectorDrawer::drawMaterialAsset(std::shared_ptr<AssetMaterialHeader> asset)
{
	const auto& assetType = asset->getMaterialType();
	bool bDataChange = false;
	
	if (assetType == EMaterialType::StandardPBR)
	{
		if (auto standardPBRMat = std::dynamic_pointer_cast<StandardPBRMaterialHeader>(asset))
		{
			bDataChange = drawStandardMaterial(standardPBRMat);
		}
	}

	if (bDataChange)
	{
		AssetRegistryManager::get()->markDirty();
	}
}