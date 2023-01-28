#include "Pch.h"
#include "AssetInspector.h"
#include "../Editor.h"
#include "DrawAssetInspector/DrawAssetInspector.h"
#include "EditorAsset.h"

using namespace Flower;
using namespace Flower::UI;

static const std::string DETAIL_AssetInspectorlIcon = ICON_FA_ID_CARD;

WidgetAssetInspector::WidgetAssetInspector()
	: Widget("  " + DETAIL_AssetInspectorlIcon + "  AssetInspector")
{

}

WidgetAssetInspector::~WidgetAssetInspector() noexcept
{

}

void WidgetAssetInspector::onInit()
{

}

void WidgetAssetInspector::onRelease()
{

}

void WidgetAssetInspector::onTick(const RuntimeModuleTickData& tickData)
{

}

void WidgetAssetInspector::onVisibleTick(const RuntimeModuleTickData& tickData)
{
	ImGui::Spacing();

	ImGui::TextDisabled("Asset inspector for asset detail edit.");

	ImGui::Spacing();
	ImGui::Separator();

	auto* contentViewer = GEditor->getContentViewer();

	const auto& selectionAssets = contentViewer->getSelectionAssets();
	if (selectionAssets.size() == 0)
	{
		ImGui::TextDisabled("No asset selection yet.");
	}
	else if (selectionAssets.size() > 1)
	{
		ImGui::TextDisabled("Current no support multi asset edit yet.");
	}
	else
	{
		const Flower::UUID& assetUUID = *selectionAssets.begin();
		ImGui::Text("Asset %s inspecting.", assetUUID.c_str());

		ImGui::Spacing();
		if (ImGui::Button("Save"))
		{
			if (AssetRegistryManager::get()->isDirty())
			{
				AssetRegistryManager::get()->save();
			}
		}
		ImGui::Spacing();

		static auto* assetSystem = GEngine->getRuntimeModule<AssetSystem>();
		auto assetEntryPtr = AssetRegistryManager::get()->getEntryMap().at(assetUUID).lock();
		if (!assetEntryPtr->getAssetHeaderID().empty())
		{
			auto& assetHeader = AssetRegistryManager::get()->getHeaderMap().at(assetEntryPtr->getAssetHeaderID());
			const auto& assetType = assetHeader->getType();
			
			if (assetType == EAssetType::StaticMesh)
			{
				if (auto staticMeshHeader = std::dynamic_pointer_cast<StaticMeshAssetHeader>(assetHeader))
				{
					AssetInspectorDrawer::drawStaticMeshAsset(staticMeshHeader);
				}
			}
			else if (assetType == EAssetType::Texture)
			{
				if (auto textureHeader = std::dynamic_pointer_cast<ImageAssetHeader>(assetHeader))
				{
					AssetInspectorDrawer::drawTextureAsset(textureHeader);
				}
			}
			else if (assetType == EAssetType::Material)
			{
				if (auto materialHeader = std::dynamic_pointer_cast<AssetMaterialHeader>(assetHeader))
				{
					AssetInspectorDrawer::drawMaterialAsset(materialHeader);
				}
			}
		}
	}
}