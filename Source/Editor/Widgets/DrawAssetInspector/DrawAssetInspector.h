#pragma once
#include "Pch.h"

struct AssetInspectorDrawer
{
	static void drawTextureAsset(std::shared_ptr<Flower::ImageAssetHeader> asset);
	static void drawStaticMeshAsset(std::shared_ptr<Flower::StaticMeshAssetHeader> asset);
	static void drawMaterialAsset(std::shared_ptr<Flower::AssetMaterialHeader> asset);
};