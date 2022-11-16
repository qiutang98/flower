#pragma once
#include "Pch.h"
#include "Widgets/WidgetsHeader.h"

struct DragAndDropAssets
{
	std::unordered_set<Flower::UUID> selectAssets;
};

class EditorAsset
{
private:
	struct AssetInfo
	{
		Flower::EAssetType type;

		std::string name;
		std::string decoratorName;
		std::string iconName;
		const char* rawResourceExtensions;
	};

	std::map<std::string, AssetInfo> m_registeredAssets;
	std::map<Flower::EAssetType, std::string> m_typeNameRelativeMap;

	// Cache set for snapshot image.
	std::unordered_map<Flower::UUID, VkDescriptorSet> m_cacheSnapShotSet;

public:
	EditorAsset() = default;

	bool registerAssetType(
		std::string name,
		std::string iconName,
		std::string decorateName,
		const char* rawResourceExtensions,
		Flower::EAssetType type
	);

	const auto& getTypeNameMap() const 
	{ 
		return m_typeNameRelativeMap; 
	}

	const auto& getRegisterMap() const 
	{ 
		return m_registeredAssets; 
	}

	VkDescriptorSet getSetByAssetAsSnapShot(Flower::GPUImageAsset* imageAsset);

	const static std::string DragDropName;
};

using EditorAssetSystem = Flower::Singleton<EditorAsset>;