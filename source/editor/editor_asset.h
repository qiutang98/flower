#pragma once

#include <asset/asset.h>
#include <rhi/rhi.h>

class EditorAsset
{
public:
	static EditorAsset* get();

	bool registerAssetType(
		std::string name,
		std::string iconName,
		std::string decorateName,
		const char* rawResourceExtensions,
		engine::EAssetType type
	);

	const auto& getTypeNameMap() const { return m_typeNameRelativeMap; }
	const auto& getRegisterMap() const { return m_registeredAssets; }

private:
	EditorAsset() = default;

	struct AssetInfo
	{
		engine::EAssetType type;
		std::string name;
		std::string decoratorName;
		std::string iconName;
		const char* rawResourceExtensions;
	};

	std::map<std::string, AssetInfo> m_registeredAssets;
	std::map<engine::EAssetType, std::string> m_typeNameRelativeMap;
};