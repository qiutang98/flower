#include "editor_asset.h"
#include "editor.h"

#include <imgui/ui.h>

using namespace engine;

EditorAsset* EditorAsset::get()
{
	static EditorAsset assets;
	return &assets;
}

bool EditorAsset::registerAssetType(
	std::string name, 
	std::string iconName, 
	std::string decorateName, 
	const char* rawResourceExtensions, 
	engine::EAssetType type)
{
	if (m_registeredAssets.contains(name))
	{
		LOG_WARN("{0} already registered, fix me.", name);
		return false;
	}

	m_registeredAssets[name] = AssetInfo
	{
		.type = type,
		.name = name,
		.decoratorName = decorateName,
		.iconName = iconName,
		.rawResourceExtensions = rawResourceExtensions,
	};

	if (type != EAssetType::Max)
	{
		m_typeNameRelativeMap[type] = name;
	}

	return true;
}

struct RegisteredStructForEditorAsset
{
	explicit RegisteredStructForEditorAsset(
		std::string name,
		std::string iconName,
		std::string decorateName,
		const char* rawResourceExtensions,
		EAssetType type = EAssetType::Max)
	{
		CHECK(EditorAsset::get()->registerAssetType(name, iconName, decorateName, rawResourceExtensions, type));
	}
};

RegisteredStructForEditorAsset registerGLTF(
	"StaticMesh",
	ICON_FA_BUILDING,
	std::string("  ") + ICON_FA_BUILDING + std::string("     StaticMesh"),
	"obj",
	EAssetType::StaticMesh
);

RegisteredStructForEditorAsset registerTexture2D(
	"Texture",
	ICON_FA_IMAGE,
	std::string("  ") + ICON_FA_IMAGE + std::string("    Texture"),
	"jpg,jpeg,png,tga,exr;jpg,jpeg;png;tga;exr",
	EAssetType::Texture
);

RegisteredStructForEditorAsset registerScene(
	"Scene",
	ICON_FA_CHESS_KING,
	std::string("  ") + ICON_FA_CHESS_KING + std::string("     Scene"),
	"scene",
	EAssetType::Scene
);

RegisteredStructForEditorAsset registerPMX(
	"PMX",
	ICON_FA_CHESS_QUEEN,
	std::string("  ") + ICON_FA_CHESS_QUEEN + std::string("    PMX"),
	"pmx",
	EAssetType::PMX
);