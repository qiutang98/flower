#include "Pch.h"
#include "EditorAsset.h"

using namespace Flower;
using namespace Flower::UI;

const std::string EditorAsset::DragDropName = "_39AssetDragDropName";

bool EditorAsset::registerAssetType(
	std::string name,
	std::string iconName,
	std::string decorateName,
	const char* rawResourceExtensions,
	EAssetType type)
{
	if (m_registeredAssets.contains(name))
	{
		LOG_WARN("{0} already registered, fix me.", name);
		return false;
	}

	AssetInfo newInfo{};
	newInfo.type = type;
	newInfo.name = name;
	newInfo.decoratorName = decorateName;
	newInfo.rawResourceExtensions = rawResourceExtensions;
	newInfo.iconName = iconName;

	m_registeredAssets[name] = newInfo;

	if (type != EAssetType::Max)
	{
		m_typeNameRelativeMap[type] = name;
	}

	return true;
}

VkDescriptorSet EditorAsset::getSetByAssetAsSnapShot(GPUImageAsset* imageAsset)
{
	const std::string identify = imageAsset->getUUID();

	VkSamplerCreateInfo info = SamplerFactory::buildBasic();
	info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	info.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;

	if (!m_cacheSnapShotSet.contains(identify))
	{
		m_cacheSnapShotSet[identify] = ImGui_ImplVulkan_AddTexture(
			RHI::SamplerManager->createSampler(info),
			imageAsset->getImage().getView(buildBasicImageSubresource()),
			VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		);
	}

	return m_cacheSnapShotSet.at(identify);
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
		CHECK(EditorAssetSystem::get()->registerAssetType(name, iconName, decorateName, rawResourceExtensions, type));
	}
};

RegisteredStructForEditorAsset registerTexture(
	"Texture",
	ICON_FA_IMAGE,
	std::string("  ") + ICON_FA_IMAGE + std::string("    Texture"),
	"jpg,jpeg,png,tga,psd,hdr;jpg,jpeg;png;tga;psd;hdr",
	EAssetType::Texture
);

RegisteredStructForEditorAsset registerStaticMesh(
	"StaticMesh",
	ICON_FA_BUILDING,
	std::string("  ") + ICON_FA_BUILDING + std::string("    StaticMesh"),
	"obj;gltf",
	EAssetType::StaticMesh
);