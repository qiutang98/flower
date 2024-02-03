#include "asset_material.h"
#include "../ui/ui.h"
#include <renderer/render_scene.h>
#include "asset/asset_manager.h"
#include "asset_texture.h"

namespace engine
{
	AssetMaterial::AssetMaterial(const AssetSaveInfo& saveInfo)
		: AssetInterface(saveInfo)
	{

	}

	void AssetMaterial::onPostAssetConstruct()
	{

	}

	VulkanImage* AssetMaterial::getSnapshotImage()
	{
		static auto* icon = &getContext()->getBuiltinTexture(EBuiltinTextures::materialIcon)->getSelfImage();

		return icon;
	}

	const AssetReflectionInfo& AssetMaterial::uiGetAssetReflectionInfo()
	{
		const static AssetReflectionInfo kInfo =
		{
			.name = "Material",
			.icon = ICON_FA_BASEBALL,
			.decoratedName = std::string("  ") + ICON_FA_BASEBALL + std::string("    Material"),
			.importConfig = { .bImportable = false, }
		};
		return kInfo;
	}

	const AssetMaterial* AssetMaterial::getCDO()
	{
		static AssetMaterial material{ };
		return &material;
	}

	bool AssetMaterial::saveImpl()
	{
		std::shared_ptr<AssetInterface> asset = getptr<AssetMaterial>();
		return saveAsset(asset, getSavePath(), false);
	}

	void AssetMaterial::unloadImpl()
	{

	}

	BSDFMaterialInfo buildDefaultBSDFMaterialInfo()
	{
		ZoneScoped;
		uint32_t linearId;
		VkSamplerCreateInfo info
		{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.magFilter = VK_FILTER_LINEAR,
			.minFilter = VK_FILTER_LINEAR,
			.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
			.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.mipLodBias = 0.0f,
			.anisotropyEnable = VK_FALSE,
			.compareEnable = VK_FALSE,
			.minLod = 0.0f,
			.maxLod = 10000.0f,
			.unnormalizedCoordinates = VK_FALSE,
		};

		getContext()->getSamplerCache().createSampler(info, linearId);

		BSDFMaterialInfo result{};
		{
			result.baseColorId = getContext()->getBuiltinTextureWhite()->getBindlessIndex();
			result.normalTexId = getContext()->getBuiltinTextureNormal()->getBindlessIndex();
			result.metalRoughnessTexId = getContext()->getBuiltinTextureMetalRoughness()->getBindlessIndex();
			result.occlusionTexId = getContext()->getBuiltinTextureWhite()->getBindlessIndex();
			result.emissiveTexId = getContext()->getBuiltinTextureTranslucent()->getBindlessIndex();

			result.baseColorSampler = linearId;
			result.normalSampler = linearId;
			result.metalRoughnessSampler = linearId;
			result.occlusionSampler = linearId;
			result.emissiveSampler = linearId;
		};

		result.cutoff = 0.5f;
		result.baseColorMul = math::vec4{ 1.0f };
		result.baseColorAdd = math::vec4{ 0.0f };
		result.metalMul = 1.0f;
		result.metalAdd = 0.0f;
		result.roughnessMul = 1.0f;
		result.roughnessAdd = 0.0f;
		result.emissiveMul = math::vec4{ 1.0f };
		result.emissiveAdd = math::vec4{ 0.0f };
		result.shadingModel = EShadingModelType_DefaultLit;

		return result;
	}

	const BSDFMaterialInfo& AssetMaterial::getAndTryBuildGPU()
	{
		buildCache();
		return m_cacheBSDFMaterialInfo;
	}

	const BSDFMaterialInfo& AssetMaterial::getGPUOnly() const
	{
		return m_cacheBSDFMaterialInfo;
	}

	BSDFMaterialTextureHandle AssetMaterial::buildCache()
	{
		std::lock_guard<std::mutex> lock(m_cacheMaterialLock);

		BSDFMaterialTextureHandle outHandle;
		bool bAllTextureReady = true;

		auto getTexID = [&](const UUID& uuid, uint32_t& outId, std::shared_ptr<GPUImageAsset>& handle)
		{
			auto asset = std::static_pointer_cast<AssetTexture>(getAssetManager()->getAsset(uuid));
			handle = asset->getGPUImage().lock();

			bAllTextureReady &= handle->isAssetReady();
			outId = handle->getReadyAsset<GPUImageAsset>()->getBindlessIndex();
		};

		// Get all texture id.
		getTexID(baseColorTexture, m_cacheBSDFMaterialInfo.baseColorId, outHandle.baseColor);
		getTexID(normalTexture, m_cacheBSDFMaterialInfo.normalTexId, outHandle.normal);
		getTexID(metalRoughnessTexture, m_cacheBSDFMaterialInfo.metalRoughnessTexId, outHandle.metalRoughness);
		getTexID(aoTexture, m_cacheBSDFMaterialInfo.occlusionTexId, outHandle.aoTexture);
		getTexID(emissiveTexture, m_cacheBSDFMaterialInfo.emissiveTexId, outHandle.emissive);

		// Other parameters.
		m_cacheBSDFMaterialInfo.baseColorMul = this->baseColorMul;
		m_cacheBSDFMaterialInfo.baseColorAdd = this->baseColorAdd;
		m_cacheBSDFMaterialInfo.metalMul     = this->metalMul;
		m_cacheBSDFMaterialInfo.metalAdd     = this->metalAdd;
		m_cacheBSDFMaterialInfo.roughnessMul = this->roughnessMul;
		m_cacheBSDFMaterialInfo.roughnessAdd = this->roughnessAdd;
		m_cacheBSDFMaterialInfo.emissiveMul  = this->emissiveMul;
		m_cacheBSDFMaterialInfo.emissiveAdd  = this->emissiveAdd;
		m_cacheBSDFMaterialInfo.cutoff       = this->cutoff;
		m_cacheBSDFMaterialInfo.shadingModel = this->shadingModelType;

		if (bAllTextureReady)
		{
			m_bCacheValid = true;
		}

		return outHandle;
	}
}