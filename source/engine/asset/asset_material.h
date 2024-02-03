#pragma once

#include "asset.h"
#include "asset_common.h"
#include <common_header.h>
#include "../graphics/context.h"

namespace engine
{
	extern BSDFMaterialInfo buildDefaultBSDFMaterialInfo();

	struct BSDFMaterialTextureHandle
	{
		std::shared_ptr<GPUImageAsset> baseColor;
		std::shared_ptr<GPUImageAsset> normal;
		std::shared_ptr<GPUImageAsset> metalRoughness;
		std::shared_ptr<GPUImageAsset> emissive;
		std::shared_ptr<GPUImageAsset> aoTexture;
	};

	class AssetMaterial : public AssetInterface
	{
		REGISTER_BODY_DECLARE(AssetInterface);

	public:
		AssetMaterial() = default;
		virtual ~AssetMaterial() = default;

		explicit AssetMaterial(const AssetSaveInfo& saveInfo);

		// ~AssetInterface virtual function.
		virtual EAssetType getType() const override
		{
			return EAssetType::darkmaterial;
		}
		virtual void onPostAssetConstruct() override;
		virtual VulkanImage* getSnapshotImage() override;
		// ~AssetInterface virtual function.

		static const AssetReflectionInfo& uiGetAssetReflectionInfo();
		const static AssetMaterial* getCDO();

	protected:
		// ~AssetInterface virtual function.
		virtual bool saveImpl() override;

		virtual void unloadImpl() override;
		// ~AssetInterface virtual function.

	public:
		const BSDFMaterialInfo& getAndTryBuildGPU();
		const BSDFMaterialInfo& getGPUOnly() const;

		BSDFMaterialTextureHandle buildCache();

	private:
		bool m_bCacheValid = false;

		// Cache state.
		std::mutex m_cacheMaterialLock;
		BSDFMaterialInfo m_cacheBSDFMaterialInfo = buildDefaultBSDFMaterialInfo();

	public:
		UUID baseColorTexture       = getBuiltinTexturesUUID(EBuiltinTextures::white);
		UUID normalTexture          = getBuiltinTexturesUUID(EBuiltinTextures::normal);
		UUID metalRoughnessTexture  = getBuiltinTexturesUUID(EBuiltinTextures::metalRoughness);
		UUID emissiveTexture        = getBuiltinTexturesUUID(EBuiltinTextures::translucent);
		UUID aoTexture              = getBuiltinTexturesUUID(EBuiltinTextures::white);

		math::vec4 baseColorMul = math::vec4{ 1.0f };
		math::vec4 baseColorAdd = math::vec4{ 0.0f };

		float metalMul = 1.0f;
		float metalAdd = 0.0f;
		float roughnessMul = 1.0f;
		float roughnessAdd = 0.0f;

		math::vec4 emissiveMul = math::vec4{ 1.0f };
		math::vec4 emissiveAdd = math::vec4{ 0.0f };

		float cutoff = 0.5f;
		EShadingModelType shadingModelType = EShadingModelType_DefaultLit;
	};
}

