#pragma once

#include "asset_common.h"
#include "asset_archive.h"
#include <util/shader_struct.h>

namespace engine
{
	class AssetMaterial : public AssetInterface
	{
	public:
		enum class EMaterialType
		{
			StandardPBR,
		};

		AssetMaterial() = default;
		AssetMaterial(const std::string& assetNameUtf8, const std::string& assetRelativeRootProjectPathUtf8, EMaterialType matType)
			: AssetInterface(assetNameUtf8, assetRelativeRootProjectPathUtf8)
			, m_materialType(matType)
		{

		}

		virtual EAssetType getType() const override { return EAssetType::Material; }
		virtual const char* getSuffix() const { return ".material"; }
		EMaterialType getMaterialType() const { return m_materialType; }

	protected:
		// Serialize field.
		ARCHIVE_DECLARE;

		EMaterialType m_materialType;
	};

	class StandardPBRMaterial : public AssetMaterial
	{
	public:
		StandardPBRMaterial();
		StandardPBRMaterial(const std::string& assetNameUtf8, const std::string& assetRelativeRootProjectPathUtf8)
			: AssetMaterial(assetNameUtf8, assetRelativeRootProjectPathUtf8, EMaterialType::StandardPBR)
		{
			m_runtimeMaterialcache = GPUMaterialStandardPBR::getDefault();
		}

	public:
		// Textures handle, keep in component which need textures.
		struct GPUTexturesHandle
		{
			// Material keep image shared reference avoid release.
			std::shared_ptr<GPUImageAsset> baseColor = nullptr;
			std::shared_ptr<GPUImageAsset> normal = nullptr;
			std::shared_ptr<GPUImageAsset> specular = nullptr;
			std::shared_ptr<GPUImageAsset> occlusion = nullptr;
			std::shared_ptr<GPUImageAsset> emissive = nullptr;
		};

		const GPUMaterialStandardPBR& getAndTryBuildGPU();
		const GPUMaterialStandardPBR& getGPUOnly() const;

		GPUTexturesHandle buildCache();

	private:
		// Runtime state, use this to know all asset in this material is ready or not.
		bool m_bAllAssetReady = false;
		GPUMaterialStandardPBR m_runtimeMaterialcache;

	public:
		ARCHIVE_DECLARE;

		UUID baseColorTexture = VulkanContext::getBuiltEngineAssetUUID(EBuiltinEngineAsset::Texture_White);
		UUID normalTexture    = VulkanContext::getBuiltEngineAssetUUID(EBuiltinEngineAsset::Texture_Normal);
		UUID specularTexture  = VulkanContext::getBuiltEngineAssetUUID(EBuiltinEngineAsset::Texture_Specular);
		UUID emissiveTexture  = VulkanContext::getBuiltEngineAssetUUID(EBuiltinEngineAsset::Texture_Translucent);
		UUID aoTexture        = VulkanContext::getBuiltEngineAssetUUID(EBuiltinEngineAsset::Texture_White);

		glm::vec4 baseColorMul = glm::vec4{ 1.0f };
		glm::vec4 baseColorAdd = glm::vec4{ 0.0f };

		float metalMul = 1.0f;
		float metalAdd = 0.0f;

		float roughnessMul = 1.0f;
		float roughnessAdd = 0.0f;

		glm::vec4 emissiveMul = glm::vec4{ 1.0f };
		glm::vec4 emissiveAdd = glm::vec4{ 0.0f };

		float cutoff  = 0.5f;
		float faceCut = 0.0f;
	};

	using StandardPBRMaterialHandle = StandardPBRMaterial::GPUTexturesHandle;
}

ASSET_ARCHIVE_IMPL_INHERIT(AssetMaterial, AssetInterface)
{
	size_t matType = size_t(m_materialType);
	ARCHIVE_NVP_DEFAULT(matType);
	m_materialType = EMaterialType(matType);
}
ASSET_ARCHIVE_END

ASSET_ARCHIVE_IMPL_INHERIT(StandardPBRMaterial, AssetMaterial)
{
	ARCHIVE_NVP_DEFAULT(baseColorTexture);
	ARCHIVE_NVP_DEFAULT(normalTexture);
	ARCHIVE_NVP_DEFAULT(specularTexture);
	ARCHIVE_NVP_DEFAULT(emissiveTexture);
	ARCHIVE_NVP_DEFAULT(aoTexture);

	ARCHIVE_NVP_DEFAULT(baseColorMul);
	ARCHIVE_NVP_DEFAULT(baseColorAdd);

	ARCHIVE_NVP_DEFAULT(metalMul);
	ARCHIVE_NVP_DEFAULT(metalAdd);

	ARCHIVE_NVP_DEFAULT(roughnessMul);
	ARCHIVE_NVP_DEFAULT(roughnessAdd);

	ARCHIVE_NVP_DEFAULT(emissiveMul);
	ARCHIVE_NVP_DEFAULT(emissiveAdd);

	ARCHIVE_NVP_DEFAULT(cutoff);
	ARCHIVE_NVP_DEFAULT(faceCut);
}
ASSET_ARCHIVE_END