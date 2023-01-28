#pragma once
#include "AssetCommon.h"
#include "LRUCache.h"
#include "AsyncUploader.h"
#include "TextureManager.h"
#include "../Renderer/Parameters.h"

namespace Flower
{
	enum class EMaterialType
	{
		StandardPBR,
	};

	class AssetMaterialHeader : public AssetHeaderInterface
	{
		ARCHIVE_DECLARE;
	private:
		EMaterialType m_materialType;

	public:
		AssetMaterialHeader() { }
		AssetMaterialHeader(const std::string& name, EMaterialType matType)
			: AssetHeaderInterface(buildUUID(), name), m_materialType(matType)
		{

		}
		virtual EAssetType getType() const 
		{ 
			return EAssetType::Material; 
		}

		EMaterialType getMaterialType() const 
		{
			return m_materialType; 
		}
	};

	class StandardPBRMaterialHeader : public AssetMaterialHeader
	{
		ARCHIVE_DECLARE;

		GPUStaticMeshStandardPBRMaterial m_runtimeMaterialcache;
		bool m_bAllAssetReady = false;

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

		const GPUStaticMeshStandardPBRMaterial& getAndTryBuildGPU();
		const GPUStaticMeshStandardPBRMaterial& getGPUOnly() const;

		GPUTexturesHandle buildCache();

		UUID baseColorTexture = EngineTextures::GWhiteTextureUUID;
		UUID normalTexture    = EngineTextures::GNormalTextureUUID;
		UUID specularTexture  = EngineTextures::GDefaultSpecularUUID;
		UUID emissiveTexture  = EngineTextures::GTranslucentTextureUUID;
		UUID aoTexture        = EngineTextures::GWhiteTextureUUID;

		glm::vec4 baseColorMul = glm::vec4{ 1.0f };
		glm::vec4 baseColorAdd = glm::vec4{ 0.0f };

		float metalMul = 1.0f;
		float metalAdd = 0.0f;
		
		float roughnessMul = 1.0f;
		float roughnessAdd = 0.0f;

		glm::vec4 emissiveMul = glm::vec4{ 1.0f };
		glm::vec4 emissiveAdd = glm::vec4{ 0.0f };

		float cutoff = 0.5f;
		float faceCut = 0.0f;

		StandardPBRMaterialHeader() { }
		StandardPBRMaterialHeader(const std::string& name)
			: AssetMaterialHeader(name, EMaterialType::StandardPBR)
		{

		}
	};

	using StandardPBRTexturesHandle = StandardPBRMaterialHeader::GPUTexturesHandle;
}