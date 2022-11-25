#pragma once
#include "AssetCommon.h"
#include "LRUCache.h"
#include "AsyncUploader.h"
#include "TextureManager.h"

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
	public:
		UUID baseColorTexture = EngineTextures::GWhiteTextureUUID;
		UUID normalTexture = EngineTextures::GNormalTextureUUID;
		UUID specularTexture = EngineTextures::GDefaultSpecularUUID;
		UUID emissiveTexture = EngineTextures::GTranslucentTextureUUID;
		UUID aoTexture = EngineTextures::GWhiteTextureUUID;

		StandardPBRMaterialHeader() { }
		StandardPBRMaterialHeader(const std::string& name)
			: AssetMaterialHeader(name, EMaterialType::StandardPBR)
		{

		}
	};
}