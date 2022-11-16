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
	private:
		EMaterialType m_materialType;

		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive)
		{
			archive(cereal::base_class<AssetHeaderInterface>(this));

			size_t matType = size_t(m_materialType);
			archive(matType);
			m_materialType = EMaterialType(matType);
		}

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
	public:
		UUID baseColorTexture = EngineTextures::GWhiteTextureUUID;
		UUID normalTexture = EngineTextures::GNormalTextureUUID;
		UUID specularTexture = EngineTextures::GDefaultSpecularUUID;
		UUID emissiveTexture = EngineTextures::GTranslucentTextureUUID;
		UUID aoTexture = EngineTextures::GWhiteTextureUUID;


		template<class Archive>
		void serialize(Archive& archive)
		{
			archive(cereal::base_class<AssetMaterialHeader>(this));
			archive(baseColorTexture, normalTexture, specularTexture, emissiveTexture, aoTexture);
		}

		StandardPBRMaterialHeader() { }
		StandardPBRMaterialHeader(const std::string& name)
			: AssetMaterialHeader(name, EMaterialType::StandardPBR)
		{

		}
	};
}


CEREAL_REGISTER_TYPE(Flower::AssetMaterialHeader);
CEREAL_REGISTER_POLYMORPHIC_RELATION(Flower::AssetHeaderInterface, Flower::AssetMaterialHeader)

CEREAL_REGISTER_TYPE(Flower::StandardPBRMaterialHeader);
CEREAL_REGISTER_POLYMORPHIC_RELATION(Flower::AssetMaterialHeader, Flower::StandardPBRMaterialHeader)