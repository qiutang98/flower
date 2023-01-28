#pragma once
#include "../Version.h"

#include "AssetCommon.h"
#include "AssetRegistry.h"
#include "TextureManager.h"
#include "MeshManager.h"
#include "MaterialManager.h"


#define MAKE_VERSION_ASSET(ClassType) CEREAL_CLASS_VERSION(ClassType, ASSET_VERSION_CONTROL)

#define ASSET_ARCHIVE_IMPL_BASIC(AssetNameXX) \
MAKE_VERSION_ASSET(Flower::AssetNameXX); \
CEREAL_REGISTER_TYPE_WITH_NAME(Flower::AssetNameXX, "Flower::"#AssetNameXX);

#define ASSET_ARCHIVE_IMPL(AssetNameXX) \
ASSET_ARCHIVE_IMPL_BASIC(AssetNameXX); \
template<class Archive> \
void Flower::AssetNameXX::serialize(Archive& archive, std::uint32_t const version)

#define ASSET_ARCHIVE_IMPL_INHERIT(AssetNameXX, AssetNamePP) \
ASSET_ARCHIVE_IMPL_BASIC(AssetNameXX); \
CEREAL_REGISTER_POLYMORPHIC_RELATION(Flower::AssetNamePP, Flower::AssetNameXX)\
template<class Archive> \
void Flower::AssetNameXX::serialize(Archive& archive, std::uint32_t const version) { \
archive(cereal::base_class<Flower::AssetNamePP>(this));

#define ASSET_ARCHIVE_IMPL_INHERIT_END }


template<class Archive>
void Flower::RegistryEntry::serialize(Archive& archive, uint32_t version)
{
	archive(m_name);
	archive(m_parent);
	archive(m_children);
	archive(m_assetHeader);
	archive(m_uuid);
}
CEREAL_CLASS_VERSION(Flower::RegistryEntry, ASSET_VERSION_CONTROL)

ASSET_ARCHIVE_IMPL(AssetHeaderInterface)
{
	archive(m_assetName);
	archive(m_uuid);
	archive(m_binDataUUID);
}

ASSET_ARCHIVE_IMPL(AssetBinInterface)
{
	archive(m_uuid);
}

ASSET_ARCHIVE_IMPL_INHERIT(ImageAssetHeader, AssetHeaderInterface)
{
	archive(
		m_width, 
		m_widthSnapShot,
		m_height, 
		m_heightSnapShot,
		m_depth,
		m_format,
		m_mipmapCount,
		m_bSrgb,
		m_bHdr,
		m_snapshotData,
		m_snapshotUUID
	);
}
ASSET_ARCHIVE_IMPL_INHERIT_END

ASSET_ARCHIVE_IMPL_INHERIT(ImageAssetBin, AssetBinInterface)
{
	archive(m_rawData);
	archive(m_mipmapData);
}
ASSET_ARCHIVE_IMPL_INHERIT_END

ASSET_ARCHIVE_IMPL_INHERIT(StaticMeshAssetHeader, AssetHeaderInterface)
{
	archive(m_subMeshes);
	archive(m_indicesCount);
	archive(m_verticesCount);
}
ASSET_ARCHIVE_IMPL_INHERIT_END

ASSET_ARCHIVE_IMPL_INHERIT(StaticMeshAssetBin, AssetBinInterface)
{
	archive(m_indices);
	archive(m_vertices);
}
ASSET_ARCHIVE_IMPL_INHERIT_END


ASSET_ARCHIVE_IMPL_INHERIT(AssetMaterialHeader, AssetHeaderInterface)
{
	size_t matType = size_t(m_materialType);
	archive(matType);
	m_materialType = EMaterialType(matType);
}
ASSET_ARCHIVE_IMPL_INHERIT_END

ASSET_ARCHIVE_IMPL_INHERIT(StandardPBRMaterialHeader, AssetMaterialHeader)
{
	archive(
		baseColorTexture, 
		normalTexture, 
		specularTexture, 
		emissiveTexture, 
		aoTexture);

	if (version > 0)
	{
		archive(
			baseColorMul, baseColorAdd, 
			metalMul, metalAdd, 
			roughnessMul, roughnessAdd, 
			emissiveMul, emissiveAdd,
			cutoff, faceCut);
	}
}
ASSET_ARCHIVE_IMPL_INHERIT_END