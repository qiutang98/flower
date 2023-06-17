#pragma once

#define kAssetVersion 1

// Archive macro for convince.

// Version and type registry.
#define ASSET_ARCHIVE_IMPL_BASIC(AssetNameXX, Version) \
	CEREAL_CLASS_VERSION(engine::AssetNameXX, Version); \
	CEREAL_REGISTER_TYPE_WITH_NAME(engine::AssetNameXX, "engine::"#AssetNameXX);

// Virtual children class.
#define ASSET_ARCHIVE_IMPL_INHERIT(AssetNameXX, AssetNamePP) \
	ASSET_ARCHIVE_IMPL_BASIC(AssetNameXX, kAssetVersion); \
	CEREAL_REGISTER_POLYMORPHIC_RELATION(engine::AssetNamePP, engine::AssetNameXX)\
	template<class Archive> \
	void engine::AssetNameXX::serialize(Archive& archive, std::uint32_t const version) { \
	archive(cereal::base_class<engine::AssetNamePP>(this));

// Non virtual class.
#define ASSET_ARCHIVE_IMPL(AssetNameXX) \
	ASSET_ARCHIVE_IMPL_BASIC(AssetNameXX, kAssetVersion); \
	template<class Archive> \
	void engine::AssetNameXX::serialize(Archive& archive, std::uint32_t const version) {


#define ASSET_ARCHIVE_END }


#define ARCHIVE_ENUM_CLASS(value)   \
	size_t enum__type__##value = (size_t)value; \
	ARCHIVE_NVP_DEFAULT(enum__type__##value); \
	value = (decltype(value))(enum__type__##value);
	