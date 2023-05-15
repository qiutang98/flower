#pragma once

#include "asset_common.h"
#include "asset_archive.h"
#include "asset_material.h"

namespace engine
{
	class PMXMaterial : public AssetMaterial
	{
	public:
		PMXMaterial();
		PMXMaterial(const std::string& assetNameUtf8, const std::string& assetRelativeRootProjectPathUtf8)
			: AssetMaterial(assetNameUtf8, assetRelativeRootProjectPathUtf8, EMaterialType::PMX)
		{

		}


	private:


	public:


	};

	// WARN: PMX mesh produced by MMD artist may **Not-Under** MIT license.
	//       So we don't change the resource of pmx raw file.
	//       We just add one additional meta data for it.
	class AssetPMX : public AssetInterface
	{
	public:
		struct ImportConfig
		{

		};

		AssetPMX() = default;
		AssetPMX(const std::string& assetNameUtf8, const std::string& assetRelativeRootProjectPathUtf8);


		virtual EAssetType getType() const override { return EAssetType::PMX; }
		virtual const char* getSuffix() const { return ".assetpmx"; }

	public:
		ARCHIVE_DECLARE;

		std::string m_pmxFilePath;

	};
}

ASSET_ARCHIVE_IMPL_INHERIT(AssetPMX, AssetInterface)
{

}
ASSET_ARCHIVE_END