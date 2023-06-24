#pragma once

#include "asset_common.h"
#include "asset_archive.h"
#include "asset_material.h"

#include <Saba/Model/MMD/PMXModel.h>
#include <Saba/Model/MMD/VMDFile.h>
#include <Saba/Model/MMD/VMDAnimation.h>
#include <Saba/Model/MMD/VMDCameraAnimation.h>

namespace engine
{
	class AssetVMD : public AssetInterface
	{
	public:
		struct ImportConfig
		{
			bool bCamera;
		};

		AssetVMD() = default;
		AssetVMD(const std::string& assetNameUtf8, const std::string& assetRelativeRootProjectPathUtf8);

		virtual EAssetType getType() const override { return EAssetType::VMD; }
		virtual const char* getSuffix() const { return ".assetvmd"; }

		static bool buildFromConfigs(
			const ImportConfig& config,
			const std::filesystem::path& projectRootPath,
			const std::filesystem::path& savePath,
			const std::filesystem::path& srcPath
		);

		std::filesystem::path getVMDFilePath() const;

	protected:


	public:
		ARCHIVE_DECLARE;

		bool m_bCamera;
	};
}

ASSET_ARCHIVE_IMPL_INHERIT(AssetVMD, AssetInterface)
{
	ARCHIVE_NVP_DEFAULT(m_bCamera);
}
ASSET_ARCHIVE_END