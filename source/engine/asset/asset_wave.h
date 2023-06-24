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
	class AssetWave : public AssetInterface
	{
	public:
		struct ImportConfig
		{
			bool bVolumetric;
		};

		AssetWave() = default;
		AssetWave(const std::string& assetNameUtf8, const std::string& assetRelativeRootProjectPathUtf8);

		virtual EAssetType getType() const override { return EAssetType::Wave; }
		virtual const char* getSuffix() const { return ".assetwave"; }

		static bool buildFromConfigs(
			const ImportConfig& config,
			const std::filesystem::path& projectRootPath,
			const std::filesystem::path& savePath,
			const std::filesystem::path& srcPath
		);

		std::filesystem::path getWaveFilePath() const;

	protected:


	public:
		ARCHIVE_DECLARE;

		bool m_bVolumetric;
	};
}

ASSET_ARCHIVE_IMPL_INHERIT(AssetWave, AssetInterface)
{
	ARCHIVE_NVP_DEFAULT(m_bVolumetric);
}
ASSET_ARCHIVE_END