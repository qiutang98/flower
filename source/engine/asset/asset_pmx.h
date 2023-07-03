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
	class PMXDrawMaterial
	{
	public:
		// Runtime build info.
		uint32_t mmdTex       = ~0;
		uint32_t mmdSphereTex = ~0;
		uint32_t mmdToonTex   = ~0;

	public:
		saba::MMDMaterial material;

		bool     bTranslucent     = false;
		bool     bHide            = false;
		float    pixelDepthOffset = 0.0f;
		bool     bCastShadow      = true; 
		float    eyeHighlightScale = 1.0f;
		float    translucentUnlitScale = 0.1f;
		EShadingModelType pmxShadingModel = EShadingModelType::StandardPBR;

		auto operator<=>(const PMXDrawMaterial&) const = default;
		template<class Archive> void serialize(Archive& archive, std::uint32_t const version)
		{
			archive(material);
			archive(bTranslucent, bHide, pixelDepthOffset);

			ARCHIVE_ENUM_CLASS(pmxShadingModel);

			if (version > 1)
			{
				archive(bCastShadow);
				if (version > 2)
				{
					archive(eyeHighlightScale, translucentUnlitScale);
				}
			}
		}
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

		static bool buildFromConfigs(
			const ImportConfig& config,
			const std::filesystem::path& projectRootPath,
			const std::filesystem::path& savePath,
			const std::filesystem::path& srcPath
		);

		std::filesystem::path getPMXFilePath() const;
		std::filesystem::path getPMXFolderPath() const { return getPMXFilePath().parent_path(); }

		void tryLoadAllTextures(const saba::PMXModel& pmx);

		const auto& getMaterials() const { return m_materials; }
		auto& getMaterials() { return m_materials; }

		virtual bool drawAssetConfig() override;

	protected:
		virtual bool saveActionImpl() override;

	public:
		ARCHIVE_DECLARE;

		std::vector<PMXDrawMaterial> m_materials;
	};
}

ASSET_ARCHIVE_IMPL_INHERIT(AssetPMX, AssetInterface)
{
	ARCHIVE_NVP_DEFAULT(m_materials);
}
ASSET_ARCHIVE_END

CEREAL_CLASS_VERSION(engine::PMXDrawMaterial, kAssetVersion);