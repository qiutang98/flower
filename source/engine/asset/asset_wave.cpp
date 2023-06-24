#include "asset_vmd.h"
#include "asset_material.h"
#include "asset_texture.h"
#include "asset_system.h"
#include "asset_wave.h"

namespace engine
{
    AssetWave::AssetWave(const std::string& assetNameUtf8, const std::string& assetRelativeRootProjectPathUtf8)
        : AssetInterface(assetNameUtf8, assetRelativeRootProjectPathUtf8)
    {

    }

    bool AssetWave::buildFromConfigs(
        const ImportConfig& config,
        const std::filesystem::path& projectRootPath,
        const std::filesystem::path& savePath,
        const std::filesystem::path& srcPath)
    {
        std::u16string assetName = savePath.filename().u16string();
        std::string assetNameUtf8 = utf8::utf16to8(assetName);
        const auto waveFileSavePath = savePath / assetName;

        if (std::filesystem::exists(savePath))
        {
            LOG_ERROR("Path {0} already exist, asset {1} import fail!", utf8::utf16to8(savePath.u16string()), assetNameUtf8);
            return false;
        }

        // Build PMX textures assets.
        if (!std::filesystem::create_directory(savePath))
        {
            LOG_ERROR("Folder {0} create failed, asset {1} import fail!", utf8::utf16to8(savePath.u16string()), assetNameUtf8);
            return false;
        }

        {
            std::filesystem::copy(srcPath, savePath);
        }

        // Save meta of pmx.
        {
            AssetWave meta(assetNameUtf8, buildRelativePathUtf8(projectRootPath, waveFileSavePath));

            meta.m_bVolumetric = config.bVolumetric;
            saveAssetMeta<AssetWave>(meta, waveFileSavePath, meta.getSuffix());
        }


        return true;
    }

    std::filesystem::path AssetWave::getWaveFilePath() const
    {
        auto projectPath = getAssetSystem()->getProjectRootPath();
        auto filePath = u"\\." + utf8::utf8to16(m_assetRelativePathUtf8);

        projectPath += filePath;
        projectPath += ".wav";

        return projectPath;
    }

}