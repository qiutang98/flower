#include "asset_vmd.h"
#include "asset_material.h"
#include "asset_texture.h"
#include "asset_system.h"

namespace engine
{
    AssetVMD::AssetVMD(const std::string& assetNameUtf8, const std::string& assetRelativeRootProjectPathUtf8)
        : AssetInterface(assetNameUtf8, assetRelativeRootProjectPathUtf8)
    {

    }

    bool AssetVMD::buildFromConfigs(
        const ImportConfig& config,
        const std::filesystem::path& projectRootPath,
        const std::filesystem::path& savePath,
        const std::filesystem::path& srcPath)
    {
        std::u16string assetName = savePath.filename().u16string();
        std::string assetNameUtf8 = utf8::utf16to8(assetName);
        const auto vmdFileSavePath = savePath / assetName;

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

        // TODO: Copy vmd file here.
        {
            std::filesystem::copy(srcPath, savePath);
        }

        // Save meta of pmx.
        {
            AssetVMD meta(assetNameUtf8, buildRelativePathUtf8(projectRootPath, vmdFileSavePath));

            meta.m_bCamera = config.bCamera;
            saveAssetMeta<AssetVMD>(meta, vmdFileSavePath, meta.getSuffix());
        }


        return true;
    }

    std::filesystem::path AssetVMD::getVMDFilePath() const
    {
        auto projectPath = getAssetSystem()->getProjectRootPath();
        auto filePath = "\\." + m_assetRelativePathUtf8;

        projectPath += filePath;
        projectPath += ".vmd";

        return projectPath;
    }

}