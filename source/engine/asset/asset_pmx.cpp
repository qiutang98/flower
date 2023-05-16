#include "asset_pmx.h"
#include "asset_material.h"
#include "asset_texture.h"
#include "asset_system.h"

namespace engine
{
    AssetPMX::AssetPMX(const std::string& assetNameUtf8, const std::string& assetRelativeRootProjectPathUtf8)
        : AssetInterface(assetNameUtf8, assetRelativeRootProjectPathUtf8)
    {

    }

    bool AssetPMX::buildFromConfigs(
        const ImportConfig& config, 
        const std::filesystem::path& projectRootPath, 
        const std::filesystem::path& savePath, 
        const std::filesystem::path& srcPath)
    {
        std::string assetNameUtf8 = utf8::utf16to8(savePath.filename().u16string());
        const auto pmxFileSavePath = savePath / assetNameUtf8;

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

        // Copy all pmx file in our project.
        const auto pmxFolderPath  = savePath / "pmx";
        std::filesystem::create_directory(pmxFolderPath);

        //
        {
            std::filesystem::copy(srcPath.parent_path(), pmxFolderPath, std::filesystem::copy_options::recursive);
        }

        // Save meta of pmx.
        {
            AssetPMX meta(assetNameUtf8, buildRelativePathUtf8(projectRootPath, pmxFileSavePath));


            saveAssetMeta<AssetPMX>(meta, pmxFileSavePath, meta.getSuffix());
        }


        return true;
    }
}