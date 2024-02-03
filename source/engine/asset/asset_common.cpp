#include "asset_common.h"

#include <lz4.h>

#include <rttr/registration>
#include "asset_manager.h"

namespace engine
{
    const uint32_t engine::kAssetVersion = 6;

    AssetSaveInfo::AssetSaveInfo(const u8str& name, const u8str& storeFolder)
        : m_name(name), m_storeFolder(storeFolder)
    {
        updateStorePath();
    }


    const std::filesystem::path AssetSaveInfo::toPath() const
    {
        return std::filesystem::path(getAssetManager()->getProjectConfig().assetPath) / getStorePath();
    }

    void AssetSaveInfo::setName(const u8str& newValue)
    {
        m_name = newValue;
        updateStorePath();
    }

    void AssetSaveInfo::setStoreFolder(const u8str& newValue)
    {
        m_storeFolder = newValue;
        updateStorePath();
    }

    const u8str AssetSaveInfo::kTempFolderStartChar = "*";
    const u8str AssetSaveInfo::kBuiltinFileStartChar = "~";

    AssetSaveInfo AssetSaveInfo::buildTemp(const u8str& name)
    {
        return AssetSaveInfo(name, kTempFolderStartChar + buildUUID());
    }

    AssetSaveInfo AssetSaveInfo::buildRelativeProject(const std::filesystem::path& savePath)
    {
        auto fileName = savePath.filename();

        auto saveFolder = savePath;
        saveFolder.remove_filename();

        const auto relativePath = buildRelativePathUtf8(getAssetManager()->getProjectConfig().assetPath, saveFolder);

        return AssetSaveInfo(utf8::utf16to8(fileName.u16string()), relativePath);
    }

    bool AssetSaveInfo::isTemp() const
    {
        return m_storeFolder.starts_with(kTempFolderStartChar);
    }

    bool AssetSaveInfo::isBuiltin() const
    {
        return isTemp() && m_name.starts_with(kBuiltinFileStartChar);
    }


    bool AssetSaveInfo::canUseForCreateNewAsset() const
    {
        if (empty() || alreadyInDisk())
        {
            return false;
        }

        // Find memory exist or not.
        return !getAssetManager()->isAssetSavePathExist(*this);
    }

    bool AssetSaveInfo::alreadyInDisk() const
    {
        CHECK(getAssetManager()->isProjectSetup());

        // Find disk exist or not.
        std::filesystem::path path = getAssetManager()->getProjectConfig().assetPath;
        path /= this->getStorePath();

        return std::filesystem::exists(path);
    }



    void AssetSaveInfo::updateStorePath()
    {
        ZoneScoped;
        if (m_storeFolder.starts_with("\\") || m_storeFolder.starts_with("/"))
        {
            m_storeFolder = m_storeFolder.erase(0, 1);
        }

        const std::filesystem::path storeFolder = utf8::utf8to16(m_storeFolder);
        const std::filesystem::path storeName = utf8::utf8to16(m_name);

        m_storePath = utf8::utf16to8((storeFolder / storeName).u16string());
    }



    bool engine::loadAssetBinaryWithDecompression(
        std::vector<uint8_t>& out, 
        const std::filesystem::path& rawSavePath)
    {
        ZoneScoped;
        if (!std::filesystem::exists(rawSavePath))
        {
            LOG_ERROR("Binary data {} miss!", utf8::utf16to8(rawSavePath.u16string()));
            return false;
        }

        std::ifstream is(rawSavePath, std::ios::binary);
        cereal::BinaryInputArchive archive(is);

        AssetCompressionHelper sizeHelper;

        std::vector<uint8_t> compressionData;
        archive(sizeHelper, compressionData);

        // Resize to src data.
        out.resize(sizeHelper.originalSize);

        LZ4_decompress_safe((const char*)compressionData.data(), (char*)out.data(), sizeHelper.compressionSize, sizeHelper.originalSize);
        return true;
    }

    bool engine::saveAssetBinaryWithCompression(
        const uint8_t* out, 
        int size, 
        const std::filesystem::path& savePath, 
        const char* suffix)
    {
        ZoneScoped;
        std::filesystem::path rawSavePath = savePath;
        rawSavePath += suffix;

        if (std::filesystem::exists(rawSavePath))
        {
            LOG_ERROR("Binary data {} already exist, make sure never import save resource at same folder!", utf8::utf16to8(rawSavePath.u16string()));
            return false;
        }

        // Save to disk.
        std::ofstream os(rawSavePath, std::ios::binary);
        cereal::BinaryOutputArchive archive(os);

        // LZ4 compression.
        std::vector<uint8_t> compressionData;

        // Compress and shrink.
        auto compressStaging = LZ4_compressBound(size);
        compressionData.resize(compressStaging);
        auto compressedSize = LZ4_compress_default((const char*)out, (char*)compressionData.data(), size, compressStaging);
        compressionData.resize(compressedSize);

        AssetCompressionHelper sizeHelper
        {
            .originalSize = size,
            .compressionSize = compressedSize,
        };

        archive(sizeHelper, compressionData);
        return true;
    }
}