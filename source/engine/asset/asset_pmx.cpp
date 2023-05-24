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
        // Importance: No all pmx allow edit second times, so just follow the euler and don't change asset.
        const auto pmxFolderPath  = savePath / "pmx";
        std::filesystem::create_directory(pmxFolderPath);

        //
        {
            std::filesystem::copy(srcPath.parent_path(), pmxFolderPath, std::filesystem::copy_options::recursive);
        }

        // Save meta of pmx.
        {
			AssetPMX meta(assetNameUtf8, buildRelativePathUtf8(projectRootPath, pmxFileSavePath));

			auto pmxModel = std::make_unique<saba::PMXModel>();
			{
				auto path = meta.getPMXFilePath();
				std::string pmxPath = path.string();
				if (!pmxModel->Load(pmxPath, "image/mmd"))
				{
					LOG_ERROR("Failed to load pmx file {0}.", pmxPath);
					return false;
				}
			}

            {
                size_t matCount = pmxModel->GetMaterialCount();
                const saba::MMDMaterial* materials = pmxModel->GetMaterials();

                meta.m_materials.resize(matCount);
                for (size_t i = 0; i < matCount; i++)
                {
                    meta.m_materials[i].material = materials[i];
                    meta.m_materials[i].bTranslucent = materials[i].m_alpha < 0.999f;
                }
            }

            saveAssetMeta<AssetPMX>(meta, pmxFileSavePath, meta.getSuffix());
        }


        return true;
    }

    std::filesystem::path AssetPMX::getPMXFilePath() const
    {
        auto projectPath = getAssetSystem()->getProjectRootPath();
        auto filePath = "\\." + m_assetRelativePathUtf8;

        projectPath += filePath;
        auto pmxFolder = projectPath.parent_path() / "pmx";
        pmxFolder /= m_assetNameUtf8;

        pmxFolder += ".pmx";
        return pmxFolder;
    }

	// NOTE: Current only reuse texture in single pmx file.
    void AssetPMX::tryLoadAllTextures(const saba::PMXModel& pmx)
    {
        std::set<std::string> texLoaded{};

		size_t matCount = pmx.GetMaterialCount();
		const saba::MMDMaterial* materials = pmx.GetMaterials();

		for (size_t i = 0; i < matCount; i++)
		{
			saba::MMDMaterial matMMD = materials[i];
			auto& workingMat = m_materials.at(i);

			CHECK(workingMat.material.m_name == matMMD.m_name);
			CHECK(workingMat.material.m_enName == matMMD.m_enName);

			// texture.
			if (!matMMD.m_texture.empty() && std::filesystem::exists(matMMD.m_texture)
				&& !texLoaded.contains(matMMD.m_texture) && !getContext()->isLRUAssetExist(matMMD.m_texture))
			{
				getContext()->getAsyncUploader().addTask(RawAssetTextureLoadTask::buildTexture(
					false,
					getContext(),
					matMMD.m_texture, // 
					matMMD.m_texture, // UUID use path here.
					VK_FORMAT_R8G8B8A8_SRGB,
					true,
					true));
				texLoaded.insert(matMMD.m_texture);
			}

			// Sp texture.
			if (!matMMD.m_spTexture.empty() && std::filesystem::exists(matMMD.m_spTexture)
				&& !texLoaded.contains(matMMD.m_spTexture) && !getContext()->isLRUAssetExist(matMMD.m_spTexture))
			{
				getContext()->getAsyncUploader().addTask(RawAssetTextureLoadTask::buildTexture(
					false,
					getContext(),
					matMMD.m_spTexture,
					matMMD.m_spTexture, // UUID use path here.
					VK_FORMAT_R8G8B8A8_SRGB,
					true,
					false));
				texLoaded.insert(matMMD.m_spTexture);
			}

			// toon texture.
			if (!matMMD.m_toonTexture.empty() && std::filesystem::exists(matMMD.m_toonTexture)
				&& !texLoaded.contains(matMMD.m_toonTexture) && !getContext()->isLRUAssetExist(matMMD.m_toonTexture))
			{
				getContext()->getAsyncUploader().addTask(RawAssetTextureLoadTask::buildTexture(
					false,
					getContext(),
					matMMD.m_toonTexture,
					matMMD.m_toonTexture, // UUID use path here.
					VK_FORMAT_R8G8B8A8_SRGB,
					true,
					false));
				texLoaded.insert(matMMD.m_toonTexture);
			}
		}
		getContext()->getAsyncUploader().flushTask();

		// Fill material id.
		for (size_t i = 0; i < matCount; i++)
		{
			saba::MMDMaterial matMMD = materials[i];
			auto& workingMat = m_materials.at(i);

			CHECK(workingMat.material.m_name == matMMD.m_name);
			CHECK(workingMat.material.m_enName == matMMD.m_enName);

			const uint32_t fallbackWhite = getContext()->getEngineTextureWhite()->getBindlessIndex();

			// texture.
			if (!matMMD.m_texture.empty() && std::filesystem::exists(matMMD.m_texture))
			{
				workingMat.mmdTex = getContext()->getOrCreateTextureAsset(matMMD.m_texture)->getBindlessIndex();
			}
			else
			{
				LOG_WARN("Lose tex {} in material {}, use fallback white.", matMMD.m_texture, matMMD.m_name);
				workingMat.mmdTex = fallbackWhite;
			}

			// Sp texture.
			if (!matMMD.m_spTexture.empty() && std::filesystem::exists(matMMD.m_spTexture))
			{
				workingMat.mmdSphereTex = getContext()->getOrCreateTextureAsset(matMMD.m_spTexture)->getBindlessIndex();
			}
			else
			{
				LOG_WARN("Lose sp tex {} in material {}, use fallback white.", matMMD.m_spTexture, matMMD.m_name);
				workingMat.mmdSphereTex = ~0;
			}

			// toon texture.
			if (!matMMD.m_toonTexture.empty() && std::filesystem::exists(matMMD.m_toonTexture))
			{
				workingMat.mmdToonTex = getContext()->getOrCreateTextureAsset(matMMD.m_toonTexture)->getBindlessIndex();
			}
			else
			{
				LOG_WARN("Lose toon tex {} in material {}, use fallback white.", matMMD.m_toonTexture, matMMD.m_name);
				workingMat.mmdToonTex = ~0;
			}
		}
	}

	bool AssetPMX::saveActionImpl()
	{
		saveAssetMeta<AssetPMX>(*this, getSavePath(), this->getSuffix(), false);
		return true;
	}
}