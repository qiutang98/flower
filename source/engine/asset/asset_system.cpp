#include "asset_system.h"
#include "asset_texture.h"
#include "asset_staticmesh.h"
#include "asset_material.h"
#include <scene/scene.h>

namespace engine
{
	void AssetSystem::registerCheck(Engine* engine)
	{
		ASSERT(engine->existRegisteredModule<VulkanContext>(),
			"When asset system enable, you must register vulkan context module before asset system.");
	}

    bool AssetSystem::init()
    {
        // Fetch and cache vulkan context handle.
        m_context = m_engine->getRuntimeModule<VulkanContext>();

        return true;
    }

    bool AssetSystem::tick(const RuntimeModuleTickData& tickData)
    {

        return true;
    }

    void AssetSystem::release()
    {
        clearCache();
    }

    void AssetSystem::setupProject(const std::filesystem::path& projectFilePath)
    {
        clearCache();

        m_projectRootPath = projectFilePath.parent_path();
        m_assetRootPath = projectFilePath.parent_path() / "asset";
        setupProjectRecursive(m_assetRootPath);
    }

    std::shared_ptr<AssetInterface> AssetSystem::getAssetByRelativeMap(const std::string& path) const
    {
        return m_assetMap.at(m_relativeAssetPathUtf8Map.at(path));
    }

    std::shared_ptr<AssetInterface> AssetSystem::getAsset(const UUID& uuid) const
    {
        return m_assetMap.at(uuid);
    }

    void AssetSystem::clearCache()
    {
        m_assetMap.clear();
        m_relativeAssetPathUtf8Map.clear();
        m_typeMap.clear();
    }

    void AssetSystem::setupProjectRecursive(const std::filesystem::path& path)
    {
        const bool bFolder = std::filesystem::is_directory(path);

        if (bFolder)
        {
            for (const auto& entry : std::filesystem::directory_iterator(path))
            {
                setupProjectRecursive(entry);
            }
        }
        else
        {
            const auto extension = path.extension().string();
            if (isEngineMetaAsset(extension))
            {
                if (isAssetTextureMeta(extension))         { insertAsset<AssetTexture>(path); }
                else if (isAssetStaticMeshMeta(extension)) { insertAsset<AssetStaticMesh>(path); }
                else if (isAssetMaterialMeta(extension))   { insertAsset<AssetMaterial>(path); }
                else if (isAssetPMXMeta(extension))        { insertAsset<AssetPMX>(path); }
                else if (isAssetSceneMeta(extension))      { insertAsset<Scene>(path);  }
                else
                {
                    UN_IMPLEMENT();
                }
            }
        }
    }

    AssetSystem* getAssetSystem()
    {
        static AssetSystem* system = Framework::get()->getEngine().getRuntimeModule<AssetSystem>();
        return system;
    }
}

