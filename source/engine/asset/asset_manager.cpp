#include "asset_manager.h"
#include "../graphics/context.h"
#include "../engine.h"

namespace engine
{
	AssetManager* engine::getAssetManager()
	{
		static AssetManager* manager = Engine::get()->getRuntimeModule<AssetManager>();
		return manager;
	}

	void AssetManager::registerCheck(Engine* engine)
	{
	}

	bool AssetManager::init()
	{
		return true;
	}

	bool AssetManager::tick(const RuntimeModuleTickData& tickData)
	{
		return true;
	}

	bool AssetManager::beforeRelease()
	{
		return true;
	}

	bool AssetManager::release()
	{
		return true;
	}

	void AssetManager::setupProject(const std::filesystem::path& inProjectPath)
	{
        ZoneScopedN("vAssetManager::setupProject(const std::filesystem::path&)");

		m_bProjectSetup = true;

		m_projectConfig.rootPath        = inProjectPath.parent_path().u16string();
		m_projectConfig.projectFilePath = inProjectPath.u16string();
		m_projectConfig.projectName     = inProjectPath.filename().replace_extension().u16string();
		m_projectConfig.assetPath       = (inProjectPath.parent_path() / "asset" ).u16string();
		m_projectConfig.cachePath       = (inProjectPath.parent_path() / "cache" ).u16string();
		m_projectConfig.configPath      = (inProjectPath.parent_path() / "config").u16string();
		m_projectConfig.logPath         = (inProjectPath.parent_path() / "log"   ).u16string();

		// Scan whole asset folder to setup asset uuid map.
		setupProjectRecursive(m_projectConfig.assetPath);
	}

	void AssetManager::setupProjectRecursive(const std::filesystem::path& path)
	{
        ZoneScopedN("AssetManager::setupProjectRecursive(const std::filesystem::path&)");
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
			if (extension.starts_with(".dark"))
			{
				tryLoadAsset(path);
			}
		}
	}

	void AssetManager::onAssetDirty(std::shared_ptr<AssetInterface> asset)
	{
		std::lock_guard<std::recursive_mutex> lock(m_assetManagerMutex);

		const auto& id = asset->getSaveInfo().getUUID();

		// You must mark asset self dirty before register in manager.
		CHECK(asset->isDirty());

		// Asset must exist.
		CHECK(m_assets.at(id));

		// Add asset to dirty asset map.
		m_dirtyAssetIds.insert(asset->getSaveInfo().getUUID());
	}

	void AssetManager::onAssetChangeSaveInfo(
		std::shared_ptr<AssetInterface> asset, const AssetSaveInfo& newInfo)
	{
		std::lock_guard<std::recursive_mutex> lock(m_assetManagerMutex);

		if (asset->getSaveInfo() != newInfo)
		{
			CHECK(m_assets.at(asset->getSaveInfo().getUUID()));
			CHECK(!m_assets.contains(newInfo.getUUID()));

			const auto& uuid = asset->getSaveInfo().getUUID();

			removeAsset(uuid, true);
			insertAsset(newInfo.getUUID(), asset, true);
		}
	}

	void AssetManager::onAssetSaved(std::shared_ptr<AssetInterface> asset)
	{
		std::lock_guard<std::recursive_mutex> lock(m_assetManagerMutex);

		const auto& id = asset->getSaveInfo().getUUID();

		// Must dirty before call discard.
		CHECK(m_dirtyAssetIds.contains(id));

		// Clear cache in dirty asset map and asset map.
		m_dirtyAssetIds.erase(id);
	}

	std::shared_ptr<AssetInterface> AssetManager::removeAsset(const UUID& id, bool bClearDirty)
	{
		std::lock_guard<std::recursive_mutex> lock(m_assetManagerMutex);

		if (std::shared_ptr<AssetInterface> asset = m_assets[id])
		{
			std::filesystem::path savePath = asset->getSaveInfo().getStorePath();

			if (bClearDirty)
			{
				m_dirtyAssetIds.erase(id);
			}

			m_assets.erase(id);
			m_assetTypeMap[savePath.extension().string()].erase(id);

			return asset;
		}

		return nullptr;
	}

	void AssetManager::insertAsset(const UUID& uuid, std::shared_ptr<AssetInterface> asset, bool bCareDirtyState)
	{
		std::filesystem::path savePath = asset->getSaveInfo().getStorePath();

		m_assets[uuid] = asset;
		m_assetTypeMap[savePath.extension().string()].insert(uuid);

		if (asset->isDirty() && bCareDirtyState)
		{
			m_dirtyAssetIds.insert(uuid);
		}
	}

	void AssetManager::onAssetUnload(std::shared_ptr<AssetInterface> asset)
	{
		std::lock_guard<std::recursive_mutex> lock(m_assetManagerMutex);
		removeAsset(asset->getSaveInfo().getUUID(), false);
	}

	void AssetManager::onDiscardChanged(std::shared_ptr<AssetInterface> asset)
	{
		std::lock_guard<std::recursive_mutex> lock(m_assetManagerMutex);

		const auto& id = asset->getSaveInfo().getUUID();

		// Must dirty before call discard.
		CHECK(asset->isDirty());

		// Discard id all cache.
		removeAsset(id, true);
	}

}