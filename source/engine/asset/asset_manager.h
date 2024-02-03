#pragma once
#include "../utils/utils.h"
#include "asset_common.h"

namespace engine
{
	struct ProjectConfig
	{
		// Project stem name, not include period and suffix.
		std::u16string projectName;

		// Project file absolute path file in this computer, this is runtime generate value.
		std::u16string projectFilePath;

		// Misc project folder path.
		std::u16string rootPath;
		std::u16string assetPath;
		std::u16string configPath;
		std::u16string logPath;
		std::u16string cachePath;
	};

	class AssetManager : public IRuntimeModule
	{
		friend AssetInterface;

	public:
		explicit AssetManager(Engine* engine) : IRuntimeModule(engine) { }
		virtual ~AssetManager() = default;

		virtual void registerCheck(Engine* engine) override;
		virtual bool init() override;
		virtual bool tick(const RuntimeModuleTickData& tickData) override;
		virtual bool beforeRelease() override;
		virtual bool release() override;

		const ProjectConfig& getProjectConfig() const { return m_projectConfig; }
		void setupProject(const std::filesystem::path& inProjectPath);

		void setupProjectRecursive(const std::filesystem::path& path);
		
		bool isProjectSetup() const { return m_bProjectSetup; }

		template<typename T>
		std::vector<std::weak_ptr<T>> getDirtyAsset() const
		{
			std::lock_guard<std::recursive_mutex> lock(m_assetManagerMutex);

			std::vector<std::weak_ptr<T>> result { };
			for (const auto& id : m_dirtyAssetIds)
			{
				if (auto sp = std::dynamic_pointer_cast<T>(m_assets.at(id)))
				{
					result.push_back(sp);
				}
			}
			return result;
		}

		bool existDirtyAsset() const
		{
			std::lock_guard<std::recursive_mutex> lock(m_assetManagerMutex);

			return !m_dirtyAssetIds.empty();
		}

		bool isAssetSavePathExist(const AssetSaveInfo& info) const
		{
			std::lock_guard<std::recursive_mutex> lock(m_assetManagerMutex);

			return m_assets.contains(info.getUUID());
		}

		template<typename T>
		std::weak_ptr<T> createAsset(const AssetSaveInfo& saveInfo = {})
		{
			std::lock_guard<std::recursive_mutex> lock(m_assetManagerMutex);
			const UUID& uuid = saveInfo.getUUID();

			static_assert(std::is_constructible_v<T, const AssetSaveInfo&>);
			static_assert(std::is_base_of_v<AssetInterface, T>);

			// Ensure save info is valid.
			if (!saveInfo.isTemp())
			{
				CHECK(saveInfo.canUseForCreateNewAsset());
			}


			// Create a new asset and mark it dirty.
			auto newAsset = std::make_shared<T>(saveInfo);
			insertAsset(uuid, newAsset, false);

			// Callback function after construct.
			newAsset->onPostAssetConstruct();
			return std::dynamic_pointer_cast<T>(newAsset);
		}

		template<typename T>
		std::weak_ptr<T> getOrLoadAsset(const std::filesystem::path& savePath)
		{
			auto asset = tryLoadAsset(savePath);
			std::shared_ptr<T> result = std::dynamic_pointer_cast<T>(asset);
			return result;
		}

		// A new asset save to disk file.
		MulticastDelegate<std::shared_ptr<AssetInterface>> onAssetNewlySavedToDisk;

		const auto& getAssetTypeMap(const std::string& type) { return m_assetTypeMap[type]; }
		std::shared_ptr<AssetInterface> getAsset(const UUID& id) const { return m_assets.at(id); }
	private:
		std::shared_ptr<AssetInterface> tryLoadAsset(const std::filesystem::path& savePath)
		{
			std::lock_guard<std::recursive_mutex> lock(m_assetManagerMutex);

			const AssetSaveInfo saveInfo = AssetSaveInfo::buildRelativeProject(savePath);
			const UUID& uuid = saveInfo.getUUID();
			CHECK(saveInfo.alreadyInDisk());

			if (!m_assets[uuid])
			{
				std::shared_ptr<AssetInterface> asset;
				CHECK(loadAsset(asset, savePath));

				insertAsset(uuid, asset, false);
			}

			return m_assets[uuid];
		}

		// Discard asset edited or created state.
		void onDiscardChanged(std::shared_ptr<AssetInterface> asset);

		void onAssetDirty(std::shared_ptr<AssetInterface> asset);

		void onAssetChangeSaveInfo(
			std::shared_ptr<AssetInterface> asset, const AssetSaveInfo& newInfo);

		void onAssetSaved(std::shared_ptr<AssetInterface> asset);
		void onAssetUnload(std::shared_ptr<AssetInterface> asset);

		std::shared_ptr<AssetInterface> removeAsset(const UUID& id, bool bClearDirty);
		void insertAsset(const UUID& uuid, std::shared_ptr<AssetInterface> asset, bool bCareDirtyState);


	protected:
		bool m_bProjectSetup = false;

		ProjectConfig m_projectConfig;

		mutable std::recursive_mutex m_assetManagerMutex;

		// Map store dirty asset ids.
		std::unordered_set<UUID> m_dirtyAssetIds;

		// Map store all engine assetes.
		std::unordered_map<UUID, std::shared_ptr<AssetInterface>> m_assets;
		std::unordered_map<std::string, std::unordered_set<UUID>> m_assetTypeMap;
	};

	extern AssetManager* getAssetManager();
}