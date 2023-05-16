#pragma once

#include "asset_common.h"

namespace engine
{
	class AssetSystem final : public IRuntimeModule
	{
	public:
		AssetSystem(Engine* engine) : IRuntimeModule(engine) { }
		~AssetSystem() = default;

		virtual void registerCheck(Engine* engine) override;
		virtual bool init() override;
		virtual bool tick(const RuntimeModuleTickData& tickData) override;
		virtual void release() override;

		// Get cache context.
		VulkanContext* getContext() const { return m_context; }

		void setupProject(const std::filesystem::path& path);
		const auto& getAssetMap() const { return m_assetMap; }

		std::shared_ptr<AssetInterface> getAssetByRelativeMap(const std::string& path) const;
		std::shared_ptr<AssetInterface> getAsset(const UUID& uuid) const;
		const auto& getProjectAssetRootPath() const { return m_assetRootPath; }
		const auto& getProjectRootPath() const { return m_projectRootPath; }
		const auto& getAssetTypeMap() const { return m_typeMap; }

		const auto& getAssetMap(EAssetType type) { return m_typeMap[type]; }

		MulticastDelegate<std::shared_ptr<AssetInterface>> onAssetDirty;

	private:
		void clearCache();

		void setupProjectRecursive(const std::filesystem::path& path);

		template<typename T>
		bool insertAsset(const std::filesystem::path& path)
		{
			static_assert(std::is_base_of_v<AssetInterface, T>, "T must derived from AssetInterface.");

			std::shared_ptr<AssetInterface> asset;

			if (loadAsset(asset, path))
			{
				const UUID uuid = asset->getUUID();
				const auto relativePath = asset->getRelativePathUtf8();

				ASSERT(m_assetMap[uuid] == nullptr, "Insert asset repeat!");
				ASSERT(m_relativeAssetPathUtf8Map[relativePath].empty(), "Insert asset repeat!");

				m_assetMap[uuid] = asset;
				m_relativeAssetPathUtf8Map[relativePath] = uuid;

				m_typeMap[asset->getType()].insert(uuid);

				return true;
			}
			else
			{
				LOG_ERROR("Fail to load asset in path {}.", utf8::utf16to8(path.u16string()));
				return false;
			}
		};

	public:
		template<typename T>
		bool reloadAsset(std::shared_ptr<AssetInterface> inAsset)
		{
			static_assert(std::is_base_of_v<AssetInterface, T>, "T must derived from AssetInterface.");
			std::filesystem::path path = inAsset->getSavePath().u16string() + utf8::utf8to16(inAsset->getSuffix());

			std::shared_ptr<AssetInterface> asset;
			if (loadAsset(asset, path))
			{
				const UUID uuid = asset->getUUID();
				m_assetMap[uuid] = asset;
				return true;
			}
			else
			{
				LOG_ERROR("Fail to load asset in path {}.", utf8::utf16to8(path.u16string()));
				return false;
			}
		}

	protected:
		// RHI context.
		VulkanContext* m_context;

		// Asset project root path.
		std::filesystem::path m_assetRootPath;
		std::filesystem::path m_projectRootPath;

		// Cache asset map.
		std::unordered_map<UUID, std::shared_ptr<AssetInterface>> m_assetMap;
		std::unordered_map<std::string, UUID> m_relativeAssetPathUtf8Map;

		std::unordered_map<EAssetType, std::unordered_set<UUID>, EnumClassHash> m_typeMap;
	};

	extern AssetSystem* getAssetSystem();
}