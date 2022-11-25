#pragma once
#include "AssetCommon.h"
#include "AssetSystem.h"

namespace Flower
{
	using RegistryUUID = UUID;
	using AssetTypeKey = size_t;



	// Asset registry is an asset tree.
	// We also cache some asset info map for quick search.

	class RegistryEntry : public std::enable_shared_from_this<RegistryEntry>
	{
		ARCHIVE_DECLARE;

		friend class AssetRegistry;

	private:
		std::string m_name;
		RegistryUUID m_uuid;

		// Hierarchy structure.
		std::weak_ptr<RegistryEntry> m_parent;
		std::vector<std::shared_ptr<RegistryEntry>> m_children;

		// Asset header uuid.
		AssetHeaderUUID m_assetHeader;

	public:
		RegistryEntry() = default;
		RegistryEntry(AssetHeaderUUID inAssetHeaderUUID, std::string name)
			: m_assetHeader(inAssetHeaderUUID), m_name(name), m_uuid(buildUUID())
		{

		}

		bool isLeaf() const
		{
			return m_children.size() == 0;
		}

		const auto& getChildren() const
		{
			return m_children;
		}

		auto& getChildren()
		{
			return m_children;
		}

		const auto& getParent()
		{
			return m_parent;
		}

		const AssetHeaderUUID& getAssetHeaderID() const
		{
			return m_assetHeader;
		}

		const std::string& getName() const
		{
			return m_name;
		}

		std::shared_ptr<AssetHeaderInterface> getHeader();
		RegistryUUID getRegistryUUID() const
		{
			return m_uuid;
		}

		bool isValid() const;
	};

	
	class AssetRegistry
	{
	private:
		bool m_bDirty = false;

		// registry misc
		std::mutex m_registryMapMutex;
		std::shared_ptr<RegistryEntry> m_registryEntryRoot;
		std::unordered_map<RegistryUUID, std::weak_ptr<RegistryEntry>> m_registryMap;
		std::unordered_map<AssetHeaderUUID, RegistryUUID> m_registryHeaderMap;

		// asset misc.
		std::mutex m_assetMapMutex;
		std::unordered_map<AssetHeaderUUID, std::shared_ptr<AssetHeaderInterface>> m_assetMap;
		std::unordered_map<AssetTypeKey, std::unordered_set<AssetHeaderUUID>> m_assetTypeSet;


		void addEntry(std::shared_ptr<RegistryEntry>);
		void removeEntry(std::shared_ptr<RegistryEntry>);

		
	public:
		AssetRegistry() = default;

		std::mutex& getRegistryMapMutex()
		{
			return m_registryMapMutex;
		}

		std::mutex& getAssetMapMutex()
		{
			return m_assetMapMutex;
		}

		void markDirty()
		{
			m_bDirty = true;
		}

		bool isDirty() const
		{
			return m_bDirty;
		}

		void save();

		void setupProject(
			const std::filesystem::path& registryPath, 
			const std::filesystem::path& headerFolderPath,
			const std::filesystem::path& binFolderPath);

		std::shared_ptr<RegistryEntry> getRoot() const
		{
			return m_registryEntryRoot;
		}

		const auto& getHeaderMap() const
		{
			return m_assetMap;
		}

		const auto& getEntryMap() const
		{
			return m_registryMap;
		}

		const auto& getEntryHeaderMap() const
		{
			return m_registryHeaderMap;
		}

		const auto& getTypeAssetSetMap() const
		{
			return m_assetTypeSet;
		}


		void release()
		{
			m_registryEntryRoot.reset();
		}

		// Add child without relationship check.
		void addChild(
			std::shared_ptr<RegistryEntry> inParent,
			std::shared_ptr<RegistryEntry> inChild,
			bool bAddInMap);

		// Remove child without relationship check.
		void removeChild(
			std::shared_ptr<RegistryEntry> inParent,
			std::shared_ptr<RegistryEntry> inChild,
			bool bRemoveInMap);

		// post-order loop.
		void loopNodeDownToTop(
			const std::function<void(std::shared_ptr<RegistryEntry>)>& func, 
			std::shared_ptr<RegistryEntry> node);

		// pre-order loop.
		void loopNodeTopToDown(
			const std::function<void(std::shared_ptr<RegistryEntry>)>& func, 
			std::shared_ptr<RegistryEntry> node);

		AssetHeaderUUID importAssetTexture(
			const std::filesystem::path& inPath, 
			std::shared_ptr<RegistryEntry> entry, 
			bool bSRGB, 
			float cutoff, 
			bool bBuildMipmap, 
			bool bHdr);

		AssetHeaderUUID importStaticMesh(
			const std::filesystem::path& inPath, 
			std::shared_ptr<RegistryEntry> entry);

		void registerAssetMap(std::shared_ptr<AssetHeaderInterface> asset, EAssetType type);

	};

	using AssetRegistryManager = Singleton<AssetRegistry>;
}