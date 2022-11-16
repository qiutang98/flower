#include "Pch.h"
#include "AssetRegistry.h"
#include "TextureManager.h"
#include "AssetSystem.h"
#include "MeshManager.h"


namespace Flower
{
	std::unique_ptr<AssetRegistry> GAssetRegistry = std::make_unique<AssetRegistry>();

	void AssetRegistry::addEntry(std::shared_ptr<RegistryEntry> entry)
	{
		m_registryMap[entry->getRegistryUUID()] = entry;
		m_registryHeaderMap[entry->getAssetHeaderID()] = entry->getRegistryUUID();
	}

	void AssetRegistry::removeEntry(std::shared_ptr<RegistryEntry> entry)
	{
		m_registryMap.erase(entry->getRegistryUUID());
		m_registryHeaderMap.erase(entry->getAssetHeaderID());
	}

	void AssetRegistry::registerAssetMap(std::shared_ptr<AssetHeaderInterface> asset, EAssetType type)
	{
		std::unique_lock lock(m_assetMapMutex);
		m_assetMap[asset->getHeaderUUID()] = asset;
		m_assetTypeSet[size_t(type)].insert(asset->getHeaderUUID());
	}

	void AssetRegistry::save()
	{
		const auto& registryPath = GEngine->getRuntimeModule<AssetSystem>()->getProjectAssetEntryPath();
		
		// Save entry tree.
		if(isDirty())
		{
			std::ofstream os(registryPath, std::ios::binary);
			cereal::BinaryOutputArchive archive(os);
			archive(m_registryEntryRoot);
		}

		const auto& binFolderPath = GEngine->getRuntimeModule<AssetSystem>()->getProjectBinFolderPath();
		const auto& headerFolderPath = GEngine->getRuntimeModule<AssetSystem>()->getProjectHeaderFolderPath();

		// Save all dirty asset header.
		std::vector<std::future<void>> scanFutures;
		for (auto& pair : m_assetMap)
		{
			auto& asset = pair.second;
			if (asset->isDirty())
			{
				scanFutures.push_back(GThreadPool::get()->submit([&]()
				{
					if (auto binData = asset->getBinData())
					{
						// Archive bin file.
						std::ofstream os(binFolderPath / binData->getBinUUID(), std::ios::binary);
						cereal::BinaryOutputArchive archive(os);
						archive(binData);

						asset->saveCallback();

						// Release owner bin file after archive.
						asset->freeBinData();
					}

					{
						// Archive header file.
						std::ofstream os(headerFolderPath / asset->getHeaderUUID(), std::ios::binary);
						cereal::BinaryOutputArchive archive(os);
						archive(asset);
					}

					// After self archive, mark undirty.
					asset->setDirty(false);
				}));
			}
		}

		for (auto& future : scanFutures)
		{
			future.wait();
		}
		m_bDirty = false;
	}

	void AssetRegistry::setupProject(
		const std::filesystem::path& registryPath, 
		const std::filesystem::path& headerFolderPath,
		const std::filesystem::path& binFolderPath)
	{
		if (!std::filesystem::exists(registryPath))
		{
			m_registryEntryRoot = std::make_shared<RegistryEntry>();
			std::filesystem::remove_all(headerFolderPath);
			std::filesystem::remove_all(binFolderPath);
			std::filesystem::create_directory(headerFolderPath);
			std::filesystem::create_directory(binFolderPath);
			return;
		}
		
		// Build asset entry tree.
		{
			std::ifstream is(registryPath, std::ios::binary);
			cereal::BinaryInputArchive archive(is);
			archive(m_registryEntryRoot);
		}
		
		// Scan whole header folder. And fill asset map.
		{
			std::unordered_set<std::filesystem::path> unvalidPathSet{ };
			std::vector<std::future<void>> scanFutures;

			for (auto const& dirEntry : std::filesystem::directory_iterator{ headerFolderPath })
			{
				scanFutures.push_back(GThreadPool::get()->submit([dirEntry, this, &unvalidPathSet]()
				{
					const auto& path = dirEntry.path();
					auto newHeader = std::make_shared<AssetHeaderInterface>();
					{
						std::ifstream is(path, std::ios::binary);
						cereal::BinaryInputArchive archive(is);
						archive(newHeader);
					}

					if (newHeader->getHeaderUUID() == path.stem().string())
					{
						std::unique_lock lock(m_assetMapMutex);

						// Store in asset map.
						m_assetMap[newHeader->getHeaderUUID()] = newHeader;
					}
					else
					{
						unvalidPathSet.insert(path);
					}
				}));
			}
			for (auto& future : scanFutures)
			{
				future.wait();
			}

			for (const auto& pathFileNeedDelete : unvalidPathSet)
			{
				std::filesystem::remove(pathFileNeedDelete);
				LOG_WARN("Path {0} unvalid header cache so delete already.", pathFileNeedDelete.string());
			}
		}

		// Loop asset entry tree ensure all entry header is valid.
		{
			loopNodeDownToTop([this](std::shared_ptr<RegistryEntry> entry)
			{
				// Unvalid leaf entry which store unvalid asset header id.
				// We need to remove it.
				if (entry->isLeaf() && !m_assetMap.contains(entry->getAssetHeaderID()))
				{
					if (auto parent = entry->getParent().lock())
					{
						AssetRegistryManager::get()->removeChild(parent, entry, false);
					}
				}
			}, m_registryEntryRoot);

			// Copy asset map to erase.
			std::unordered_map<UUID, std::shared_ptr<AssetHeaderInterface>> entryAssetSet = m_assetMap;
			loopNodeTopToDown([this, &entryAssetSet](std::shared_ptr<RegistryEntry> entry)
			{
				// Also cache uuid map.
				addEntry(entry);

				if (entry->isLeaf())
				{
					entryAssetSet.erase(entry->getAssetHeaderID());
				}
			}, m_registryEntryRoot);

			// Erase unused header asset.
			for(auto& unusedAsset : entryAssetSet)
			{
				auto& asset = unusedAsset.second;
				
				// Remove disk unused header file.
				std::filesystem::remove(headerFolderPath / asset->getHeaderUUID());

				// Also try remove disk unused bin file.
				const auto binFilePath = binFolderPath / asset->getBinUUID();
				if (std::filesystem::exists(binFilePath))
				{
					std::filesystem::remove(binFilePath);
				}

				// Remove cache map unused header.
				m_assetMap.erase(asset->getHeaderUUID());
			}
		}

		// Prepare asset type map
		for(auto& assetPair : m_assetMap)
		{
			m_assetTypeSet[(size_t)assetPair.second->getType()].insert(assetPair.first);
		}
	}

	std::shared_ptr<AssetHeaderInterface> RegistryEntry::getHeader()
	{
		return AssetRegistryManager::get()->getHeaderMap().at(m_assetHeader);
	}

	void AssetRegistry::loopNodeDownToTop(
		const std::function<void(std::shared_ptr<RegistryEntry>)>& func,
		std::shared_ptr<RegistryEntry> node)
	{
		auto& children = node->getChildren();
		for (auto& child : children)
		{
			loopNodeDownToTop(func, child);
		}

		func(node);
	}

	void AssetRegistry::loopNodeTopToDown(
		const std::function<void(std::shared_ptr<RegistryEntry>)>& func,
		std::shared_ptr<RegistryEntry> node)
	{
		func(node);

		auto& children = node->getChildren();
		for (auto& child : children)
		{
			loopNodeTopToDown(func, child);
		}
	}

	AssetHeaderUUID AssetRegistry::importAssetTexture(
		const std::filesystem::path& inPath, 
		std::shared_ptr<RegistryEntry> entry, 
		bool bSRGB,
		float cutoff,
		bool bBuildMipmap,
		bool bHdr)
	{
		CHECK(!entry->isValid());
		
		LOG_INFO("Importing asset {0} from disk...", inPath.string());

		std::shared_ptr<ImageAssetHeader> newAssetTex = std::make_shared<ImageAssetHeader>(inPath.stem().string());
		newAssetTex->setHdr(bHdr);

		if (bHdr)
		{
			if (!newAssetTex->initFromRaw2DHDR(inPath, bBuildMipmap))
			{
				return {};
			}

			
		}
		else
		{
			if (!newAssetTex->initFromRaw2DLDR(inPath, bSRGB, cutoff, bBuildMipmap))
			{
				return {};
			}
		}

		registerAssetMap(newAssetTex, EAssetType::Texture);

		markDirty();
		return newAssetTex->getHeaderUUID();
	}

	AssetHeaderUUID AssetRegistry::importStaticMesh(const std::filesystem::path& inPath, std::shared_ptr<RegistryEntry> entry)
	{
		if (m_registryEntryRoot != entry)
		{
			CHECK(!entry->isLeaf());
		}

		LOG_INFO("Importing asset {0} from disk...", inPath.string());

		std::shared_ptr<StaticMeshAssetHeader> newAssetMesh = std::make_shared<StaticMeshAssetHeader>(inPath.stem().string());
		if (!newAssetMesh->initFromRawStaticMesh(inPath, entry))
		{
			return {};
		}

		registerAssetMap(newAssetMesh, EAssetType::StaticMesh);

		markDirty();
		return newAssetMesh->getHeaderUUID();
	}

	void AssetRegistry::removeChild(
		std::shared_ptr<RegistryEntry> inParent,
		std::shared_ptr<RegistryEntry> inChild,
		bool bRemoveInMap)
	{
		std::unique_lock lock(m_registryMapMutex);

		std::vector<std::shared_ptr<RegistryEntry>>::iterator it;

		size_t id = 0;
		while (inParent->m_children[id] != inChild)
		{
			id++;
		}

		// Swap to the end and pop.
		if (id < inParent->m_children.size())
		{
			if (bRemoveInMap)
			{
				removeEntry(inParent->m_children[id]);
			}

			std::swap(
				inParent->m_children[id], 
				inParent->m_children[inParent->m_children.size() - 1]);
			inParent->m_children.pop_back();
		}

		AssetRegistryManager::get()->markDirty();
	}

	void AssetRegistry::addChild(
		std::shared_ptr<RegistryEntry> inParent,
		std::shared_ptr<RegistryEntry> inChild,
		bool bAddInMap)
	{
		std::unique_lock lock(m_registryMapMutex);

		inParent->m_children.push_back(inChild);
		inChild->m_parent = inParent;

		if (bAddInMap)
		{
			addEntry(inChild);
		}
	}

	bool RegistryEntry::isValid() const
	{
		return AssetRegistryManager::get()->getHeaderMap().contains(m_assetHeader);
	}
}
