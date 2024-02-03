#include "asset.h"

#include <filesystem>
#include <rttr/registration>

#include <nameof/nameof.hpp>
#include "asset_manager.h"
#include "asset.h"
#include "asset.h"
#include "asset.h"
#include "asset.h"
#include <engine/graphics/context.h>
#include <engine/asset/assimp_import.h>

namespace engine
{
	AssetInterface::AssetInterface(const AssetSaveInfo& saveInfo)
		: m_saveInfo(saveInfo)
	{

	}

	std::string AssetInterface::getSuffix() const
	{
		std::string finalName = std::format(".{}", nameof::nameof_enum(getType()));
		return finalName;
	}

	// Unload asset from memory.
	void AssetInterface::unload()
	{
		unloadImpl();
		getAssetManager()->onAssetUnload(shared_from_this());
	}

	bool AssetInterface::save()
	{
		if (!isDirty())
		{
			return false;
		}

		if (m_saveInfo.empty())
		{
			LOG_ERROR("You must config save info before save an asset!");
			return false;
		}

		const bool bNewlySavedToDisk = !m_saveInfo.alreadyInDisk();

		bool bSaveResult = saveImpl();
		if (bSaveResult)
		{
			getAssetManager()->onAssetSaved(shared_from_this());
			m_bDirty = false;
		}

		if (bNewlySavedToDisk)
		{
			getAssetManager()->onAssetNewlySavedToDisk.broadcast(shared_from_this());
		}

		return bSaveResult;
	}

	bool AssetInterface::markDirty()
	{
		if (m_bDirty)
		{
			return false;
		}

		m_bDirty = true;
		getAssetManager()->onAssetDirty(getptr<AssetInterface>());

		return true;
	}

	std::filesystem::path AssetInterface::getSavePath() const
	{
		if (m_saveInfo.empty())
		{
			return {};
		}

		std::filesystem::path asset = getAssetManager()->getProjectConfig().assetPath;

		std::u16string storePath = m_saveInfo.getStorePath();
		auto result = asset / storePath;
		auto result2 = asset.append(storePath);
		return result;
	}

	void AssetInterface::discardChanged()
	{
		if (isDirty())
		{
			getAssetManager()->onDiscardChanged(shared_from_this());
			
			m_bDirty = false;
		}
	}

	bool AssetInterface::changeSaveInfo(const AssetSaveInfo& newInfo)
	{
		if (newInfo == m_saveInfo)
		{
			return false;
		}

		if (!newInfo.canUseForCreateNewAsset())
		{
			return false;
		}

		getAssetManager()->onAssetChangeSaveInfo(shared_from_this(), newInfo);
		m_saveInfo = newInfo;

		markDirty();
		return true;
	}

	UUID AssetInterface::getSnapshotUUID() const
	{
		return m_saveInfo.getSnapshotUUID();
	}

	std::filesystem::path AssetInterface::getSnapshotPath() const
	{
		std::u16string p = utf8::utf8to16(getSnapshotUUID());
		std::filesystem::path cache = getAssetManager()->getProjectConfig().cachePath;

		return cache / p;
	}

	UUID AssetInterface::getBinUUID() const
	{
		return m_saveInfo.getBinUUID();
	}

	std::filesystem::path AssetInterface::getBinPath() const
	{
		std::u16string p = utf8::utf8to16(getBinUUID());
		std::filesystem::path cache = getAssetManager()->getProjectConfig().cachePath;

		return cache / p;
	}

	VulkanImage* AssetInterface::getSnapshotImage()
	{
		return getContext()->getBuiltinTextureWhite()->getReadyImage();
	}

	std::filesystem::path AssetInterface::getRawAssetPath() const
	{
		CHECK(!m_rawAssetPath.empty());

		std::filesystem::path asset = getAssetManager()->getProjectConfig().assetPath;

		return asset / utf8::utf8to16(m_rawAssetPath);
	}
}

