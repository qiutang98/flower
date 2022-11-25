#pragma once
#include "../Pch.h"
#include "../Core/Core.h"
#include "../Core/UUID.h"
#include "../RHI/RHI.h"

namespace Flower
{
	constexpr size_t GAssetTextureChannels = 4;
	constexpr size_t GAssetSnapshotMaxDim = 128;

	enum class EAssetType : size_t
	{
		StaticMesh = 0,
		Texture,
		Material,
		Max
	};

	using AssetHeaderUUID = UUID;
	using AssetBinUUID = UUID;

	class AssetBinInterface
	{
		ARCHIVE_DECLARE;
	protected:
		AssetBinUUID m_uuid;

	public:
		AssetBinInterface() = default;

		AssetBinInterface(AssetBinUUID uuid, const std::string& name)
			: m_uuid(uuid)
		{

		}

		const AssetBinUUID& getBinUUID() const
		{
			return m_uuid;
		}

		virtual EAssetType getType() const = 0;
	};


	class AssetHeaderInterface
	{
		ARCHIVE_DECLARE;
	protected:
		std::string m_assetName;

		// Header UUID.
		AssetHeaderUUID m_uuid;

		// Bin UUID.
		AssetBinUUID m_binDataUUID;

	protected:
		// Some runtime states.
		bool m_bDirty = true;

		std::shared_ptr<AssetBinInterface> m_cacheBinData = nullptr;
		std::weak_ptr<AssetBinInterface>   m_cacheBinDataRef;



	public:
		AssetHeaderInterface() = default;
		AssetHeaderInterface(AssetHeaderUUID uuid, const std::string& name)
			: m_uuid(uuid), m_assetName(name)
		{

		}

		void setDirty(bool bState)
		{
			m_bDirty = bState;
		}

		const bool isDirty() const
		{
			return m_bDirty;
		}

		const AssetHeaderUUID& getHeaderUUID() const
		{
			return m_uuid;
		}

		const AssetBinUUID& getBinUUID() const
		{
			return m_binDataUUID;
		}

		const std::string& getName() const
		{
			return m_assetName;
		}

		void markDirty()
		{
			m_bDirty = true;
		}

		virtual EAssetType getType() const
		{
			return EAssetType::Max;
		}

		virtual void saveCallback()
		{

		}

		void freeBinData()
		{
			m_cacheBinData = nullptr;
		}

		// Return cache bin data, if no cache return nullptr.
		std::shared_ptr<AssetBinInterface> getBinData();

		// Load bin data.
		std::shared_ptr<AssetBinInterface> loadBinData();

		void setCacheBinData(std::shared_ptr<AssetBinInterface> inBin);

		template<typename T>
		std::shared_ptr<T> getBinData()
		{
			return std::dynamic_pointer_cast<T>(getBinData());
		}
	};
}