#include "Pch.h"
#include "AssetCommon.h"
#include "AssetSystem.h"

namespace Flower
{
	void AssetHeaderInterface::setCacheBinData(std::shared_ptr<AssetBinInterface> inBin)
	{
		// CHECK(m_cacheBinData == nullptr);

		m_cacheBinData = inBin;
		m_cacheBinDataRef = m_cacheBinData;
		m_binDataUUID = m_cacheBinData->getBinUUID();

		markDirty();
	}

	std::shared_ptr<AssetBinInterface> AssetHeaderInterface::loadBinData()
	{
		// Return if exist.
		if (auto cacheBin = getBinData())
		{
			return cacheBin;
		}

		// No exist, need load bin.
		const auto& binFolderPath = GEngine->getRuntimeModule<AssetSystem>()->getProjectBinFolderPath();
		std::ifstream is(binFolderPath / getBinUUID(), std::ios::binary);
		cereal::BinaryInputArchive archive(is);

		std::shared_ptr<AssetBinInterface> binDataPtr;
		archive(binDataPtr);

		return binDataPtr;
	}

	std::shared_ptr<AssetBinInterface> AssetHeaderInterface::getBinData()
	{
		if (m_cacheBinData)
		{
			return m_cacheBinData;
		}
		else if (auto cacheWeakData = m_cacheBinDataRef.lock())
		{
			// When asset header archive save, it will release m_cacheBinData.
			// But it may still valid on other system, so try fallback to waek ptr reference.
			return cacheWeakData;
		}

		// No bin data, need load.
		return nullptr;
	}
}