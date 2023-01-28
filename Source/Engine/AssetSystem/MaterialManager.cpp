#include "Pch.h"
#include "MaterialManager.h"
#include "AssetRegistry.h"

namespace Flower
{
	const GPUStaticMeshStandardPBRMaterial& StandardPBRMaterialHeader::getAndTryBuildGPU()
	{
		if (!m_bAllAssetReady)
		{
			buildCache();
		}
		return m_runtimeMaterialcache;
	}
	const GPUStaticMeshStandardPBRMaterial& StandardPBRMaterialHeader::getGPUOnly() const
	{
		return m_runtimeMaterialcache;
	}
	StandardPBRTexturesHandle StandardPBRMaterialHeader::buildCache()
	{
		m_bAllAssetReady = true;
		m_runtimeMaterialcache = GPUStaticMeshStandardPBRMaterial::buildDeafult();

		GPUTexturesHandle handle;

		auto loadTexIfChange = [&](const UUID& in, std::shared_ptr<GPUImageAsset>& outId)
		{
			auto tex = TextureManager::get()->getImage(in);
			if (tex == nullptr)
			{
				auto texHeader = std::dynamic_pointer_cast<ImageAssetHeader>(AssetRegistryManager::get()->getHeaderMap().at(in));
				tex = TextureManager::get()->getOrCreateImage(texHeader);
			}
			outId = tex;
		};

		// Load all texture to GPU if render need it.
		loadTexIfChange(this->baseColorTexture, handle.baseColor);
		loadTexIfChange(this->normalTexture,    handle.normal);
		loadTexIfChange(this->specularTexture,  handle.specular);
		loadTexIfChange(this->aoTexture,        handle.occlusion);
		loadTexIfChange(this->emissiveTexture,  handle.emissive);

		auto getTexID = [&](const std::shared_ptr<GPUImageAsset>& in, uint32_t& outId)
		{
			m_bAllAssetReady &= in->isAssetReady();
			if (in && in->isAssetReady())
			{
				outId = in->getReadyAsset()->getBindlessIndex();
			}
		};

		// Get all texture id.
		getTexID(handle.baseColor, m_runtimeMaterialcache.baseColorId);
		getTexID(handle.normal,    m_runtimeMaterialcache.normalTexId);
		getTexID(handle.specular,  m_runtimeMaterialcache.specTexId);
		getTexID(handle.occlusion, m_runtimeMaterialcache.occlusionTexId);
		getTexID(handle.emissive,  m_runtimeMaterialcache.emissiveTexId);

		// Other parameters.
		m_runtimeMaterialcache.baseColorMul = this->baseColorMul;
		m_runtimeMaterialcache.baseColorAdd = this->baseColorAdd;
		m_runtimeMaterialcache.metalMul     = this->metalMul;
		m_runtimeMaterialcache.metalAdd     = this->metalAdd;
		m_runtimeMaterialcache.roughnessMul = this->roughnessMul;
		m_runtimeMaterialcache.roughnessAdd = this->roughnessAdd;
		m_runtimeMaterialcache.emissiveMul  = this->emissiveMul;
		m_runtimeMaterialcache.emissiveAdd  = this->emissiveAdd;
		m_runtimeMaterialcache.cutoff       = this->cutoff;
		m_runtimeMaterialcache.faceCut      = this->faceCut;

		return handle;
	}
}