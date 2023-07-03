#include "asset_material.h"
#include "asset_system.h"
#include "asset_texture.h"

namespace engine
{
	StandardPBRMaterial::StandardPBRMaterial()
	{
		m_runtimeMaterialcache = GPUMaterialStandardPBR::getDefault();
	}

	const GPUMaterialStandardPBR& StandardPBRMaterial::getAndTryBuildGPU()
    {
        if (!m_bAllAssetReady)
        {
            buildCache();
        }
        return m_runtimeMaterialcache;
    }

    const GPUMaterialStandardPBR& StandardPBRMaterial::getGPUOnly() const
    {
        return m_runtimeMaterialcache;
    }

    StandardPBRMaterial::GPUTexturesHandle StandardPBRMaterial::buildCache()
    {
        m_bAllAssetReady = true;
        m_runtimeMaterialcache = GPUMaterialStandardPBR::getDefault();

        GPUTexturesHandle handle;

		// Load all texture to GPU if render need it.
		handle.baseColor = getContext()->getOrCreateTextureAsset(this->baseColorTexture);
		handle.normal = getContext()->getOrCreateTextureAsset(this->normalTexture);
		handle.specular = getContext()->getOrCreateTextureAsset(this->specularTexture);
		handle.occlusion = getContext()->getOrCreateTextureAsset(this->aoTexture);
		handle.emissive = getContext()->getOrCreateTextureAsset(this->emissiveTexture);

		auto getTexID = [&](const std::shared_ptr<GPUImageAsset>& in, uint32_t& outId)
		{
			m_bAllAssetReady &= in->isAssetReady();
			if (in && in->isAssetReady())
			{
				outId = in->getReadyAsset<GPUImageAsset>()->getBindlessIndex();
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

		m_runtimeMaterialcache.shadingModel = shadingModelConvert(this->shadingModelType);

		return handle;
    }

	bool AssetMaterial::drawAssetConfig()
	{
		return AssetInterface::drawAssetConfig();
	}

	bool StandardPBRMaterial::drawAssetConfig()
	{
		bool bDataChange = false;
		bDataChange |= AssetMaterial::drawAssetConfig();


		bDataChange |= ui::drawShadingModelSelect(this->shadingModelType);

		bDataChange |= ui::drawVector4("BaseColor Mul", this->baseColorMul, glm::vec4{ 1.0f }, ImGui::GetFontSize() * 6.0f);
		bDataChange |= ui::drawVector4("BaseColor Add", this->baseColorAdd, glm::vec4{ 0.0f }, ImGui::GetFontSize() * 6.0f);

		bDataChange |= ui::drawVector4("Emissive Mul", this->emissiveMul, glm::vec4{ 1.0f }, ImGui::GetFontSize() * 6.0f);
		bDataChange |= ui::drawVector4("Emissive Add", this->emissiveAdd, glm::vec4{ 0.0f }, ImGui::GetFontSize() * 6.0f);

		bDataChange |= ui::drawFloat("Metallic Mul", this->metalMul, 1.0f);
		bDataChange |= ui::drawFloat("Metallic Add", this->metalAdd, 0.0f);
		bDataChange |= ui::drawFloat("Roughness Mul", this->roughnessMul, 1.0f);
		bDataChange |= ui::drawFloat("Roughness Add", this->roughnessAdd, 0.0f);

		if (bDataChange)
		{
			this->buildCache();
			setDirty();
		}

		return bDataChange;
	}

	bool StandardPBRMaterial::saveActionImpl()
	{
		saveAssetMeta<StandardPBRMaterial>(*this, getSavePath(), this->getSuffix(), false);
		return true;
	}


}