#include "Pch.h"
#include "Parameters.h"
#include "../AssetSystem/TextureManager.h"
#include "../AssetSystem/MeshManager.h"
#include "../AssetSystem/MaterialManager.h"
#include "../Scene/Component/DirectionalLight.h"
#include "RenderSettingContext.h"

namespace Flower
{
	GPUStaticMeshStandardPBRMaterial GPUStaticMeshStandardPBRMaterial::buildDeafult()
	{
		uint32_t linearId;
		VkSamplerCreateInfo info
		{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.magFilter = VK_FILTER_LINEAR,
			.minFilter = VK_FILTER_LINEAR,
			.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
			.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.mipLodBias = 0.0f,
			.anisotropyEnable = VK_FALSE,
			.compareEnable = VK_FALSE,
			.minLod = 0.0f,
			.maxLod =  10000.0f,
			.unnormalizedCoordinates = VK_FALSE,
		};

		RHI::SamplerManager->createSampler(info, linearId);
		return GPUStaticMeshStandardPBRMaterial
		{
			.baseColorId = EngineTextures::GGreyTextureId,
			.baseColorSampler = linearId,
			.normalTexId = EngineTextures::GNormalTextureId,
			.normalSampler = linearId,
			.specTexId = EngineTextures::GDefaultSpecularId,
			.specSampler = linearId,
			.occlusionTexId = EngineTextures::GWhiteTextureId,
			.occlusionSampler = linearId,
			.emissiveTexId = EngineTextures::GTranslucentTextureId,
			.emissiveSampler = linearId,
		};
	}

	bool CPUStaticMeshStandardPBRMaterial::buildWithMaterialUUID(UUID materialId)
	{
		// Unvalid materials.
		if (!AssetRegistryManager::get()->getHeaderMap().contains(materialId))
		{
			return true; // always fallback.
		}

		bool bAllAssetReady = true;

		auto header = std::dynamic_pointer_cast<StandardPBRMaterialHeader>(AssetRegistryManager::get()->getHeaderMap().at(materialId));

		auto loadTex = [&](const UUID& in, std::shared_ptr<GPUImageAsset>& outId)
		{
			auto tex = TextureManager::get()->getImage(in);
			if (tex == nullptr)
			{
				auto assetHeader = std::dynamic_pointer_cast<ImageAssetHeader>(AssetRegistryManager::get()->getHeaderMap().at(in));
				tex = TextureManager::get()->getOrCreateImage(assetHeader);
			}

			bAllAssetReady &= tex->isAssetReady();
			outId = tex;
		};

		loadTex(header->baseColorTexture, baseColor);
		loadTex(header->normalTexture, normal);
		loadTex(header->specularTexture, specular);
		loadTex(header->aoTexture, occlusion);
		loadTex(header->emissiveTexture, emissive);

		return bAllAssetReady;
	}

	GPUStaticMeshStandardPBRMaterial CPUStaticMeshStandardPBRMaterial::buildGPU()
	{
		// Build fallback first.
		GPUStaticMeshStandardPBRMaterial result = GPUStaticMeshStandardPBRMaterial::buildDeafult();

		auto loadTex = [&](const std::shared_ptr<GPUImageAsset>& in, uint32_t& outId)
		{
			if (in)
			{
				if (in->isAssetReady())
				{
					outId = in->getReadyAsset()->getBindlessIndex();
				}
			}
		};

		loadTex(baseColor, result.baseColorId);
		loadTex(normal, result.normalTexId);
		loadTex(specular, result.specTexId);
		loadTex(occlusion, result.occlusionTexId);
		loadTex(emissive, result.emissiveTexId);

		return result;
	}

	// All units in kilometers
	void EarthAtmosphere::reset()
	{
		// Earth
		this->bottomRadius = kEarthBottomRadius;
		this->topRadius = kEarthTopRadius;
		this->groundAlbedo = { 0.3f, 0.3f, 0.3f };

		DensityProfile rayleighDensityProfile{};
		DensityProfile mieDensityProfile{};
		DensityProfile absorptionDensityProfile{};

		rayleighDensityProfile.layers[0] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
		rayleighDensityProfile.layers[1] = { 0.0f, 1.0f, -1.0f / kEarthRayleighScaleHeight, 0.0f, 0.0f };
		mieDensityProfile.layers[0] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
		mieDensityProfile.layers[1] = { 0.0f, 1.0f, -1.0f / kEarthMieScaleHeight, 0.0f, 0.0f };
		absorptionDensityProfile.layers[0] = { 25.0f, 0.0f, 0.0f,  1.0f / 15.0f, -2.0f / 3.0f };
		absorptionDensityProfile.layers[1] = { 0.0f, 0.0f, 0.0f, -1.0f / 15.0f,  8.0f / 3.0f };

		memcpy(this->rayleighDensity, &rayleighDensityProfile, sizeof(rayleighDensityProfile));
		memcpy(this->mieDensity, &mieDensityProfile, sizeof(mieDensityProfile));
		memcpy(this->absorptionDensity, &absorptionDensityProfile, sizeof(absorptionDensityProfile));

		this->rayleighScattering = { 0.005802f, 0.013558f, 0.033100f };		// 1/km

		// Mie scattering
		this->mieScattering = { 0.003996f, 0.003996f, 0.003996f };			// 1/km
		this->mieExtinction = { 0.004440f, 0.004440f, 0.004440f };			// 1/km
		this->miePhaseFunctionG = 0.8f;

		// Ozone absorption
		this->absorptionExtinction = { 0.000650f, 0.001881f, 0.000085f };	// 1/km

		this->multipleScatteringFactor = 1.0f;
		this->atmospherePreExposure = 1.0f;
		this->viewRayMarchMinSPP = 14;  // [1, 30]
		this->viewRayMarchMaxSPP = 31; // [2, 31]
	}
	
}


