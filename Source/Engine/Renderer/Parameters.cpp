#include "Pch.h"
#include "Parameters.h"
#include "../AssetSystem/TextureManager.h"
#include "../AssetSystem/MeshManager.h"
#include "../AssetSystem/MaterialManager.h"
#include "../Scene/Component/SunSky.h"
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



	// An atmosphere layer of width 'width', and whose density is defined as
	//   'exp_term' * exp('exp_scale' * h) + 'linear_term' * h + 'constant_term',
	// clamped to [0,1], and where h is the altitude.
	struct DensityProfileLayer
	{
		float width;
		float expTerm;
		float expScale;
		float linearTerm;
		float constantTerm;
	};

	// An atmosphere density profile made of several layers on top of each other
	// (from bottom to top). The width of the last layer is ignored, i.e. it always
	// extend to the top atmosphere boundary. The profile values vary between 0
	// (null density) to 1 (maximum density).
	struct DensityProfile
	{
		DensityProfileLayer layers[2];
	};

	// Values shown here are the result of integration over wavelength power spectrum integrated with paricular function.
	// Refer to https://github.com/ebruneton/precomputed_atmospheric_scattering for details.
	constexpr float kEarthBottomRadius = 6360.0f;

	// 100km atmosphere radius, less edge visible and it contain 99.99% of the atmosphere medium 
	// https://en.wikipedia.org/wiki/K%C3%A1rm%C3%A1n_line
	constexpr float kEarthTopRadius = kEarthBottomRadius + 80.0f;
	constexpr float kEarthRayleighScaleHeight = 8.0f;
	constexpr float kEarthMieScaleHeight = 1.2f;

	// All units in kilometers
	void EarthAtmosphere::resetAtmosphere()
	{
		auto& inAtmosphere = *this;

		inAtmosphere.bottomRadius = kEarthBottomRadius;
		inAtmosphere.topRadius = kEarthTopRadius;
		inAtmosphere.groundAlbedo = { 0.3f, 0.3f, 0.3f };

		DensityProfile rayleighDensityProfile{};
		DensityProfile mieDensityProfile{};
		DensityProfile absorptionDensityProfile{};

		rayleighDensityProfile.layers[0] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
		rayleighDensityProfile.layers[1] = { 0.0f, 1.0f, -1.0f / kEarthRayleighScaleHeight, 0.0f, 0.0f };
		mieDensityProfile.layers[0] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
		mieDensityProfile.layers[1] = { 0.0f, 1.0f, -1.0f / kEarthMieScaleHeight, 0.0f, 0.0f };
		absorptionDensityProfile.layers[0] = { 25.0f, 0.0f, 0.0f,  1.0f / 15.0f, -2.0f / 3.0f };
		absorptionDensityProfile.layers[1] = { 0.0f, 0.0f, 0.0f, -1.0f / 15.0f,  8.0f / 3.0f };

		memcpy(inAtmosphere.rayleighDensity, &rayleighDensityProfile, sizeof(rayleighDensityProfile));
		memcpy(inAtmosphere.mieDensity, &mieDensityProfile, sizeof(mieDensityProfile));
		memcpy(inAtmosphere.absorptionDensity, &absorptionDensityProfile, sizeof(absorptionDensityProfile));

		static const glm::vec3 rayleighScattering = { 0.005802f, 0.013558f, 0.033100f };

		inAtmosphere.rayleighScatteringColor = glm::normalize(rayleighScattering);		// 1/km
		inAtmosphere.rayleighScatterLength = glm::length(rayleighScattering);

		static const glm::vec3 mieScattering = { 0.003996f, 0.003996f, 0.003996f };

		inAtmosphere.mieScatteringColor = glm::normalize(mieScattering);	 // 1/km
		inAtmosphere.mieScatteringLength = glm::length(mieScattering);

		static const glm::vec3 mieExtinction = { 0.004440f, 0.004440f, 0.004440f };
		static const glm::vec3 mieAbs = (mieExtinction - mieScattering);

		inAtmosphere.mieAbsLength = glm::length(mieAbs);
		inAtmosphere.mieAbsColor = mieAbs / inAtmosphere.mieAbsLength;

		inAtmosphere.miePhaseFunctionG = 0.8f;

		static const glm::vec3 absorptionExtinction = { 0.000650f, 0.001881f, 0.000085f };

		inAtmosphere.absorptionColor = glm::normalize(absorptionExtinction);
		inAtmosphere.absorptionLength = glm::length(absorptionExtinction);


		inAtmosphere.multipleScatteringFactor = 1.0f;
		inAtmosphere.atmospherePreExposure = 1.0f;
		inAtmosphere.viewRayMarchMaxSPP = 31; // [2, 31]
		inAtmosphere.viewRayMarchMinSPP = 14; // [1, 31]
	}
	
	void EarthAtmosphere::resetCloud()
	{
		auto& inAtmosphere = *this;

		inAtmosphere.cloudAreaStartHeight = inAtmosphere.bottomRadius + 1.0f; // km
		inAtmosphere.cloudAreaThickness = 10.0f; // km

		inAtmosphere.cloudWeatherUVScale = { 0.005f, 0.005f }; // vec2(0.005)
		inAtmosphere.cloudCoverage = 0.5f; // 0.50
		inAtmosphere.cloudDensity  = 0.1f;  // 0.10

		inAtmosphere.cloudShadingSunLightScale = 1.0f; // 5.0
		inAtmosphere.cloudFogFade = 0.005f; // 0.005
		inAtmosphere.cloudMaxTraceingDistance = 50.0f; // 50.0 km
		inAtmosphere.cloudTracingStartMaxDistance = 350.0f; // 350.0 km

		inAtmosphere.cloudDirection = glm::normalize(glm::vec3{ 0.8, 0.2, 0.4 });
		inAtmosphere.cloudSpeed = 0.1f;
	}

}


