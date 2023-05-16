#include <rhi/rhi.h>
#include "shader_struct.h"

namespace engine
{
	GPUMaterialStandardPBR buildDefaultGPUMaterialStandardPBR()
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
			.maxLod = 10000.0f,
			.unnormalizedCoordinates = VK_FALSE,
		};

		getContext()->getSamplerCache().createSampler(info, linearId);

		GPUMaterialStandardPBR result =
		{
			.baseColorId = getContext()->getEngineTextureWhite()->getBindlessIndex(),
			.baseColorSampler = linearId,
			.normalTexId = getContext()->getEngineTextureNormal()->getBindlessIndex(),
			.normalSampler = linearId,
			.specTexId = getContext()->getEngineTextureSpecular()->getBindlessIndex(),
			.specSampler = linearId,
			.occlusionTexId = getContext()->getEngineTextureWhite()->getBindlessIndex(),
			.occlusionSampler = linearId,
			.emissiveTexId = getContext()->getEngineTextureTranslucent()->getBindlessIndex(),
			.emissiveSampler = linearId,
		};

		result.cutoff = 0.5f;
		result.faceCut = 0.0f;
		result.baseColorMul = math::vec4{ 1.0f };
		result.baseColorAdd = math::vec4{ 0.0f };
		result.metalMul = 1.0f;
		result.metalAdd = 0.0f;
		result.roughnessMul = 1.0f;
		result.roughnessAdd = 0.0f;
		result.emissiveMul = math::vec4{ 1.0f };
		result.emissiveAdd = math::vec4{ 0.0f };

		return result;
	}

	GPUMaterialStandardPBR GPUMaterialStandardPBR::getDefault()
	{
		static GPUMaterialStandardPBR defaultMaterial = buildDefaultGPUMaterialStandardPBR();
		return defaultMaterial;
	}

	void AtmosphereConfig::resetAtmosphere()
	{
		// Unit in kilometers.

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
		constexpr float kEarthTopRadius = kEarthBottomRadius + 60.0f;
		constexpr float kEarthRayleighScaleHeight = 8.0f;
		constexpr float kEarthMieScaleHeight = 1.2f;

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

		static const math::vec3 rayleighScattering = { 0.005802f, 0.013558f, 0.033100f };

		inAtmosphere.rayleighScatteringColor = math::vec3(0.175287f, 0.409607f, 1.0f);// math::normalize(rayleighScattering);
		inAtmosphere.rayleighScatterLength = 0.0331f;//  math::length(rayleighScattering);

		static const math::vec3 mieScattering = { 0.003996f, 0.003996f, 0.003996f };

		inAtmosphere.mieScatteringColor = math::vec3(1.0f, 1.0f, 1.0f); // math::normalize(mieScattering);
		inAtmosphere.mieScatteringLength = 0.003996f;// math::length(mieScattering);

		static const math::vec3 mieExtinction = { 0.004440f, 0.004440f, 0.004440f };
		static const math::vec3 mieAbs = (mieExtinction - mieScattering);

		inAtmosphere.mieAbsLength = 0.000444f;// math::length(mieAbs);
		inAtmosphere.mieAbsColor = math::vec3(1.0f, 1.0f, 1.0f);// mieAbs / inAtmosphere.mieAbsLength;

		inAtmosphere.miePhaseFunctionG = 0.8f;

		static const math::vec3 absorptionExtinction = { 0.000650f, 0.001881f, 0.000085f };

		inAtmosphere.absorptionColor = math::vec3(0.345561f, 1.0f, 0.045189f);// math::normalize(absorptionExtinction);
		inAtmosphere.absorptionLength = 0.001881f; // math::length(absorptionExtinction);

		inAtmosphere.multipleScatteringFactor = 1.0f;
		inAtmosphere.atmospherePreExposure = 1.0f;
		inAtmosphere.viewRayMarchMaxSPP = 31; // [2, 31]
		inAtmosphere.viewRayMarchMinSPP = 14; // [1, 31]
	}

	void AtmosphereConfig::resetCloud()
	{
		auto& inAtmosphere = *this;

		inAtmosphere.cloudAreaStartHeight = inAtmosphere.bottomRadius + 1.5f;// 3.3f; // km
		inAtmosphere.cloudAreaThickness = 3.5f;// 3.4f; // km

		inAtmosphere.cloudWeatherUVScale = { 0.02f, 0.02f }; // vec2(0.005)
		inAtmosphere.cloudCoverage = 0.5f; // 0.50
		inAtmosphere.cloudDensity = 1.0f;  // 0.10

		inAtmosphere.cloudShadingSunLightScale = 1.0f; // 5.0
		inAtmosphere.cloudFogFade = 1.0f; // 0.005
		inAtmosphere.cloudMaxTraceingDistance = 50.0f; // 50.0 km
		inAtmosphere.cloudTracingStartMaxDistance = 350.0f; // 350.0 km

		inAtmosphere.cloudDirection = glm::normalize(glm::vec3{ 0.8f, 0.0f, 0.4f });
		inAtmosphere.cloudSpeed = 0.05f;

		inAtmosphere.cloudMultiScatterExtinction = 0.175f;
		inAtmosphere.cloudMultiScatterScatter = 1.0f;

		inAtmosphere.cloudBasicNoiseScale  = 0.3f;
		inAtmosphere.cloudDetailNoiseScale = 0.6f;

		inAtmosphere.cloudAlbedo = { 1.0f , 1.0f, 1.0f };
		inAtmosphere.cloudPhaseForward = 0.8f;

		inAtmosphere.cloudPhaseBackward = -0.5f;
		inAtmosphere.cloudPhaseMixFactor = 0.5f;
		inAtmosphere.cloudPowderScale = 1.0f;
		inAtmosphere.cloudPowderPow = 1.0f;

		inAtmosphere.cloudLightStepMul = 1.5f;
		inAtmosphere.cloudLightBasicStep = 0.025f;
		inAtmosphere.cloudLightStepNum = 12;
		inAtmosphere.cloudEnableGroundContribution = 1;


		inAtmosphere.cloudMarchingStepNum = 128;
		inAtmosphere.cloudSunLitMapOctave = 5;
		inAtmosphere.cloudNoiseScale = 0.6f;

		inAtmosphere.cloudGodRay = 1;
		inAtmosphere.cloudGodRayScale = 50.0f;
	}

}