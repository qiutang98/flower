#pragma once
#include "RendererCommon.h"
#include "Parameters.h"

namespace Flower
{
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

	struct EarthAtmosphereContext
	{
		EarthAtmosphereContext()
		{
			reset();
		}

		void release()
		{

		}

		void reset()
		{
			earthAtmosphere.reset();

			rayleighDensityProfile.layers[0] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
			rayleighDensityProfile.layers[1] = { 0.0f, 1.0f, -1.0f / kEarthRayleighScaleHeight, 0.0f, 0.0f };
			mieDensityProfile.layers[0] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
			mieDensityProfile.layers[1] = { 0.0f, 1.0f, -1.0f / kEarthMieScaleHeight, 0.0f, 0.0f };
			absorptionDensityProfile.layers[0] = { 25.0f, 0.0f, 0.0f,  1.0f / 15.0f, -2.0f / 3.0f };
			absorptionDensityProfile.layers[1] = { 0.0f, 0.0f, 0.0f, -1.0f / 15.0f,  8.0f / 3.0f };

			cloudBottomAltitude = 5.0f; // km
			cloudHeight = 10.0f; // km

			earthAtmosphere.cloudAreaStartHeight = earthAtmosphere.bottomRadius + cloudBottomAltitude;
			earthAtmosphere.cloudAreaThickness = earthAtmosphere.bottomRadius + cloudHeight;

			{
				auto rayleighScattering = earthAtmosphere.rayleighScattering;
				auto mieScattering = earthAtmosphere.mieScattering;
				auto mieAbsorption = glm::max(glm::vec3(0.0f), earthAtmosphere.mieExtinction - earthAtmosphere.mieScattering);

				mieScatteringLength = glm::length(mieScattering);
				mieScatteringColor = mieScatteringLength == 0.0f ? glm::vec3(0.0f) : glm::normalize(mieScattering);
				mieAbsLength = glm::length(mieAbsorption);
				mieAbsColor = mieAbsLength == 0.0f ? glm::vec3(0.0f) : glm::normalize(mieAbsorption);
				rayleighScatteringLength = glm::length(rayleighScattering);
				rayleighScatteringColor = rayleighScatteringLength == 0.0f ? glm::vec3(0.0f) : glm::normalize(rayleighScattering);
				atmosphereHeight = earthAtmosphere.topRadius - earthAtmosphere.bottomRadius;
				mieScaleHeight = -1.0f / (mieDensityProfile.layers[1].expScale);
				rayleighScaleHeight = -1.0f / rayleighDensityProfile.layers[1].expScale;
				absorptionLength = glm::length(earthAtmosphere.absorptionExtinction);
				absorptionColor = absorptionLength == 0.0f ? glm::vec3(0.0f) : glm::normalize(earthAtmosphere.absorptionExtinction);
			}
		}

		EarthAtmosphere earthAtmosphere{};
		float mieScatteringLength;
		glm::vec3 mieScatteringColor;
		float mieAbsLength;
		glm::vec3 mieAbsColor;
		float mieScaleHeight;
		float rayleighScatteringLength;
		glm::vec3 rayleighScatteringColor;
		float rayleighScaleHeight;
		float absorptionLength;
		glm::vec3 absorptionColor;
		float atmosphereHeight;
		glm::vec3 groundAbledo = { 0.3f, 0.3f, 0.3f };

		DensityProfile rayleighDensityProfile{};
		DensityProfile mieDensityProfile{};
		DensityProfile absorptionDensityProfile{};

		float cloudBottomAltitude;
		float cloudHeight;
	};

	class IBLLightingContext
	{
	private:
		bool m_bIBLDirty = false;

	public:
		float intensity = 1.0f;

		bool bEnableIBLLight = false;

		// TODO: When ibl bake ready, should release this owner.
		std::shared_ptr<GPUImageAsset> hdrSrc = nullptr;

		bool iblEnable() const;

		void reset()
		{
			bEnableIBLLight = false;
			hdrSrc = nullptr;
		}

		void release()
		{
			hdrSrc = nullptr;
		}

		void setDirty(bool bState)
		{
			m_bIBLDirty = bState;
		}

		bool needRebuild() const;
			
	};

	enum class LummapCase
	{
		Campfire,
		Lamp,
		P3_1000_nits,
		Rec2020_1000_nits,
		Rec709_5000_nits,
	};

	struct RenderSetting
	{
		IBLLightingContext ibl;
		EarthAtmosphereContext earthAtmosphere;

		float bloomIntensity = 1.0f;
		float bloomRadius = 0.85f;

		float bloomThreshold = 0.4f;
		float bloomThresholdSoft = 0.6f;

		float AUTOEXPOSURE_lowPercent = 0.5f;
		float AUTOEXPOSURE_highPercent = 0.95f;

		float AUTOEXPOSURE_minBrightness = -5.0f;
		float AUTOEXPOSURE_maxBrightness = 7.0f;

		float AUTOEXPOSURE_speedDown = 1.0f;
		float AUTOEXPOSURE_speedUp = 2.0f;

		float AUTOEXPOSURE_exposureCompensation = 0.0f;

		int GTAO_sliceNum = 2;
		int GTAO_stepNum = 8;
		float GTAO_radius = 0.8f;
		float GTAO_thickness = 0.8f;

		float GTAO_Power = 2.0f;
		float GTAO_Intensity = 1.0f;

		RHI::DisplayMode displayMode = RHI::DisplayMode::DISPLAYMODE_SDR;
		LummapCase lummapCase = LummapCase::Campfire;

		float tonemmaper_s = 4.0f; // scale 
		float tonemapper_P = 1000.0f;  // Max brightness.
		float tonemapper_a = 1.2f;  // contrast
		float tonemapper_m = 0.22f; // linear section start
		float tonemapper_l = 0.4f;  // linear section length
		float tonemapper_c = 1.33f; // black
		float tonemapper_b = 0.0f;  // pedestal

		void reset()
		{
			ibl.reset();
			earthAtmosphere.reset();
		}

		void release()
		{
			ibl.release();
			earthAtmosphere.release();
		}
	};

	using RenderSettingManager = Singleton<RenderSetting>;
}