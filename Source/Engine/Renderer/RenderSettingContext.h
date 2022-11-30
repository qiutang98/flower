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

	struct RenderSetting
	{
		IBLLightingContext ibl;
		EarthAtmosphereContext earthAtmosphere;

		RHI::DisplayMode displayMode = RHI::DisplayMode::DISPLAYMODE_SDR;

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