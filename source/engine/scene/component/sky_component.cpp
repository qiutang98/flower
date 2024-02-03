#include "sky_component.h"


#include <iconFontcppHeaders/IconsFontAwesome6.h>
#include <editor/widgets/content.h>
#include <editor/editor.h>
#include <renderer/render_scene.h>

namespace engine
{
	// Values shown here are the result of integration over wavelength power spectrum integrated with paricular function.
	// Refer to https://github.com/ebruneton/precomputed_atmospheric_scattering for details.
	constexpr float kEarthBottomRadius = 6360.0f;

	AtmosphereParametersInputs defaultAtmosphereParameters()
	{
		AtmosphereParametersInputs inAtmosphere{};

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



		// 100km atmosphere radius, less edge visible and it contain 99.99% of the atmosphere medium 
		// https://en.wikipedia.org/wiki/K%C3%A1rm%C3%A1n_line
		constexpr float kEarthTopRadius = kEarthBottomRadius + 60.0f;
		constexpr float kEarthRayleighScaleHeight = 8.0f;
		constexpr float kEarthMieScaleHeight = 1.2f;

		inAtmosphere.bottomRadius = kEarthBottomRadius;
		inAtmosphere.topRadius    = kEarthTopRadius;
		inAtmosphere.groundAlbedo = { 0.3f, 0.3f, 0.3f };

		DensityProfile rayleighDensityProfile   { };
		DensityProfile mieDensityProfile        { };
		DensityProfile absorptionDensityProfile { };

		rayleighDensityProfile.layers[0]   = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
		rayleighDensityProfile.layers[1]   = { 0.0f, 1.0f, -1.0f / kEarthRayleighScaleHeight, 0.0f, 0.0f };
		mieDensityProfile.layers[0]        = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
		mieDensityProfile.layers[1]        = { 0.0f, 1.0f, -1.0f / kEarthMieScaleHeight, 0.0f, 0.0f };
		absorptionDensityProfile.layers[0] = { 25.0f, 0.0f, 0.0f,  1.0f / 15.0f, -2.0f / 3.0f };
		absorptionDensityProfile.layers[1] = { 0.0f, 0.0f, 0.0f, -1.0f / 15.0f,  8.0f / 3.0f };

		memcpy(&inAtmosphere.rayleighDensity[0].x,   &rayleighDensityProfile,   sizeof(rayleighDensityProfile));
		memcpy(&inAtmosphere.mieDensity[0].x,        &mieDensityProfile,        sizeof(mieDensityProfile));
		memcpy(&inAtmosphere.absorptionDensity[0].x, &absorptionDensityProfile, sizeof(absorptionDensityProfile));

		static const math::vec3 rayleighScattering = { 0.005802f, 0.013558f, 0.033100f };
		inAtmosphere.rayleighScatteringColor = math::normalize(rayleighScattering);
		inAtmosphere.rayleighScatterLength = math::length(rayleighScattering);


		static const math::vec3 mieScattering = { 0.003996f, 0.003996f, 0.003996f };
		inAtmosphere.mieScatteringColor = math::normalize(mieScattering);
		inAtmosphere.mieScatteringLength = math::length(mieScattering);

		static const math::vec3 mieExtinction = { 0.004440f, 0.004440f, 0.004440f };
		static const math::vec3 mieAbs = (mieExtinction - mieScattering);

		inAtmosphere.mieAbsLength = math::length(mieAbs);
		inAtmosphere.mieAbsColor = math::normalize(mieAbs);

		inAtmosphere.miePhaseFunctionG = 0.8f;

		static const math::vec3 absorptionExtinction = { 0.000650f, 0.001881f, 0.000085f };

		inAtmosphere.absorptionColor = math::normalize(absorptionExtinction);
		inAtmosphere.absorptionLength = math::length(absorptionExtinction);

		// sebh's multi scatter factor.
		inAtmosphere.multipleScatteringFactor = 1.0f;

		inAtmosphere.viewRayMarchMaxSPP = 31; // [2, 31]
		inAtmosphere.viewRayMarchMinSPP = 14; // [1, 31]

		return inAtmosphere;
	}

	CloudParametersInputs defaultCloudParameters(float earthRadius)
	{
		CloudParametersInputs inputs { };

		float validRadius = earthRadius <= 0.0f ? kEarthBottomRadius : earthRadius;

		inputs.cloudAreaStartHeight = validRadius + 3.0f;// Cumulonimbus in GuangZhou usually at 500 - 800m.
		inputs.cloudAreaThickness = 8.0f;// 3.4f; // km

		inputs.cloudWeatherUVScale = { 0.02f, 0.02f }; // vec2(0.005)
		inputs.cloudCoverage = 0.5f; // 0.50
		inputs.cloudDensity  = 1.0f;  // 0.10

		inputs.cloudShadingSunLightScale = 1.0f; // 5.0
		inputs.cloudFogFade = 1.0f; // 0.005
		inputs.cloudMaxTraceingDistance = 100.0f; // 50.0 km
		inputs.cloudTracingStartMaxDistance = 350.0f; // 350.0 km

		inputs.cloudDirection = glm::normalize(glm::vec3{ 0.8f, 0.0f, 0.4f });
		inputs.cloudSpeed = 0.05f;

		inputs.cloudMultiScatterExtinction = 0.175f;
		inputs.cloudMultiScatterScatter = 1.0f;

		inputs.cloudBasicNoiseScale = 0.3f;
		inputs.cloudDetailNoiseScale = 0.6f;

		inputs.cloudAlbedo = { 1.0f , 1.0f, 1.0f };
		inputs.cloudPhaseForward = 0.5f;

		inputs.cloudPhaseBackward = -0.5f;
		inputs.cloudPhaseMixFactor = 0.5f;
		inputs.cloudPowderScale = 1.0f;
		inputs.cloudPowderPow = 1.0f;

		inputs.cloudLightStepMul = 1.5f;
		inputs.cloudLightBasicStep = 15.0f;
		inputs.cloudLightStepNum = 12;
		inputs.cloudEnableGroundContribution = 1;


		inputs.cloudMarchingStepNum = 128;
		inputs.cloudSunLitMapOctave = 5;
		inputs.cloudNoiseScale = 0.01f;

		inputs.cloudGodRay = 0;
		inputs.cloudGodRayScale = 50.0f;

		inputs.cloudShadowExtent = 10.0f;

		inputs.cloudAmbientScale = 1.0f;

		return inputs;
	}

	bool uiDrawCloudParameters(CloudParametersInputs& inout, const AtmosphereParametersInputs& atmosphere)
	{
		bool bResult = false;

		if (ImGui::CollapsingHeader("Cloud Config"))
		{
			CloudParametersInputs copyValue = inout;

			ImGui::Spacing();
			if (ImGui::Button("Reset Cloud"))
			{
				copyValue = defaultCloudParameters(atmosphere.bottomRadius);
			}
			ImGui::Spacing();

		
			float cloudBottomAltitude = copyValue.cloudAreaStartHeight - atmosphere.bottomRadius;

			ImGui::PushItemWidth(200.0f);
			ImGui::DragFloat("start height(km)", &cloudBottomAltitude, 0.1f, 0.0f, 20.0f);
			ImGui::DragFloat("thickness(km)", &copyValue.cloudAreaThickness, 0.1f, 0.1f, 20.0f);


			ImGui::DragFloat("coverage", &copyValue.cloudCoverage, 0.01f, 0.0f, 1.0f);
			ImGui::DragFloat("density", &copyValue.cloudDensity, 0.1f, 0.0f, 5.0f);
			ImGui::DragFloat("ambient scale", &copyValue.cloudAmbientScale, 0.01f, 0.01f, 1.0f);
			ImGui::DragFloat("sun light scale", &copyValue.cloudShadingSunLightScale, 0.1f, 0.1f, 10.0f);
			ImGui::DragFloat3("wind dir", &copyValue.cloudDirection.x, 0.01f, 0.0f);
			ImGui::DragFloat("wind speed", &copyValue.cloudSpeed, 0.1f, 0.0f, 1.0f);
			ImGui::DragFloat("multi scatter", &copyValue.cloudMultiScatterScatter, 0.1f, 0.0f, 1.0f);
			ImGui::DragFloat("multi extinction", &copyValue.cloudMultiScatterExtinction, 0.1f, 0.0f, 1.0f);
			ImGui::DragFloat("phase forward", &copyValue.cloudPhaseForward, 0.01f, 0.01f, 0.99f);
			ImGui::DragFloat("phase backward", &copyValue.cloudPhaseBackward, 0.01f, -0.99f, -0.01f);
			ImGui::DragFloat("phase mix factor", &copyValue.cloudPhaseMixFactor, 0.01f, 0.01f, 0.99f);

			ImGui::ColorEdit3("cloud albedo", &copyValue.cloudAlbedo.x);
		//  ImGui::DragFloat("light step mul", &copyValue.cloudLightStepMul, 0.01f, 1.01f, 1.5f);
			ImGui::DragFloat("light shadow len", &copyValue.cloudLightBasicStep, 0.01f, 1.0f, 30.0f);
			ImGui::DragInt("light step num", &copyValue.cloudLightStepNum, 1, 6, 24);

			ImGui::DragInt("light marching step", &copyValue.cloudMarchingStepNum, 1, 24, 512);
#if 0
			ImGui::DragInt("noise octaves", &copyValue.cloudSunLitMapOctave, 1, 2, 8);

#endif
			ImGui::DragFloat("ground mix sky", &copyValue.cloudNoiseScale, 0.01f, 0.01f, 5.0f);

			ImGui::DragFloat("max tracing distance", &copyValue.cloudMaxTraceingDistance, 1.0f, 10.0f, 100.0f);
			ImGui::DragFloat("max tracing start distance", &copyValue.cloudTracingStartMaxDistance, 1.0f, 300.0f, 500.0f);

			ImGui::DragFloat2("Wether UV scale", &copyValue.cloudWeatherUVScale.x, 0.0f, 0.01f);
			ImGui::DragFloat("basic noise scale", &copyValue.cloudBasicNoiseScale, 0.01f, 0.0f, 1.0f);
			ImGui::DragFloat("detail noise scale", &copyValue.cloudDetailNoiseScale, 0.01f, 0.0f, 1.0f);

			ImGui::DragFloat("shadow extent(km)", &copyValue.cloudShadowExtent, 0.1f, 1.0f, 50.0f);
			bool bEnableGroundContribution = copyValue.cloudEnableGroundContribution != 0;
			ImGui::Checkbox("enable ambient contribution", &bEnableGroundContribution);
			copyValue.cloudEnableGroundContribution = bEnableGroundContribution ? 1 : 0;
			ImGui::DragFloat("ground contribution", &copyValue.cloudFogFade, 0.01f, 0.0f, 1.0f);
			ImGui::DragFloat("cloud powder scale", &copyValue.cloudPowderScale, 0.1f, 0.01f, 100.0f);
			ImGui::DragFloat("cloud powder pow", &copyValue.cloudPowderPow, 0.1f, 0.01f, 10.0f);

			bool bEnableGodRay = copyValue.cloudGodRay != 0;
			ImGui::Checkbox("enable cloud godray", &bEnableGodRay);
			copyValue.cloudGodRay = bEnableGodRay ? 1 : 0;

			ImGui::DragFloat("cloud godray scale", &copyValue.cloudGodRayScale, 1.0f, 1.0f, 100.0f);


			ImGui::PopItemWidth();
			ImGui::Spacing();

			copyValue.cloudAreaStartHeight = cloudBottomAltitude + atmosphere.bottomRadius;

			if (copyValue != inout)
			{
				inout = copyValue;
				bResult = true;
			}
		}

		return bResult;
	}

	bool uiDrawAtmosphereParameters(AtmosphereParametersInputs& inout, CloudParametersInputs& inoutCloud)
	{
		bool bResult = false;

		if (ImGui::CollapsingHeader("Atmosphere Config"))
		{
			AtmosphereParametersInputs copyValue = inout;

			ImGui::Spacing();
			if (ImGui::Button("Reset Atmosphere"))
			{
				copyValue = defaultAtmosphereParameters();
			}
			ImGui::Spacing();

			float atmosphereHeight = copyValue.topRadius - copyValue.bottomRadius;

			ui::beginGroupPanel("Base");
			ImGui::PushItemWidth(100.0f);
			{
				ImGui::DragFloat("Mie Phase",     &copyValue.miePhaseFunctionG, 0.0f, 0.01f, 0.99f);
				ImGui::DragFloat("Multi Scatter", &copyValue.multipleScatteringFactor, 0.01f, 0.01f, 1.0f, "%.6f");
				ImGui::SliderInt("Min SPP", &copyValue.viewRayMarchMinSPP, 1, 30);
				ImGui::SliderInt("Max SPP", &copyValue.viewRayMarchMaxSPP, 2, 31);

				float oldBottomRadius = copyValue.bottomRadius;
				if (ImGui::DragFloat("Planet Radius(km)", &copyValue.bottomRadius, 100.0f, 8000.0f))
				{
					// Update cloud start height.
					float cloudBottomAltitude = inoutCloud.cloudAreaStartHeight - oldBottomRadius;
					inoutCloud.cloudAreaStartHeight = copyValue.bottomRadius + cloudBottomAltitude;
				}
				ImGui::DragFloat("Atmosphere Height(km)", &atmosphereHeight, 10.0f, 150.0f);
			}
			ImGui::PopItemWidth();
			ui::endGroupPanel();

			ui::beginGroupPanel("Detailed Color");
			ImGui::PushItemWidth(150.0f);
			{
				ImGui::ColorEdit3("Ground Albedo", &copyValue.groundAlbedo.x);

				ImGui::ColorEdit3("Mie Scatter Color", &copyValue.mieScatteringColor.x);
				 ImGui::DragFloat("Mie Scatter Scale", &copyValue.mieScatteringLength, 0.00001f, 0.1f, 10.0f, "%.6f");

				ImGui::ColorEdit3("Mie Absorb Color", &copyValue.mieAbsColor.x);
				 ImGui::DragFloat("Mie Absorb Scale", &copyValue.mieAbsLength, 0.00001f, 10.0f, 1000.0f, "%.6f");

				ImGui::ColorEdit3("Ray Scatter Color", &copyValue.rayleighScatteringColor.x);
				 ImGui::DragFloat("Ray Scatter Scale", &copyValue.rayleighScatterLength, 0.00001f, 10.0f, 1000.0f, "%.6f");

				ImGui::ColorEdit3("Ozone Absorb Color", &copyValue.absorptionColor.x);
				 ImGui::DragFloat("Ozone Absorb Scale", &copyValue.absorptionLength, 0.00001f, 10.0f, 1000.0f, "%.6f");
			}
			ImGui::PopItemWidth();
			ui::endGroupPanel();

			copyValue.topRadius = copyValue.bottomRadius + atmosphereHeight;

			if (copyValue != inout)
			{
				inout = copyValue;
				bResult = true;
			}
		}

		return bResult;
	}

	inline static bool drawRaytraceShadowConfig(RaytraceShadowConfig& inout)
	{
		ImGui::PushID("RaytraceShadowConfig");
		bool bChangedValue = false;

		auto copyValue = inout;

		if (ImGui::CollapsingHeader("RayTrace Shadow Setting"))
		{
			ImGui::DragFloat("Ray Min Range", &copyValue.rayMinRange, 0.0001f, 0.0001f, 1.0f);
			ImGui::DragFloat("Ray Max Range", &copyValue.rayMaxRange, 100.0f, 500.0f, 10000.0f);

			ImGui::DragFloat("Light Radius", &copyValue.lightRadius, 0.001f, 0.001f, 0.1f);
		}

		if (copyValue != inout)
		{
			inout = copyValue;
			bChangedValue = true;
		}
		
		ImGui::PopID();

		return bChangedValue;
	}

	inline static bool drawCascadeConfig(CascadeShadowConfig& inout)
	{
		bool bChangedValue = false;

		auto copyValue = inout;

		if (ImGui::CollapsingHeader("Cascade Shadow Setting"))
		{
			ui::beginGroupPanel("Cascade Config");
			ImGui::PushItemWidth(100.0f);


			bChangedValue |= ImGui::Checkbox("Enable SDSM", (bool*)&copyValue.bSDSM);

			ImGui::DragInt("Count",     &copyValue.cascadeCount,    1.0f,  1, (int)kMaxCascadeNum);
			ImGui::DragInt("Dimension", &copyValue.percascadeDimXY, 512, 512, 4096);

			ImGui::DragFloat("Split Lambda", &copyValue.splitLambda,          0.01f, 0.00f, 2.00f);
			ImGui::DragFloat("Max Distance", &copyValue.maxDrawDepthDistance, 10.0f, 50.0f, 2000.0f);

			ImGui::DragFloat("Bias Const", &copyValue.shadowBiasConst, 0.01f, -5.0f, 5.0f);
			ImGui::DragFloat("Bias Slope", &copyValue.shadowBiasSlope, 0.01f, -5.0f, 5.0f);

			ImGui::DragFloat("Filter Size", &copyValue.filterSize, 0.01f, 0.01f, 10.0f);
			ImGui::DragFloat("Mix Edge", &copyValue.cascadeMixBorder, 0.01f, 0.05f, 0.50f);

			bChangedValue |= ImGui::Checkbox("Enable Contact Shadow", (bool*)&copyValue.bContactShadow);
			ImGui::DragInt("Contact Shadow SampleCount", &copyValue.contactShadowSampleNum, 1, 4, 32);
			ImGui::DragFloat("Contact Shadow Length", &copyValue.contactShadowLen, 0.1f, 0.1f, 2.0f);
			ImGui::PopItemWidth();
			ui::endGroupPanel();
		}

		copyValue.percascadeDimXY = math::clamp(getNextPOT(copyValue.percascadeDimXY), 512, 4096);
		copyValue.cascadeCount = math::clamp(copyValue.cascadeCount, 1, (int)kMaxCascadeNum);

		if (copyValue != inout)
		{
			inout = copyValue;
			bChangedValue = true;
		}

		return bChangedValue;
	}

	inline static bool drawSunLight(const char* name, SkyLightInfo& inout)
	{
		bool bChangedValue = false;
		SkyLightInfo copyInout = inout;

		ImGui::PushID(name);
		{
			ImGui::Spacing();

			ImGui::ColorEdit3("Color", &copyInout.color.x);
			ImGui::DragFloat("Intensity", &copyInout.intensity, 0.25f, 0.0f, 1000.0f);

			ImGui::ColorEdit3("ShadowColor", &copyInout.shadowColor.x);
			ImGui::DragFloat("ShadowColor Intensity", &copyInout.shadowColorIntensity, 0.25f, 0.0f, 1000.0f);

			copyInout.shadowType = ui::drawComboEnumSelect(copyInout.shadowType, (size_t)EShadowType_Max, "ShadowType");
			if (copyInout.shadowType == EShadowType_CascadeShadowMap)
			{
				drawCascadeConfig(copyInout.cascadeConfig);
			}
			else if (copyInout.shadowType == EShadowType_Raytrace)
			{
				drawRaytraceShadowConfig(copyInout.rayTraceConfig);
			}


			ImGui::Spacing();
		}
		ImGui::PopID();

		if (copyInout != inout)
		{
			inout = copyInout;
			bChangedValue = true;
		}

		return bChangedValue;
	}

	void SkyComponent::tick(const RuntimeModuleTickData& tickData)
	{
		m_prevFrameSun = m_sun;

		// Update skylight info.
		{
			m_sun.direction = getSunDirection();
		}
	}

	bool SkyComponent::uiDrawComponent()
	{
		bool bChangedValue = false;

		bChangedValue |= drawSunLight("Sun", m_sun);

		bChangedValue |= ImGui::Checkbox("LocalTime", &m_bLocalTime);
		bChangedValue |= m_tod.uiDraw(m_bLocalTime);
		bChangedValue |= uiDrawAtmosphereParameters(m_atmosphere, m_cloud);
		bChangedValue |= uiDrawCloudParameters(m_cloud, m_atmosphere);

		return bChangedValue;
	}

	const UIComponentReflectionDetailed& SkyComponent::uiComponentReflection()
	{
		static const UIComponentReflectionDetailed reflection =
		{
			.bOptionalCreated = true,
			.iconCreated = ICON_FA_SUN + std::string("   Sky"),
		};
		return reflection;
	}

	math::vec3 SkyComponent::getSunDirection() const
	{
		// Get scene node direction as sun direction.
		if (auto node = m_node.lock())
		{
			const auto& worldMatrix = node->getTransform()->getWorldMatrix();
			constexpr math::vec3 forward = math::vec3(0.0f, -1.0f, 0.0f);
			auto result = math::normalize(math::vec3(worldMatrix * vec4(forward, 0.0f)));
			if (result == forward)
			{
				result = math::normalize(math::vec3(1e-3f, -1.0f, 1e-3f));
			}
			return result;
		}

		return math::normalize(math::vec3(0.1f, -0.8f, 0.2f));
	}

	bool SkyComponent::collectSkyLight(RenderScene& renderScene)
	{
		if (m_sun.direction != m_prevFrameSun.direction ||
			m_sun.intensity != m_prevFrameSun.intensity ||
			m_sun.color     != m_prevFrameSun.color)
		{
			renderScene.clearAllReflectionCapture();
		}

		return true;
	}

	TimeOfDay::TimeOfDay()
	{
		auto timeNow = std::chrono::system_clock::now();
		std::time_t tt = std::chrono::system_clock::to_time_t(timeNow);
		std::tm tm = *std::localtime(&tt);

		year  = 1900 + tm.tm_year;
		month = 1    + tm.tm_mon;
		day   = tm.tm_mday;

		hour   = tm.tm_hour;
		minute = tm.tm_min;
		second = tm.tm_sec;
	}

	bool TimeOfDay::uiDraw(bool bLocalTime)
	{
		const auto copy = *this;
		{
			ImGui::Unindent();
			ui::beginGroupPanel("Time of Day");

			ImGui::BeginGroup();

			ImGui::DragInt3("YMD", &year);
			ui::hoverTip("Sky year month day config.");

			ImGui::DragInt3("HMS", &hour);
			ui::hoverTip("Sky hour minute seconds config.");

			year = math::clamp(year, 0, 3000);
			month = math::clamp(month, 1, 12);
			day = math::clamp(day, 1, 31);

			if (month == 2)
			{
				bool b29Day = false;
				int maxDay = 28;

				if (year % 4 == 0)
				{
					if (year % 100 == 0)
					{
						if (year % 400 == 0)
						{
							maxDay = 29;
						}
					}
					else
					{
						maxDay = 29;
					}
				}

				day = math::clamp(day, 1, maxDay);
			}

			hour = math::clamp(hour, 0, 23);
			minute = math::clamp(minute, 0, 59);
			second = math::clamp(second, 0, 59);

			ImGui::EndGroup();

			ui::endGroupPanel();
			ImGui::Indent();
		}
		if (bLocalTime)
		{
			*this = {};
		}
 
		return copy != *this;
	}

	CascadeShadowConfig engine::defaultCascadeConfig()
	{
		CascadeShadowConfig config{ };

		config.bSDSM = true;
		config.bContactShadow = true;
		config.cascadeCount = 4;
		config.percascadeDimXY = 2048;
		config.splitLambda = 0.95f;

		config.maxDrawDepthDistance = 2000.0f;

		// Reverse Z.
		config.shadowBiasConst = -1.25f;
		config.shadowBiasSlope = -1.75f;

		config.filterSize = 5.0f;
		config.cascadeMixBorder = 0.10f;

		config.contactShadowLen = 1.0f;
		config.contactShadowSampleNum = 8;
		return config;
	}

	RaytraceShadowConfig engine::defaultRaytraceShadowConfig()
	{
		RaytraceShadowConfig config;
		config.rayMinRange = 0.01f;
		config.rayMaxRange = 1000.0f;
		config.lightRadius = 0.05f;

		return config;
	}


	SkyLightInfo engine::defaultSun()
	{
		SkyLightInfo sun{ };

		sun.color = temperature2Color(6500.0f);
		sun.direction = math::normalize(vec3(1e-2f, -0.9f, 1e-3f));
		sun.intensity = 1.0f;
		sun.shadowType = EShadowType_CascadeShadowMap;
		sun.cascadeConfig = defaultCascadeConfig();

		sun.shadowColor = vec3(1.0, 0.7, 0.4);
		sun.shadowColorIntensity = 1.0f;
		sun.rayTraceConfig = defaultRaytraceShadowConfig();
		return sun;
	}
}