#include "postprocess_component.h"

#include <iconFontcppHeaders/IconsFontAwesome6.h>
#include <editor/widgets/content.h>
#include <editor/editor.h>

namespace engine
{
	constexpr float kExposureDefaultValue = +10.0f;
	constexpr float kExposureMaxEv = +9.0f;
	constexpr float kExposureMinEv = -9.0f;
	constexpr float kExposureDiff = +kExposureMaxEv - kExposureMinEv;
	constexpr float kExposureScale = +1.0f / kExposureDiff;
	constexpr float kExposureOffset = -kExposureMinEv * kExposureScale;

	bool uiDrawPostprocessing(PostprocessVolumeSetting& inout)
	{
		bool bResult = false;

		auto copySetting = inout;

		ui::drawCollapsingHeader("Exposure", [&]() 
		{
			ImGui::Checkbox("Auto Exposure", (bool*)&copySetting.bAutoExposureEnable);
			if (copySetting.bAutoExposureEnable)
			{
				ImGui::DragFloat("Low Percent", &copySetting.autoExposureLowPercent, 1e-3f, 0.01f, 0.5f);
				ImGui::DragFloat("High Percent", &copySetting.autoExposureHighPercent, 1e-3f, 0.5f, 0.99f);

				ImGui::DragFloat("Min Brightness", &copySetting.autoExposureMinBrightness, 0.5f, -9.0f, 9.0f);
				ImGui::DragFloat("Max Brightness", &copySetting.autoExposureMaxBrightness, 0.5f, -9.0f, 9.0f);

				ImGui::DragFloat("Speed Down", &copySetting.autoExposureSpeedDown, 0.1f, 0.0f);
				ImGui::DragFloat("Speed Up", &copySetting.autoExposureSpeedUp, 0.1f, 0.0f);

				ImGui::DragFloat("Exposure Compensation", &copySetting.autoExposureExposureCompensation, 1.0f, 0.0f);
			}
			else
			{
				ImGui::DragFloat("Fix exposure", &copySetting.autoExposureFixExposure, 0.1f, 0.1f, 100.0f);
			}

		});

		ui::drawCollapsingHeader("SSAO", [&]()
		{
			ImGui::Checkbox("GTAO Filter", (bool*)&copySetting.ssao_bGTAO);



			ImGui::DragInt("Slice Count", &copySetting.ssao_sliceCount, 1, 1, 8);

			ImGui::DragInt("Step Count", &copySetting.ssao_stepCount, 1, 1, 12);
			
			if (copySetting.ssao_bGTAO != 0)
			{
				ImGui::DragFloat("GTAO Radius", &copySetting.gtao_radius, 0.1f, 0.5f, 4.0f);
				ImGui::DragFloat("GTAO FallofEnd", &copySetting.gtao_falloffEnd, 0.1f, 2.0f, 10.0f);
				ImGui::DragFloat("GTAO Thickness", &copySetting.gtao_thickness, 0.1f, 0.1f, 1.0f);
			}
			else
			{
				ImGui::DragFloat("SSAO Radius", &copySetting.ssao_viewRadius, 0.1f, 0.5f, 4.0f);
				ImGui::DragFloat("SSAO Falloff", &copySetting.ssao_falloff, 0.1f, 0.1f, 1.0f);
			}


			ImGui::DragFloat("Post Power", &copySetting.ssao_power, 0.1f, 0.1f, 3.0f);
			ImGui::DragFloat("Post Intensity", &copySetting.ssao_intensity, 0.01f, 0.0f, 1.0f);
		});

		ui::drawCollapsingHeader("Temporal Anti-Alias", [&]() 
		{
			ImGui::Checkbox("Filter Color", (bool*)&copySetting.bTAAEnableColorFilter);
			ImGui::DragFloat("Anti Filcker", &copySetting.taaAntiFlicker, 0.01f, 0.0f, 1.0f);
			ImGui::DragFloat("History Sharpen", &copySetting.taaHistorySharpen, 0.01f, 0.0f, 1.0f);
			ImGui::DragFloat("Base Blend", &copySetting.taaBaseBlendFactor, 0.01f, 0.0f, 1.0f);
		});

		ui::drawCollapsingHeader("Bloom", [&]() 
		{
			ImGui::DragFloat("Intensity", &copySetting.bloomIntensity, 0.01f, 0.0f, 1.0);
			ImGui::DragFloat("Radius", &copySetting.bloomRadius, 0.01f, 0.0f, 1.0);

			ImGui::DragFloat("Threshold", &copySetting.bloomThreshold, 0.01f, 0.0f, 1.0);
			ImGui::DragFloat("Threshold soft", &copySetting.bloomThresholdSoft, 0.01f, 0.0f, 1.0);
		});



		ui::drawCollapsingHeader("Tonemapper", [&]() 
		{
			ImGui::DragFloat("Expand Gamut", &copySetting.expandGamutFactor, 1e-2f, 0.0f, 1.0f);

			copySetting.tonemapper_type = ui::drawComboEnumSelect(copySetting.tonemapper_type, (size_t)ETonemapperType_Max, "Type");

			if (copySetting.tonemapper_type == ETonemapperType_GT)
			{
				ImGui::DragFloat("Contrast", &copySetting.tonemapper_a, 1e-2f, 0.5f, 2.0f);
				ImGui::DragFloat("Black", &copySetting.tonemapper_c,    1e-2f, 1.0f, 2.0f);
				ImGui::DragFloat("Pedestal", &copySetting.tonemapper_b, 1e-2f, -1.0, 2.0f);
				ImGui::DragFloat("Linear Section Start", &copySetting.tonemapper_m, 1e-2f, 0.0f, 0.9f);
				ImGui::DragFloat("linear Section Length", &copySetting.tonemapper_l, 1e-2f, 0.0f, 1.0f);
			}
			else if (copySetting.tonemapper_type == ETonemapperType_FilmACES)
			{
				ImGui::DragFloat("Slope", &copySetting.tonemapper_filmACESSlope, 0.1f, 0.1f, 100.0f);
				ImGui::DragFloat("Toe", &copySetting.tonemapper_filmACESToe, 0.1f, 0.1f, 100.0f);
				ImGui::DragFloat("Shoulder", &copySetting.tonemapper_filmACESShoulder, 0.1f, 0.1f, 100.0f);
				ImGui::DragFloat("Blackclip", &copySetting.tonemapper_filmACESBlackClip, 0.1f, 0.1f, 100.0f);
				ImGui::DragFloat("Whiteclip", &copySetting.tonemapper_filmACESWhiteClip, 0.1f, 0.1f, 100.0f);

				ImGui::DragFloat("PreDesaturate", &copySetting.tonemapper_filmACESPreDesaturate, 0.1f, 0.1f, 1.0f);
				ImGui::DragFloat("PostDesaturate", &copySetting.tonemapper_filmACESPostDesaturate, 0.1f, 0.1f, 1.0f);
				ImGui::DragFloat("RedModifier", &copySetting.tonemapper_filmACESRedModifier, 0.1f, 0.0f, 1.0f);
				ImGui::DragFloat("GlowScale", &copySetting.tonemapper_filmACESGlowScale, 0.1f, 0.0f, 1.0f);
			}

		});

		if (copySetting != inout)
		{
			inout = copySetting;
			bResult = true;
		}
		return bResult;
	}

	bool PostprocessComponent::uiDrawComponent()
	{
		bool bChangedValue = false;

		bChangedValue |= uiDrawPostprocessing(m_setting);

		return bChangedValue;
	}

	const UIComponentReflectionDetailed& PostprocessComponent::uiComponentReflection()
	{
		static const UIComponentReflectionDetailed reflection =
		{
			.bOptionalCreated = true,
			.iconCreated = ICON_FA_STAR + std::string("  Postprocess"),
		};
		return reflection;
	}



	PostprocessVolumeSetting defaultPostprocessVolumeSetting()
	{
		return PostprocessVolumeSetting
		{
			// Auto exposure.
			.bAutoExposureEnable              = true,
			.autoExposureFixExposure          = kExposureDefaultValue,
			.autoExposureLowPercent           =  0.50f,
			.autoExposureHighPercent          =  0.95f,
			.autoExposureMinBrightness        = -3.00f,
			.autoExposureMaxBrightness        =  0.00f,
			.autoExposureSpeedDown            =  1.00f,
			.autoExposureSpeedUp              =  2.00f,
			.autoExposureExposureCompensation =  1.00f,

			.bTAAEnableColorFilter = true,
			.taaAntiFlicker        = 0.5f,
			.taaHistorySharpen     = 0.35f,
			.taaBaseBlendFactor    = 0.875f,

			// Bloom scene.
			.bloomIntensity     = 1.0f,
			.bloomRadius        = 0.80f,
			.bloomThreshold     = 0.80f,
			.bloomThresholdSoft = 0.60f,

			// Expand gamut factor.
			.expandGamutFactor  = 1.0f, 

			// Tonemapper GT.
			.tonemapper_P = 1.00f,
			.tonemapper_a = 1.00f,
			.tonemapper_m = 0.22f,
			.tonemapper_l = 0.40f,
			.tonemapper_c = 1.33f,
			.tonemapper_b = 0.00f,

			// Tonemapper ACES film.
			.tonemapper_type                   = ETonemapperType_FilmACES,
			.tonemapper_filmACESSlope          = 0.91f,
			.tonemapper_filmACESToe            = 0.55f,
			.tonemapper_filmACESShoulder       = 0.26f,
			.tonemapper_filmACESBlackClip      = 0.00f,
			.tonemapper_filmACESWhiteClip      = 0.04f,
			.tonemapper_filmACESPreDesaturate  = 0.96f,
			.tonemapper_filmACESPostDesaturate = 0.93f,
			.tonemapper_filmACESRedModifier    = 1.00f,
			.tonemapper_filmACESGlowScale      = 1.00f,

			// ssao.
			.ssao_sliceCount = 2,
			.ssao_stepCount  = 4,
			.ssao_intensity  = 1.0f,
			.ssao_power      = 1.0f,
			.ssao_viewRadius = 0.2f,
			.ssao_falloff    = 0.1f,

			.gtao_radius     = 0.5f,
			.gtao_thickness  = 0.1f,
			.gtao_falloffEnd = 2.0f,
		};
	}

	PostprocessVolumeSetting computePostprocessSettingDetail(
		const PerFrameData& perframe,
		const PostprocessVolumeSetting& in, 
		float deltaTime)
	{
		PostprocessVolumeSetting result = in;

		{
			result.autoExposureScale = kExposureScale;
			result.autoExposureOffset = kExposureOffset;

			result.autoExposureLowPercent = math::clamp(in.autoExposureLowPercent, 0.01f, 0.99f);
			result.autoExposureHighPercent = math::clamp(in.autoExposureHighPercent, 0.01f, 0.99f);

			result.autoExposureMinBrightness = math::exp2(in.autoExposureMinBrightness);
			result.autoExposureMaxBrightness = math::exp2(in.autoExposureMaxBrightness);

			result.autoExposureDeltaTime = deltaTime;
		}

		// Tonemapper
		{
			float P = in.tonemapper_P;  // Max brightness.
			float a = in.tonemapper_a;  // contrast
			float m = in.tonemapper_m;  // linear section start
			float l = in.tonemapper_l;  // linear section length
			float c = in.tonemapper_c;  // black
			float b = in.tonemapper_b;  // pedestal

			float l0 = ((P - m) * l) / a;
			float L0 = m - m / a;
			float L1 = m + (1.0 - m) / a;
			float S0 = m + l0;
			float S1 = m + a * l0;
			float C2 = (a * P) / (P - S1);
			float CP = -C2 / P;

			result.tonemapper_l0 = l0;
			result.tonemapper_L1 = L1;
			result.tonemapper_S0 = S0;
			result.tonemapper_S1 = S1;
			result.tonemapper_C2 = C2;
			result.tonemapper_CP = CP;
		}

		// ssao.
		{
			const float viewRadius = in.ssao_viewRadius;
			const float falloff = in.ssao_falloff;

			result.ssao_uvRadius = viewRadius * 0.5f * math::max(
				perframe.camProj[0][0], perframe.camProj[1][1]);

			float falloffRange = viewRadius * falloff;
			float falloffFrom = viewRadius * (1.0f - falloff);

			result.ssao_falloffMul = -1.0f / falloffRange;
			result.ssao_falloffAdd = falloffFrom / falloffRange + 1.0f;
		}

		return result;
	}

}