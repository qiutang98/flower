#include "Pch.h"
#include "../Detail.h"
#include "DrawComponent.h"

using namespace Flower;
using namespace Flower::UI;

void ComponentDrawer::drawPostprocessVolume(std::shared_ptr<SceneNode> node)
{
	std::shared_ptr<PostprocessVolumeComponent> comp = node->getComponent<PostprocessVolumeComponent>();

	auto copySetting = comp->getSetting();

	if (ImGui::CollapsingHeader("Vignette"))
	{
		ImGui::PushID("Vignette");
		ImGui::Spacing();
		ImGui::Indent();
		ImGui::PushItemWidth(100.0f);

		ImGui::Checkbox("Enable", &copySetting.bEnableVignette);

		if (copySetting.bEnableVignette)
		{
			ImGui::SliderFloat("Falloff", &copySetting.vignette_falloff, 0.0f, 1.0f);
		}

		ImGui::PopItemWidth();
		ImGui::Unindent();
		ImGui::Spacing();
		ImGui::PopID();
	}

	if (ImGui::CollapsingHeader("Fringe"))
	{
		ImGui::PushID("Fringe");
		ImGui::Spacing();
		ImGui::Indent();
		ImGui::PushItemWidth(100.0f);

		enum FocusMode
		{
			Mode_Off = 0,
			Mode_Conrady = 1,
			Mode_Barrel = 2,
			Mode_LateralShift = 3,
		};
		int mode = copySetting.bEnableFringeMode;
		if (ImGui::RadioButton("Off", mode == Mode_Off)) { mode = (int)Mode_Off; } ImGui::SameLine();
		if (ImGui::RadioButton("Conrady", mode == Mode_Conrady)) { mode = (int)Mode_Conrady; } ImGui::SameLine();
		if (ImGui::RadioButton("Barrel", mode == Mode_Barrel)) { mode = (int)Mode_Barrel; } ImGui::SameLine();
		if (ImGui::RadioButton("LateralShift", mode == Mode_LateralShift)) { mode = (int)Mode_LateralShift; }

		copySetting.bEnableFringeMode = mode;

		if (copySetting.bEnableFringeMode > 0)
		{
			if (copySetting.bEnableFringeMode == Mode_LateralShift)
			{
				ImGui::SliderFloat("lateral shift", &copySetting.fringe_lateralShift, 0.0f, 2.0f);
			}
			else
			{
				ImGui::SliderFloat("barrel strength", &copySetting.fringe_barrelStrength, 0.0f, 3.0f);
				ImGui::SliderFloat("zoom strength", &copySetting.fringe_zoomStrength, 0.0f, 3.0f);
			}
		}

		ImGui::PopItemWidth();
		ImGui::Unindent();
		ImGui::Spacing();
		ImGui::PopID();
	}

	if (ImGui::CollapsingHeader("Bloom Setting"))
	{
		ImGui::PushID("BloomSetting");
		ImGui::Spacing();
		ImGui::Indent();
		ImGui::PushItemWidth(100.0f);

		ImGui::DragFloat("Intensity", &copySetting.bloomIntensity, 0.01f, 0.0f, 1.0);
		ImGui::DragFloat("Radius", &copySetting.bloomRadius, 0.01f, 0.0f, 1.0);

		ImGui::DragFloat("Threshold", &copySetting.bloomThreshold, 0.01f, 0.0f, 1.0);
		ImGui::DragFloat("Threshold soft", &copySetting.bloomThresholdSoft, 0.01f, 0.0f, 1.0);

		ImGui::PopItemWidth();
		ImGui::Unindent();
		ImGui::Spacing();
		ImGui::PopID();
	}

	if (ImGui::CollapsingHeader("Exposure setting"))
	{
		ImGui::PushID("ExposureSetting");
		ImGui::Spacing();
		ImGui::Indent();
		ImGui::PushItemWidth(100.0f);


		ImGui::DragFloat("Low Percent", &copySetting.autoExposureLowPercent, 1e-3f, 0.01f, 0.5f);
		ImGui::DragFloat("High Percent", &copySetting.autoExposureHighPercent, 1e-3f, 0.5f, 0.99f);

		ImGui::DragFloat("Min Brightness", &copySetting.autoExposureMinBrightness, 0.5f, -9.0f, 9.0f);
		ImGui::DragFloat("Max Brightness", &copySetting.autoExposureMaxBrightness, 0.5f, -9.0f, 9.0f);

		ImGui::DragFloat("Speed Down", &copySetting.autoExposureSpeedDown, 0.1f, 0.0f);
		ImGui::DragFloat("Speed Up", &copySetting.autoExposureSpeedUp, 0.1f, 0.0f);

		ImGui::DragFloat("Exposure Compensation", &copySetting.autoExposureExposureCompensation, 1.0f, 0.0f);

		ImGui::PopItemWidth();
		ImGui::Unindent();
		ImGui::Spacing();
		ImGui::PopID();
	}

	if (ImGui::CollapsingHeader("GTAO Setting"))
	{
		ImGui::PushID("GTAO");
		ImGui::Spacing();
		ImGui::Indent();
		ImGui::PushItemWidth(100.0f);

		ImGui::DragInt("slice num", &copySetting.gtaoSliceNum, 1, 1, 8);
		if (copySetting.gtaoSliceNum >= 1)
		{
			copySetting.gtaoSliceNum = (int)glm::clamp(getNextPOT(copySetting.gtaoSliceNum), 1u, 8u);
		}
		else
		{
			copySetting.gtaoSliceNum = 1;
		}

		ImGui::DragInt("step times", &copySetting.gtaoStepNum, 1, 1, 12);
		ImGui::DragFloat("radius", &copySetting.gtaoRadius, 0.1f, 0.5f, 4.0f);
		ImGui::DragFloat("thickness", &copySetting.gtaoThickness, 0.1f, 0.1f, 1.0f);

		ImGui::DragFloat("ao power", &copySetting.gtaoPower, 0.1f, 0.1f, 3.0f);
		ImGui::DragFloat("ao intensity", &copySetting.gtaoIntensity, 0.1f, 0.0f, 1.0f);

		ImGui::PopItemWidth();
		ImGui::Unindent();
		ImGui::Spacing();
		ImGui::PopID();
	}

	if (ImGui::CollapsingHeader("Global Tonemmaper Setting"))
	{
		ImGui::Spacing();
		ImGui::Indent();
		ImGui::PushItemWidth(100.0f);

		ImGui::DragFloat("Max brightness", &copySetting.tonemapper_P, 1.0f, 1.0f);
		ImGui::DragFloat("scale", &copySetting.tonemmaper_s, 1.0f, 1.0f);

		ImGui::DragFloat("contrast", &copySetting.tonemapper_a, 1.0f, 1.0f);
		ImGui::DragFloat("linear section start", &copySetting.tonemapper_m, 1.0f, 1.0f);
		ImGui::DragFloat("linear section length", &copySetting.tonemapper_l, 1.0f, 1.0f);
		ImGui::DragFloat("black", &copySetting.tonemapper_c, 1.0f, 1.0f);
		ImGui::DragFloat("pedestal", &copySetting.tonemapper_b, 1.0f, 1.0f);



		ImGui::PopItemWidth();
		ImGui::Unindent();
		ImGui::Spacing();
	}

	if (ImGui::CollapsingHeader("Depth of field Setting"))
	{
		ImGui::PushID("Dof");
		ImGui::Spacing();
		ImGui::Indent();
		ImGui::PushItemWidth(100.0f);

		ImGui::Checkbox("Enable", &copySetting.bDofEnable);

		if (copySetting.bDofEnable)
		{
			ImGui::Checkbox("Near Blur", &copySetting.dof_bNearBlur);

			enum FocusMode
			{
				Mode_Point = 0,
				Mode_Distance = 1,
				Mode_PMXCharater = 2,
			};
			int mode = copySetting.dof_focusMode;

			if (ImGui::RadioButton("Fix Point", mode == Mode_Point)) { mode = (int)Mode_Point; } ImGui::SameLine();
			if (ImGui::RadioButton("Distance", mode == Mode_Distance)) { mode = (int)Mode_Distance; } ImGui::SameLine();
			if (ImGui::RadioButton("PMX Character", mode == Mode_PMXCharater)) { mode = (int)Mode_PMXCharater; }
			copySetting.dof_focusMode = mode;

			if (mode == Mode_Distance)
			{
				ImGui::DragFloat("focus distance", &copySetting.dof_focusDistance, 0.5f, 0.01f);
			}
			else if (mode == Mode_Point)
			{
				UIHelper::drawVector3("World Pos", copySetting.dof_focusPoint, glm::vec3{ 0.0f }, ImGui::GetFontSize() * 4.0f);
			}
			else if (mode == Mode_PMXCharater)
			{
				enum PMXFocusMode
				{
					FocusMode_Far = 0,
					FocusMode_Near = 1,
					FocusMode_Avg = 2,
				};
				int mode = copySetting.dof_trackPMXMode;

				if (ImGui::RadioButton("Far", mode == FocusMode_Far)) { mode = (int)FocusMode_Far; } ImGui::SameLine();
				if (ImGui::RadioButton("Near", mode == FocusMode_Near)) { mode = (int)FocusMode_Near; } ImGui::SameLine();
				if (ImGui::RadioButton("Avg", mode == FocusMode_Avg)) { mode = (int)FocusMode_Avg; }
				copySetting.dof_trackPMXMode = mode;

				ImGui::DragFloat("offset", &copySetting.dof_pmxFoucusMinOffset, 0.001f);
			}

			ImGui::DragFloat("f-stop", &copySetting.dof_aperture, 0.5f, 0.1f);

			ImGui::Checkbox("focus length from camera fov", &copySetting.dof_bUseCameraFOV);
			if (!copySetting.dof_bUseCameraFOV)
			{
				ImGui::SliderFloat("focus length", &copySetting.dof_focusLength, 10.0f, 300.0f);
			}

			ImGui::SliderInt("kernal size", &copySetting.dof_kernelSize, 0, 3);

			ImGui::DragFloat("film height", &copySetting.dof_FilmHeight, 0.001f, 0.0f);

		}


		ImGui::PopItemWidth();
		ImGui::Unindent();
		ImGui::Spacing();
		ImGui::PopID();
	}

	if (comp->changeSetting(copySetting))
	{
		comp->markDirty();
	}
}