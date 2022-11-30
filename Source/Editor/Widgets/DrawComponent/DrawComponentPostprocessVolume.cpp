#include "Pch.h"
#include "../Detail.h"
#include "DrawComponent.h"

using namespace Flower;
using namespace Flower::UI;

void ComponentDrawer::drawPostprocessVolume(std::shared_ptr<SceneNode> node)
{
	std::shared_ptr<PostprocessVolumeComponent> comp = node->getComponent<PostprocessVolumeComponent>();

	auto copySetting = comp->getSetting();

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

	if (comp->changeSetting(copySetting))
	{
		comp->markDirty();
	}
}