#include "component_draw.h"
#include <imgui/ui.h>
#include <scene/scene.h>
#include "../editor/editor.h"
#include "../editor/widgets/project_content.h"
#include <scene/component/postprocess.h>

using namespace engine;
using namespace engine::ui;

void ComponentDrawer::drawPostprocess(std::shared_ptr<engine::SceneNode> node)
{
	std::shared_ptr<PostprocessVolumeComponent> comp = node->getComponent<PostprocessVolumeComponent>();
	auto copySetting = comp->getSetting();

	if (ImGui::CollapsingHeader("Exposure setting"))
	{
		ImGui::PushID("ExposureSetting");
		ImGui::Spacing();
		ImGui::Indent();
		ImGui::PushItemWidth(100.0f);

		// TODO: Change to physic camera parameters.
		ImGui::Checkbox("Auto Exposure", &copySetting.bAutoExposure);
		if (copySetting.bAutoExposure)
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
			ImGui::DragFloat("Fix exposure", &copySetting.fixExposure, 0.1f, 0.1f, 100.0f);
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

#if 0
	if (ImGui::CollapsingHeader("GTAO Setting"))
	{
		ImGui::PushID("GTAO");
		ImGui::Spacing();
		ImGui::Indent();
		ImGui::PushItemWidth(100.0f);

		ImGui::DragInt("slice num", &copySetting.gtaoSliceNum, 1, 1, 8);
		if (copySetting.gtaoSliceNum >= 1)
		{
			copySetting.gtaoSliceNum = glm::clamp(getNextPOT(copySetting.gtaoSliceNum), 1, 8);
		}
		else
		{
			copySetting.gtaoSliceNum = 1;
		}

		ImGui::DragInt("step times", &copySetting.gtaoStepNum, 1, 1, 12);
		ImGui::DragFloat("radius", &copySetting.gtaoRadius, 0.1f, 0.5f, 4.0f);
		ImGui::DragFloat("thickness", &copySetting.gtaoThickness, 0.1f, 0.1f, 1.0f);

		ImGui::DragFloat("ao power", &copySetting.gtaoPower, 0.1f, 0.1f, 3.0f);
		ImGui::DragFloat("ao intensity", &copySetting.gtaoIntensity, 0.01f, 0.0f, 1.0f);

		ImGui::PopItemWidth();
		ImGui::Unindent();
		ImGui::Spacing();
		ImGui::PopID();
	}
#endif

	if (ImGui::CollapsingHeader("SSGI Setting"))
	{
		ImGui::PushID("SSGI");
		ImGui::Spacing();
		ImGui::Indent();
		ImGui::PushItemWidth(100.0f);

		ImGui::DragInt("slice num", &copySetting.ssgiSliceCount, 1, 1, 8);

		ImGui::DragInt("step times", &copySetting.ssgiStepCount, 1, 1, 12);
		ImGui::DragFloat("radius", &copySetting.ssgiViewRadius, 0.1f, 0.5f, 4.0f);

		ImGui::DragFloat("falloff", &copySetting.ssgiFalloff, 0.1f, 0.1f, 1.0f);

		ImGui::DragFloat("ao power", &copySetting.ssgiPower, 0.1f, 0.1f, 3.0f);
		ImGui::DragFloat("ao intensity", &copySetting.ssgiIntensity, 0.01f, 0.0f, 1.0f);

		ImGui::PopItemWidth();
		ImGui::Unindent();
		ImGui::Spacing();
		ImGui::PopID();
	}

	comp->changeSetting(copySetting);
}