#include "component_draw.h"
#include <imgui/ui.h>
#include <scene/scene.h>
#include "../editor/editor.h"
#include "../editor/widgets/project_content.h"
#include <scene/component/sky.h>

using namespace engine;
using namespace engine::ui;

void drawCascadeConfig(CascadeShadowConfig& inout)
{
	auto copyValue = inout;

	if (ImGui::CollapsingHeader("Shadow Setting"))
	{
		ui::beginGroupPanel("Cascade Config");
		ImGui::PushItemWidth(100.0f);
		ImGui::DragInt("Count", &copyValue.cascadeCount, 1.0f, 1, kMaxCascadeNum);
		ImGui::DragInt("ShadowMap Size", &copyValue.percascadeDimXY, 512, kMinCascadeDimSize, kMaxCascadeDimSize);
		ImGui::DragFloat("Split Lambda", &copyValue.cascadeSplitLambda, 0.01f, 0.0f, 2.0f);
		ImGui::DragFloat("Border Adopt", &copyValue.cascadeBorderAdopt, 0.00001f, 0.0f, 0.001f);
		ImGui::DragFloat("Edge Lerp Threshold", &copyValue.cascadeEdgeLerpThreshold, 0.01f, 0.0f, 1.0f);
		ImGui::DragFloat("Max Draw Distance", &copyValue.maxDrawDepthDistance, 10.0f, 50.0f, 800.0f);
		ImGui::PopItemWidth();
		ui::endGroupPanel();

		ui::beginGroupPanel("Filter Config");
		ImGui::PushItemWidth(100.0f);
		ImGui::DragFloat("Size", &copyValue.shadowFilterSize, 0.01f, 0.0f, 2.0f);
		ImGui::DragFloat("Max Size", &copyValue.maxFilterSize, 0.01f, 0.0f, 10.0f);
		ImGui::DragFloat("Depth Bias Const", &copyValue.shadowBiasConst, 0.01f, -5.0f, 5.0f);
		ImGui::DragFloat("Depth Bias Slope", &copyValue.shadowBiasSlope, 0.01f, -5.0f, 5.0f);
		ImGui::PopItemWidth();
		ui::endGroupPanel();
	}

	copyValue.percascadeDimXY = glm::clamp(getNextPOT(copyValue.percascadeDimXY), kMinCascadeDimSize, kMaxCascadeDimSize);
	copyValue.cascadeCount = glm::clamp(copyValue.cascadeCount, 1, kMaxCascadeNum);

	if (copyValue != inout)
	{
		inout = copyValue;
	}
}

void drawAtmosphereConfig(AtmosphereConfig& inout)
{
	if (ImGui::CollapsingHeader("Atmosphere Config"))
	{
		ImGui::Spacing();
		if (ImGui::Button("Reset Atmosphere"))
		{
			inout.resetAtmosphere();
		}
		ImGui::Spacing();
		auto copyValue = inout;

		float atmosphereHeight = copyValue.topRadius - copyValue.bottomRadius;

		ui::beginGroupPanel("Size Config");
		ImGui::PushItemWidth(100.0f);
		{

			ImGui::DragFloat("Pre Exposure", &copyValue.atmospherePreExposure, 0.01f, 0.01f, 1.0f);
			ImGui::DragFloat("Mie Phase", &copyValue.miePhaseFunctionG, 0.0f, 0.01f, 0.99f);
			ImGui::DragFloat("Multi Scatter", &copyValue.multipleScatteringFactor, 0.01f, 0.01f, 1.0f);
			ImGui::SliderInt("Min SPP", &copyValue.viewRayMarchMinSPP, 1, 30);
			ImGui::SliderInt("Max SPP", &copyValue.viewRayMarchMaxSPP, 2, 31);
			ImGui::DragFloat("Planet Radius(km)", &copyValue.bottomRadius, 100.0f, 8000.0f);
			ImGui::DragFloat("Atmosphere Height(km)", &atmosphereHeight, 10.0f, 150.0f);
		}
		ImGui::PopItemWidth();
		ui::endGroupPanel();

		ui::beginGroupPanel("Render Config");
		ImGui::PushItemWidth(150.0f);
		{
			ImGui::ColorEdit3("Ground Albedo", &copyValue.groundAlbedo.x);

			ImGui::ColorEdit3("Mie Scatter Color", &copyValue.mieScatteringColor.x);
			ImGui::DragFloat("Mie Scatter Scale", &copyValue.mieScatteringLength, 0.00001f, 0.1f);

			ImGui::ColorEdit3("Mie Absorb Color", &copyValue.mieAbsColor.x);
			ImGui::DragFloat("Mie Absorb Scale", &copyValue.mieAbsLength, 0.00001f, 10.0f);

			ImGui::ColorEdit3("Ray Scatter Color", &copyValue.rayleighScatteringColor.x);
			ImGui::DragFloat("Ray Scatter Scale", &copyValue.rayleighScatterLength, 0.00001f, 10.0f);

			ImGui::ColorEdit3("Ozone Absorb Color", &copyValue.absorptionColor.x);
			ImGui::DragFloat("Ozone Absorb Scale", &copyValue.absorptionLength, 0.00001f, 10.0f);

			// Additional info.
			float earthAtmosphereMieScaleHeight = -1.0f / copyValue.mieDensity[7];
			float earthAtmosphereRayleighScaleHeight = -1.0f / copyValue.rayleighDensity[7];

			ImGui::DragFloat("Mie Scale Height", &earthAtmosphereMieScaleHeight, 0.5f, 20.0f);
			ImGui::DragFloat("Ray Scale Height", &earthAtmosphereRayleighScaleHeight, 0.5f, 20.0f);
		}
		ImGui::PopItemWidth();
		ui::endGroupPanel();

		copyValue.topRadius = copyValue.bottomRadius + atmosphereHeight;

		if (copyValue != inout)
		{
			inout = copyValue;
		}
	}

	if (ImGui::CollapsingHeader("Cloud Setting"))
	{
		ImGui::PushItemWidth(150.0f);
		ImGui::Spacing();
		if (ImGui::Button("Reset Cloud"))
		{
			inout.resetCloud();
		}
		ImGui::Spacing();
		auto copyValue = inout;

		float cloudBottomAltitude = copyValue.cloudAreaStartHeight - copyValue.bottomRadius;

		bool bEnableGroundContribution = copyValue.cloudEnableGroundContribution != 0;
		ImGui::Checkbox("enable ground contribution", &bEnableGroundContribution);
		copyValue.cloudEnableGroundContribution = bEnableGroundContribution ? 1 : 0;

		ImGui::DragFloat("start height(km)", &cloudBottomAltitude, 0.1f, 0.0f, 20.0f);
		ImGui::DragFloat("thickness(km)", &copyValue.cloudAreaThickness, 0.1f, 0.1f, 20.0f);
		ImGui::DragFloat("shadow extent(km)", &copyValue.cloudShadowExtent, 0.1f, 1.0f, 50.0f);

		ImGui::DragFloat("coverage", &copyValue.cloudCoverage, 0.1f, 0.0f, 1.0f);
		ImGui::DragFloat("density", &copyValue.cloudDensity, 0.1f, 0.0f, 1.0f);
		ImGui::DragFloat("fog fade", &copyValue.cloudFogFade, 0.001f, 0.0f, 0.1f);
		ImGui::DragFloat("max tracing distance", &copyValue.cloudMaxTraceingDistance, 1.0f, 10.0f, 100.0f);
		ImGui::DragFloat("max tracing start distance", &copyValue.cloudTracingStartMaxDistance, 1.0f, 300.0f, 500.0f);

		ImGui::DragFloat2("Wether UV scale", &copyValue.cloudWeatherUVScale.x, 0.0f, 0.01f);

		ImGui::DragFloat("basic noise scale", &copyValue.cloudBasicNoiseScale, 0.01f, 0.0f, 1.0f);
		ImGui::DragFloat("detail noise scale", &copyValue.cloudDetailNoiseScale, 0.01f, 0.0f, 1.0f);

		ImGui::DragFloat("sun light scale", &copyValue.cloudShadingSunLightScale, 0.1f, 0.1f, 10.0f);


		ImGui::DragFloat3("wind dir", &copyValue.cloudDirection.x, 0.01f, 0.0f);
		ImGui::DragFloat("wind speed", &copyValue.cloudSpeed, 0.1f, 0.0f, 1.0f);


		ImGui::DragFloat("multi scatter", &copyValue.cloudMultiScatterScatter, 0.1f, 0.0f, 1.0f);
		ImGui::DragFloat("multi extinction", &copyValue.cloudMultiScatterExtinction, 0.1f, 0.0f, 1.0f);

		ImGui::DragFloat("phase forward", &copyValue.cloudPhaseForward, 0.01f, 0.01f, 0.99f);
		ImGui::DragFloat("phase backward", &copyValue.cloudPhaseBackward, 0.01f, -0.99f, -0.01f);
		ImGui::DragFloat("phase mix factor", &copyValue.cloudPhaseMixFactor, 0.01f, 0.01f, 0.99f);
		ImGui::DragFloat("cloud powder scale", &copyValue.cloudPowderScale, 0.1f, 0.01f, 100.0f);
		ImGui::DragFloat("cloud powder pow", &copyValue.cloudPowderPow, 0.1f, 0.01f, 10.0f);

		ImGui::ColorEdit3("cloud albedo", &copyValue.cloudAlbedo.x);
		ImGui::DragFloat("light step mul", &copyValue.cloudLightStepMul, 0.01f, 1.01f, 1.5f);
		ImGui::DragFloat("light basic step", &copyValue.cloudLightBasicStep, 0.01f, 0.10f, 0.5f);
		ImGui::DragInt("light step num", &copyValue.cloudLightStepNum, 1, 6, 24);

		ImGui::Spacing();
		ImGui::PopItemWidth();

		copyValue.cloudAreaStartHeight = cloudBottomAltitude + copyValue.bottomRadius;

		if (copyValue != inout)
		{
			inout = copyValue;
		}
	}
}

void ComponentDrawer::drawSky(std::shared_ptr<SceneNode> node)
{
	auto comp = node->getComponent<SkyComponent>();

	drawLight(comp);

	ImGui::Separator();
	ImGui::Spacing();

	drawCascadeConfig(comp->getCacsadeConfig());
	drawAtmosphereConfig(comp->getAtmosphereConfig());
}