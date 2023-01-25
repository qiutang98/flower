#include "Pch.h"
#include "../Detail.h"
#include "DrawComponent.h"

using namespace Flower;
using namespace Flower::UI;

void ComponentDrawer::drawSunSky(std::shared_ptr<SceneNode> node)
{
	std::shared_ptr<SunSkyComponent> comp = node->getComponent<SunSkyComponent>();
	drawLight(comp);

	ImGui::Separator();

	ImGui::Text("Shadow Setting");
	ImGui::Spacing();

	if (ImGui::CollapsingHeader("Cascade Config"))
	{
		ImGui::PushItemWidth(100.0f);

		int cascadeCount = (int)comp->getCascadeCount();
		ImGui::DragInt("Cascade Count", &cascadeCount, 1.0f, 1, GMaxCascadePerDirectionalLight);
		comp->setCascadeCount((uint32_t)cascadeCount);

		int dimXY = (int)comp->getPerCascadeDimXY();
		ImGui::DragInt("Cascade DimXY", &dimXY, 512, 512, 4096);
		comp->setPerCascadeDimXY((uint32_t)dimXY);

		float splitLambda = comp->getCascadeSplitLambda();
		ImGui::DragFloat("Cascade Split Lambda", &splitLambda, 0.01f, 0.0f, 2.0f);
		comp->setCascadeSplitLambda(splitLambda);

		float cascadeBorderAdopt = comp->getCascadeBorderAdopt();
		ImGui::DragFloat("Cascade border adopt", &cascadeBorderAdopt, 0.00001f, 0.0f, 0.001f);
		comp->setCascadeBorderAdopt(cascadeBorderAdopt);

		float cascadeEdgeLerpThreshold = comp->getCascadeEdgeLerpThreshold();
		ImGui::DragFloat("Cascade Edge Lerp Threshold", &cascadeEdgeLerpThreshold, 0.01f, 0.0f, 1.0f);
		comp->setCascadeEdgeLerpThreshold(cascadeEdgeLerpThreshold);


		float maxDrawDistance = comp->getMaxDrawDepthDistance();
		ImGui::DragFloat("MaxDraw distance from near position", &maxDrawDistance, 10.0f, 50.0f, 800.0f);
		comp->setMaxDrawDepthDistance(maxDrawDistance);


		ImGui::PopItemWidth();
	}

	if (ImGui::CollapsingHeader("Shadow Draw Config"))
	{
		ImGui::PushItemWidth(100.0f);

		float filterSize = comp->getShadowFilterSize();
		ImGui::DragFloat("Filter Size", &filterSize, 0.01f, 0.0f, 2.0f);
		comp->setShadowFilterSize(filterSize);

		float maxFilterSize = comp->getMaxFilterSize();
		ImGui::DragFloat("Max Filter Size", &maxFilterSize, 0.01f, 0.0f, 10.0f);
		comp->setMaxFilterSize(maxFilterSize);

		float biasConst = comp->getShadowBiasConst();
		ImGui::DragFloat("Depth Draw Bias Const", &biasConst, 0.01f, -5.0f, 5.0f);
		comp->setShadowBiasConst(biasConst);

		float biasSlope = comp->getShadowBiasSlope();
		ImGui::DragFloat("Depth Draw Bias Slope", &biasSlope, 0.01f, -5.0f, 5.0f);
		comp->setShadowBiasSlope(biasSlope);

		ImGui::PopItemWidth();
	}


	ImGui::Separator();

	ImGui::Text("Atmosphere Sky Setting");
	ImGui::Spacing();

	if (ImGui::CollapsingHeader("Atmosphere Sky Config"))
	{
		ImGui::Indent();
		ImGui::PushItemWidth(100.0f);

		auto earthAtmosphere = comp->getAtmosphere();
		float cloudBottomAltitude = earthAtmosphere.cloudAreaStartHeight - earthAtmosphere.bottomRadius;
		float atmosphereHeight = earthAtmosphere.topRadius - earthAtmosphere.bottomRadius; 

		// Set height info.
		if (ImGui::CollapsingHeader("Earth Cloud"))
		{
			ImGui::PushItemWidth(180.0f);

			if(ImGui::Button("Reset Cloud"))
			{
				earthAtmosphere.resetCloud();
				cloudBottomAltitude = earthAtmosphere.cloudAreaStartHeight - earthAtmosphere.bottomRadius;
			}

			ImGui::DragFloat("Start height(km)", &cloudBottomAltitude, 0.1f, 0.0f, 20.0f);
			ImGui::DragFloat("Thickness(km)", &earthAtmosphere.cloudAreaThickness, 0.1f, 0.1f, 20.0f);
			ImGui::DragFloat("Cloud shadow extent(km)", &earthAtmosphere.cloudShadowExtent, 0.1f, 1.0f, 50.0f);

			ImGui::DragFloat("cloud coverage", &earthAtmosphere.cloudCoverage, 0.1f, 0.0f, 1.0f);
			ImGui::DragFloat("cloud density", &earthAtmosphere.cloudDensity, 0.1f, 0.0f, 1.0f);
			ImGui::DragFloat("cloud fog fade", &earthAtmosphere.cloudFogFade, 0.001f, 0.0f, 0.1f);
			ImGui::DragFloat("cloud max tracing distance", &earthAtmosphere.cloudMaxTraceingDistance, 1.0f, 10.0f, 100.0f);
			ImGui::DragFloat("cloud max tracing start distance", &earthAtmosphere.cloudTracingStartMaxDistance, 1.0f, 300.0f, 500.0f);
			
			ImGui::SliderFloat2("Wether UV scale", &earthAtmosphere.cloudWeatherUVScale.x, 0.0f, 0.01f);
			ImGui::DragFloat("cloud sun light scale", &earthAtmosphere.cloudShadingSunLightScale, 0.1f, 0.1f, 10.0f);
			ImGui::Spacing();
			ImGui::PopItemWidth();
		}

		if (ImGui::CollapsingHeader("Earth Atmosphere"))
		{
			ImGui::PushItemWidth(180.0f);

			if(ImGui::Button("Reset Atmosphere"))
			{
				earthAtmosphere.resetAtmosphere();
				atmosphereHeight = earthAtmosphere.topRadius - earthAtmosphere.bottomRadius; 
			}

			
		
			ImGui::DragFloat("PreExposure",   &earthAtmosphere.atmospherePreExposure, 0.01f, 0.01f, 1.0f);
			ImGui::DragFloat("Mie phase",     &earthAtmosphere.miePhaseFunctionG, 0.0f, 0.01f, 0.99f);
			ImGui::DragFloat("Multi scatter", &earthAtmosphere.multipleScatteringFactor, 0.01f, 0.01f, 1.0f);

			// 
			{
				int minVal = earthAtmosphere.viewRayMarchMinSPP;
				int maxSample = earthAtmosphere.viewRayMarchMaxSPP;
				ImGui::SliderInt("Min SPP", &minVal, 1, 30);
				ImGui::SliderInt("Max SPP", &maxSample, 2, 31);

				earthAtmosphere.viewRayMarchMinSPP = minVal;
				earthAtmosphere.viewRayMarchMaxSPP = maxSample;
			}

			ImGui::DragFloat("Planet radius(km)", &earthAtmosphere.bottomRadius, 100.0f, 8000.0f);
			ImGui::DragFloat("Atmos height(km)", &atmosphereHeight, 10.0f, 150.0f);
			
			ImGui::ColorEdit3("Ground albedo", &earthAtmosphere.groundAlbedo.x);

			ImGui::Spacing();
			ImGui::Separator();

			ImGui::Spacing();

			ImGui::ColorEdit3("MieScattCoeff", &earthAtmosphere.mieScatteringColor.x);
			ImGui::DragFloat("MieScattScale",  &earthAtmosphere.mieScatteringLength, 0.00001f, 0.1f);
			ImGui::ColorEdit3("MieAbsorCoeff", &earthAtmosphere.mieAbsColor.x);
			ImGui::DragFloat("MieAbsorScale",  &earthAtmosphere.mieAbsLength, 0.00001f, 10.0f);
			ImGui::ColorEdit3("RayScattCoeff", &earthAtmosphere.rayleighScatteringColor.x);
			ImGui::DragFloat("RayScattScale",  &earthAtmosphere.rayleighScatterLength, 0.00001f, 10.0f);
			ImGui::ColorEdit3("AbsorptiCoeff", &earthAtmosphere.absorptionColor.x);
			ImGui::DragFloat("AbsorptiScale",  &earthAtmosphere.absorptionLength, 0.00001f, 10.0f);

			// Additional info.
			float earthAtmosphereMieScaleHeight = -1.0f / earthAtmosphere.mieDensity[7];
			float earthAtmosphereRayleighScaleHeight = -1.0f / earthAtmosphere.rayleighDensity[7];

			ImGui::DragFloat("MieScaleHeight", &earthAtmosphereMieScaleHeight, 0.5f, 20.0f);
			ImGui::DragFloat("RayScaleHeight", &earthAtmosphereRayleighScaleHeight, 0.5f, 20.0f);
			
			ImGui::PopItemWidth();

			earthAtmosphere.mieDensity[7] = -1.0f / earthAtmosphereMieScaleHeight;
			earthAtmosphere.rayleighDensity[7] = -1.0f / earthAtmosphereRayleighScaleHeight;
		}

		earthAtmosphere.cloudAreaStartHeight = cloudBottomAltitude + earthAtmosphere.bottomRadius;
		earthAtmosphere.topRadius = earthAtmosphere.bottomRadius + atmosphereHeight;

		ImGui::PopItemWidth();

		ImGui::Unindent();

		comp->changeAtmosphere(earthAtmosphere);
	}

}