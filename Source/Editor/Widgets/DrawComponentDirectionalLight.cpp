#include "Pch.h"
#include "Detail.h"
#include "DrawComponent.h"

using namespace Flower;
using namespace Flower::UI;

void ComponentDrawer::drawDirectionalLight(std::shared_ptr<SceneNode> node)
{
	std::shared_ptr<DirectionalLightComponent> comp = node->getComponent<DirectionalLightComponent>();
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

}