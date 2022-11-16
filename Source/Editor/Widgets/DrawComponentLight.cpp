#include "Pch.h"
#include "Detail.h"
#include "DrawComponent.h"

using namespace Flower;
using namespace Flower::UI;

void ComponentDrawer::drawLight(std::shared_ptr<LightComponent> comp)
{
	
	glm::vec3 color = comp->getColor();
	ImGui::ColorEdit3("Color", &color[0]);
	comp->setColor(color);


	ImGui::PushItemWidth(100.0f);

	float intensity = comp->getIntensity();
	ImGui::DragFloat("Intensity", &intensity, 0.25f, 0.0f, 20.0f);
	comp->setIntensity(intensity);

	ImGui::PopItemWidth();
}