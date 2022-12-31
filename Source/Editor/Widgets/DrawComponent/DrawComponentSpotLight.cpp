#include "Pch.h"
#include "../Detail.h"
#include "DrawComponent.h"

using namespace Flower;
using namespace Flower::UI;

void ComponentDrawer::drawSpotLight(std::shared_ptr<SceneNode> node)
{
	std::shared_ptr<SpotLightComponent> comp = node->getComponent<SpotLightComponent>();

	drawLight(comp);

	ImGui::Separator();

	ImGui::Checkbox("Cast Shadow", &comp->bCastShadow);

	ImGui::DragFloat("Inner Cone", &comp->innerCone, 0.1, 0.0, comp->outerCone);
	ImGui::DragFloat("Outer Cone", &comp->outerCone, 0.1, comp->innerCone, glm::pi<float>() * 0.5f);
	ImGui::DragFloat("Range", &comp->range, 1.0f, 0.0f);
}