#include "Pch.h"
#include "Detail.h"
#include "DrawComponent.h"

using namespace Flower;
using namespace Flower::UI;

void ComponentDrawer::drawSpotLight(std::shared_ptr<SceneNode> node)
{
	std::shared_ptr<SpotLightComponent> comp = node->getComponent<SpotLightComponent>();

	drawLight(comp);
}