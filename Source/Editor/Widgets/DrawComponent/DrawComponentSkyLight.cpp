#include "Pch.h"
#include "../Detail.h"
#include "DrawComponent.h"

using namespace Flower;
using namespace Flower::UI;

void ComponentDrawer::drawSkyLight(std::shared_ptr<SceneNode> node)
{
	std::shared_ptr<SkyLightComponent> comp = node->getComponent<SkyLightComponent>();

}