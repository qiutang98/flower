#include "Pch.h"
#include "../Detail.h"
#include "DrawComponent.h"

using namespace Flower;
using namespace Flower::UI;

void ComponentDrawer::drawLandscape(std::shared_ptr<SceneNode> node)
{
	std::shared_ptr<LandscapeComponent> comp = node->getComponent<LandscapeComponent>();
}