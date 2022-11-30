#include "Pch.h"
#include "../Detail.h"
#include "DrawComponent.h"

using namespace Flower;
using namespace Flower::UI;

void ComponentDrawer::drawPMX(std::shared_ptr<SceneNode> node)
{
	std::shared_ptr<PMXComponent> comp = node->getComponent<PMXComponent>();

}