#include "Pch.h"
#include "Component.h"
#include "SceneNode.h"
#include "Scene.h"

namespace Flower
{
	void Component::markDirty()
	{
		m_node.lock()->getScene()->setDirty(true);
	}
}