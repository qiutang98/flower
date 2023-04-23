#include "component.h"
#include "scene_node.h"
#include "scene_graph.h"

namespace engine
{
	void Component::setNode(std::weak_ptr<SceneNode> node)
	{
		m_node = node;
	}

	std::shared_ptr<SceneNode> Component::getNode()
	{
		return m_node.lock();
	}

	bool Component::isValid() const
	{
		return m_node.lock().get();
	}

	void Component::markDirty()
	{
		m_node.lock()->getScene()->setDirty(true);
	}
}