#include "component.h"
#include "scene_node.h"
#include "scene.h"

namespace engine
{
	const UIComponentReflectionDetailed& 
	Component::uiComponentReflection()
	{
		static UIComponentReflectionDetailed reflection { };
		return reflection;
	}

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
		m_node.lock()->getScene()->markDirty();
	}
}