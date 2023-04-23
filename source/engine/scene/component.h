#pragma once

#include <util/util.h>
#include <asset/asset_archive.h>

namespace engine
{
	class SceneNode;

	class Component
	{
	public:
		Component() = default;
		Component(std::shared_ptr<SceneNode> sceneNode) : m_node(sceneNode) { }

		virtual ~Component() = default;

		// Interface.
		virtual void init() { }
		virtual void tick(const RuntimeModuleTickData& tickData) {}
		virtual void release() { }

		// Change owner node.
		void setNode(std::weak_ptr<SceneNode> node);

		// Get owner node.
		std::shared_ptr<SceneNode> getNode();

		// Component is valid or not.
		bool isValid() const;

		// Mark dirty.
		void markDirty();

	protected:
		ARCHIVE_DECLARE;
		
		// Component host node.
		std::weak_ptr<SceneNode> m_node;
	};
}

