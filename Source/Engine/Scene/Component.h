#pragma once

#include "Pch.h"
#include "../Core/Core.h"
#include "../RuntimeModule.h"

namespace Flower
{
	class SceneNode;

	class Component
	{
		friend class cereal::access;

	protected:
		std::weak_ptr<SceneNode> m_node;

	public:
		Component() = default;
		Component(std::shared_ptr<SceneNode> sceneNode) :
			m_node(sceneNode)
		{

		}

		virtual ~Component() { }

		void setNode(std::weak_ptr<SceneNode> node) 
		{ 
			m_node = node; 
		}

		bool isValid() const { return m_node.lock().get(); }

		std::shared_ptr<SceneNode> getNode() 
		{
			CHECK(isValid());
			return m_node.lock(); 
		}

	public:
		void markDirty();

		virtual void init() { }

		// Tick function should call every frame.
		virtual void tick(const RuntimeModuleTickData& tickData) {} 

		virtual void release() { }
	};
}