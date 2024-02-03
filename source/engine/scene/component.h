#pragma once

#include "scene_common.h"
#include <iconFontcppHeaders/IconsFontAwesome6Brands.h>

namespace engine
{
	struct UIComponentReflectionDetailed
	{
		bool bOptionalCreated = false;
		std::string iconCreated = ICON_FA_NODE;
	};

	class Component
	{
		REGISTER_BODY_DECLARE();

	public:
		Component() = default;
		Component(std::shared_ptr<SceneNode> sceneNode) : m_node(sceneNode) { }

		virtual ~Component() = default;

		virtual bool uiDrawComponent() { return false; }
		static const UIComponentReflectionDetailed& uiComponentReflection();

		// Interface.
		virtual void init() { }
		virtual void onGameBegin() { }
		virtual void onGamePause() { }
		virtual void onGameContinue() { }
		virtual void onGameStop() { }
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
		// Component host node.
		std::weak_ptr<SceneNode> m_node;
	};

	class RenderableComponent : public Component
	{
		REGISTER_BODY_DECLARE(Component);

	public:
		RenderableComponent() = default;
		RenderableComponent(std::shared_ptr<SceneNode> sceneNode) : Component(sceneNode) { }
		virtual ~RenderableComponent() = default;

	public:

	};
}

