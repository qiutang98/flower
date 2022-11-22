#pragma once
#include "../Component.h"

namespace Flower
{
	class SceneNode;
	class LandscapeComponent : public Component
	{
		friend class cereal::access;

	public:
		LandscapeComponent() = default;
		virtual ~LandscapeComponent() = default;

		LandscapeComponent(std::shared_ptr<SceneNode> sceneNode)
			: Component(sceneNode)
		{

		}

	public:
		virtual void tick(const RuntimeModuleTickData& tickData) override;
	};
}