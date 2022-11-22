#pragma once
#include "../Component.h"

namespace Flower
{
	class PMXComponent : public Component
	{
		friend class cereal::access;

	public:
		PMXComponent() = default;
		virtual ~PMXComponent() = default;

		PMXComponent(std::shared_ptr<SceneNode> sceneNode)
			: Component(sceneNode)
		{

		}

	public:


	};
}