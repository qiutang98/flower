#pragma once
#include "Light.h"

namespace Flower
{
	class SpotLightComponent : public LightComponent
	{
	public:
		SpotLightComponent() = default;
		virtual ~SpotLightComponent() = default;

		SpotLightComponent(std::shared_ptr<SceneNode> sceneNode)
			: LightComponent(sceneNode)
		{

		}

	protected:

	};
}