#pragma once
#include "Light.h"

namespace engine
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


		
	public:
		ARCHIVE_DECLARE;

		// If cast shadow, it will be an importance spot light.
		bool bCastShadow = true; 
		float innerCone = 0.0f;
		float outerCone = glm::pi<float>() * 0.5f;
		float range = 100.0f;
	};
}

