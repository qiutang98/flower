#pragma once
#include "Light.h"

namespace Flower
{
	class SpotLightComponent : public LightComponent
	{
		ARCHIVE_DECLARE;

#pragma region SerializeField
		////////////////////////////// Serialize area //////////////////////////////
	public:
		bool bCastShadow = true; // If cast shadow, it will be an importance spot light.
		float innerCone = 0.0f;
		float outerCone = glm::pi<float>() * 0.5f;
		float range = 100.0f;

		////////////////////////////// Serialize area //////////////////////////////
#pragma endregion SerializeField

	public:
		SpotLightComponent() = default;
		virtual ~SpotLightComponent() = default;

		SpotLightComponent(std::shared_ptr<SceneNode> sceneNode)
			: LightComponent(sceneNode)
		{

		}
	};
}

