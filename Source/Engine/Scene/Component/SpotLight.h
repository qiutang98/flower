#pragma once
#include "Light.h"

namespace Flower
{
	class SpotLightComponent : public LightComponent
	{
		ARCHIVE_DECLARE;

#pragma region SerializeField
		////////////////////////////// Serialize area //////////////////////////////
	protected:

		////////////////////////////// Serialize area //////////////////////////////
#pragma endregion SerializeField

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

