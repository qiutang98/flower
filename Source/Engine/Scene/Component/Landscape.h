#pragma once
#include "../Component.h"

namespace Flower
{
	class SceneNode;
	class LandscapeComponent : public Component
	{
		ARCHIVE_DECLARE;

#pragma region SerializeField
	////////////////////////////// Serialize area //////////////////////////////
	protected:

	////////////////////////////// Serialize area //////////////////////////////
#pragma endregion SerializeField

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

