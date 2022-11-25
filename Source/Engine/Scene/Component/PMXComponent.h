#pragma once
#include "../Component.h"

namespace Flower
{
	class PMXComponent : public Component
	{
		ARCHIVE_DECLARE;

#pragma region SerializeField
		////////////////////////////// Serialize area //////////////////////////////
	protected:

		////////////////////////////// Serialize area //////////////////////////////
#pragma endregion SerializeField

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
