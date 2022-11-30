#pragma once
#include "../Component.h"

namespace Flower
{
	class SkyLightComponent : public Component
	{
		ARCHIVE_DECLARE;

#pragma region SerializeField
		////////////////////////////// Serialize area //////////////////////////////
	protected:

		bool m_bRealtimeCapture = true;

		////////////////////////////// Serialize area //////////////////////////////
#pragma endregion SerializeField

	public:
		SkyLightComponent() = default;
		virtual ~SkyLightComponent() = default;

		SkyLightComponent(std::shared_ptr<SceneNode> sceneNode)
			: Component(sceneNode)
		{

		}

	public:
		bool isRealtimeCapture() const { return m_bRealtimeCapture; }
		bool setRealtimeCapture(bool bState);
	};
}


