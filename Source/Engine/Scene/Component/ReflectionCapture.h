#pragma once
#include "../Component.h"

namespace Flower
{
	class ReflectionCaptureComponent : public Component
	{
		ARCHIVE_DECLARE;

#pragma region SerializeField
		////////////////////////////// Serialize area //////////////////////////////
	protected:
		bool m_bUseIBLTexture = true;

		// World space effect radius.
		float m_radius;

		////////////////////////////// Serialize area //////////////////////////////
#pragma endregion SerializeField

	public:
		ReflectionCaptureComponent() = default;
		virtual ~ReflectionCaptureComponent() = default;

		ReflectionCaptureComponent(std::shared_ptr<SceneNode> sceneNode)
			: Component(sceneNode)
		{

		}

	public:
		bool UseIBLTexture() const { return m_bUseIBLTexture; }
		bool setUseIBLTexture(bool bState);

		float getRadius() const { return m_radius; }
		bool setRadius(float in);
	};
}


