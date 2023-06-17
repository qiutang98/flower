#pragma once

#include "../component.h"

namespace engine
{
	class LightComponent : public Component
	{
	public:
		LightComponent() = default;
		virtual ~LightComponent() = default;

		LightComponent(std::shared_ptr<SceneNode> sceneNode)
			: Component(sceneNode)
		{

		}

	public:
		const math::vec3& getColor() const { return m_color; }
		const math::vec3& getForward() const { return m_forward; }
		float getIntensity() const { return m_intensity; }

		// Forward of this node, maybe should set in one forward component.
		bool setForward(const math::vec3& in);
		bool setColor(const math::vec3& in);
		bool setIntensity(float in);

		math::vec3 getDirection() const;
		math::vec3 getPosition() const;

		bool isRayTraceShadow() const { return m_bRayTraceShadow; }
		bool setRayTraceShadow(bool bState);
		
	protected:
		ARCHIVE_DECLARE;

		math::vec3 m_color = { 1.0f, 1.0f, 1.0f };
		math::vec3 m_forward = { 0.0f, -1.0f, 0.0f };
		float m_intensity = 1.0f;
		bool m_bRayTraceShadow = false;
	};
}


