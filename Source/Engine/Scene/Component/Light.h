#pragma once
#include "../Component.h"

namespace Flower
{
	class LightComponent : public Component
	{
		ARCHIVE_DECLARE;

#pragma region SerializeField
	////////////////////////////// Serialize area //////////////////////////////
	protected:

		glm::vec3 m_color = { 1.0f, 1.0f, 1.0f };
		glm::vec3 m_forward = { 0.0f, -1.0f, 0.0f };

		float m_intensity = 10.0f;

	////////////////////////////// Serialize area //////////////////////////////
#pragma endregion SerializeField

	public:
		LightComponent() = default;
		virtual ~LightComponent() = default;

		LightComponent(std::shared_ptr<SceneNode> sceneNode)
			: Component(sceneNode)
		{

		}

	public:
		glm::vec3 getColor() const 
		{ 
			return m_color; 
		}

		glm::vec3 getForward() const 
		{ 
			return m_forward; 
		}

		float getIntensity() const 
		{ 
			return m_intensity; 
		}

		bool setForward(const glm::vec3& in);

		bool setColor(const glm::vec3& in);

		bool setIntensity(float in);

		glm::vec4 getDirection() const;
		glm::vec3 getPosition() const;
	};
}


