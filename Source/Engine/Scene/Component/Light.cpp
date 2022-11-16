#include "Pch.h"
#include "Light.h"
#include "../SceneNode.h"
#include "../Scene.h"

namespace Flower
{
	bool LightComponent::setForward(const glm::vec3& inForward)
	{
		if (m_forward != inForward)
		{
			m_forward = inForward;
			markDirty();
			return true;
		}

		return false;
	}

	bool LightComponent::setColor(const glm::vec3& inColor)
	{
		if (m_color != inColor)
		{
			m_color = inColor;
			markDirty();
			return true;
		}

		return false;
	}

	bool LightComponent::setIntensity(float in)
	{
		if (m_intensity != in)
		{
			m_intensity = in;
			markDirty();
			return true;
		}

		return false;
	}

	glm::vec4 LightComponent::getDirection() const
	{
		if (auto node = m_node.lock())
		{
			const auto worldMatrix = node->getTransform()->getWorldMatrix();
			glm::vec4 forward = glm::vec4(m_forward.x, m_forward.y, m_forward.z, 0.0f);

			return glm::normalize(worldMatrix * forward);
		}

		return glm::vec4(0.0f, -1.0f, 0.0f, 0.0f);
	}
}