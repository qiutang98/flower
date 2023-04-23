#include "light.h"
#include "scene/scene_archive.h"

namespace engine
{
	bool LightComponent::setForward(const math::vec3& inForward)
	{
		if (m_forward != inForward)
		{
			m_forward = inForward;
			markDirty();
			return true;
		}

		return false;
	}

	bool LightComponent::setColor(const math::vec3& inColor)
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

	math::vec3 LightComponent::getDirection() const
	{
		if (auto node = m_node.lock())
		{
			const auto worldMatrix = node->getTransform()->getWorldMatrix();
			math::vec4 forward = math::vec4(m_forward.x, m_forward.y, m_forward.z, 0.0f);

			return math::normalize(math::vec3(worldMatrix * forward));
		}

		return math::vec3(0.0f, -1.0f, 0.0f);
	}

	math::vec3 LightComponent::getPosition() const
	{
		if (auto node = m_node.lock())
		{
			const auto worldMatrix = node->getTransform()->getWorldMatrix();
			math::vec4 posOrign = math::vec4(0.0f, 0.0f, 0.0f, 1.0f);

			return math::vec3(worldMatrix * posOrign);
		}

		return math::vec3(0.0f, 0.0f, 0.0f);
	}
}