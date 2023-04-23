#include "transform.h"
#include "../scene_node.h"

namespace engine
{
	void Transform::tick(const RuntimeModuleTickData& tickData)
	{
		m_prevWorldMatrix = m_worldMatrix;
		updateWorldTransform();
	}

	void Transform::invalidateWorldMatrix()
	{
		m_bUpdateFlag = true;

		// also notify all children their transform dirty.
		auto& children = getNode()->getChildren();
		for (auto& child : children)
		{
			child->getTransform()->invalidateWorldMatrix();
		}
	}

	void Transform::setTranslation(const glm::vec3& translation)
	{
		m_translation = translation;
		invalidateWorldMatrix();
	}

	void Transform::setRotation(const glm::vec3& rotation)
	{
		m_rotation = rotation;
		invalidateWorldMatrix();
	}

	void Transform::setScale(const glm::vec3& scale)
	{
		m_scale = scale;
		invalidateWorldMatrix();
	}

	void Transform::setMatrix(const glm::mat4& matrix)
	{
		if (getNode()->getParent()->isRoot())
		{
			decomposeTransform(matrix, m_translation, m_rotation, m_scale);
			invalidateWorldMatrix();
		}
		else
		{
			const math::mat4 parentInverse = math::inverse(getNode()->getParent()->getTransform()->getWorldMatrix());
			const math::mat4 localNew = parentInverse * matrix;

			decomposeTransform(localNew, m_translation, m_rotation, m_scale);
			invalidateWorldMatrix();
		}


	}

	glm::mat4 Transform::computeLocalMatrix() const
	{
		// TRS - style.
		return 
			math::translate(glm::mat4(1.0f), m_translation) * 
			math::toMat4(glm::quat(m_rotation)) *
			math::scale(glm::mat4(1.0f), m_scale);
	}


	void Transform::updateWorldTransform()
	{
		if (m_bUpdateFlag)
		{
			m_worldMatrix = computeLocalMatrix();
			auto parent = getNode()->getParent();

			// recursive multiply all parent's world matrix.
			if (parent)
			{
				auto transform = parent->getComponent<Transform>();
				m_worldMatrix = transform->getWorldMatrix() * m_worldMatrix;
			}

			m_bUpdateFlag = !m_bUpdateFlag;
		}
	}
}