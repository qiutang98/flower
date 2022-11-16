#include "Pch.h"
#include "Transform.h"
#include "../SceneNode.h"
#include "../Component.h"
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace Flower
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

	void Transform::setRotation(const glm::quat& rotation)
	{
		m_rotation = rotation;
		invalidateWorldMatrix();
	}

	void Transform::setScale(const glm::vec3& scale)
	{
		m_scale = scale;
		invalidateWorldMatrix();
	}

	const glm::vec3& Transform::getTranslation() const
	{
		return m_translation;
	}

	glm::vec3& Transform::getTranslation()
	{
		return m_translation;
	}

	const glm::quat& Transform::getRotation() const
	{
		return m_rotation;
	}

	glm::quat& Transform::getRotation()
	{
		return m_rotation;
	}

	const glm::vec3& Transform::getScale() const
	{
		return m_scale;
	}

	glm::vec3& Transform::getScale()
	{
		return m_scale;
	}

	void Transform::setMatrix(const glm::mat4& matrix)
	{
		glm::vec3 skew;
		glm::vec4 perspective;
		glm::decompose(matrix, m_scale, m_rotation, m_translation, skew, perspective);
		m_rotation = glm::conjugate(m_rotation);
	}

	glm::mat4 Transform::getMatrix() const
	{
		// TRS - style 
		glm::mat4 scale = glm::scale(m_scale);
		glm::mat4 rotator = glm::mat4_cast(m_rotation);
		glm::mat4 translate = glm::translate(glm::mat4(1.0f), m_translation);

		// maybe i should cache result instead of calculate every call.
		return translate * rotator * scale;
	}

	glm::mat4 Transform::getWorldMatrix()
	{
		return m_worldMatrix;
	}

	glm::mat4 Transform::getPrevWorldMatrix()
	{
		return m_prevWorldMatrix;
	}

	void Transform::updateWorldTransform()
	{
		if (m_bUpdateFlag)
		{
			m_worldMatrix = getMatrix();
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