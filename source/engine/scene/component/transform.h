#pragma once

#include "../component.h"

namespace engine
{
	class Transform : public Component
	{
		REGISTER_BODY_DECLARE(Component);
	public:
		Transform() = default;
		Transform(std::shared_ptr<SceneNode> sceneNode) : Component(sceneNode) { }

		virtual ~Transform() = default;

		// Interface override.
		virtual void tick(const RuntimeModuleTickData& tickData) override;

		virtual bool uiDrawComponent() override;
		static const UIComponentReflectionDetailed& uiComponentReflection();

		// Getter.
		math::vec3& getTranslation() { return m_translation; }
		const math::vec3& getTranslation() const { return m_translation; }
		math::vec3& getRotation() { return m_rotation; }
		const math::quat& getRotation() const { return m_rotation; }
		math::vec3& getScale() { return m_scale; }
		const math::vec3& getScale() const { return m_scale; }

		// Mark world matrix dirty, also change child's dirty state.
		void invalidateWorldMatrix();

		// setter.
		void setTranslation(const math::vec3& translation);
		void setRotation(const math::vec3& rotation);
		void setScale(const math::vec3& scale);
		void setMatrix(const math::mat4& matrix);

		// Get final world matrix. relate to parent.
		const math::mat4& getWorldMatrix() const { return m_worldMatrix; }

		// Get last tick world matrix result.
		const math::mat4& getPrevWorldMatrix() const { return m_prevWorldMatrix; }

		void updateWorldTransform();

	protected:
		// Compute local matrix.
		math::mat4 computeLocalMatrix() const;

	protected:
		// need update?
		bool m_bUpdateFlag = true;

		// world space matrix.
		math::mat4 m_worldMatrix = math::mat4(1.0);

		// Prev-frame world matrix.
		math::mat4 m_prevWorldMatrix = math::mat4(1.0);

	protected:
		math::vec3 m_translation = { .0f, .0f, .0f };
		math::vec3 m_rotation = { .0f, .0f, .0f };
		math::vec3 m_scale = { 1.f, 1.f, 1.f };
	};
}
