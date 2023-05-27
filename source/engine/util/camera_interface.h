#pragma once

#include "math.h"
#include "macro.h"

#include <array>
#include "shader_struct.h"

namespace engine
{
	struct Frustum
	{
		std::array<math::vec4, 6> planes;

		enum side
		{
			LEFT  = 0,
			DOWN  = 1,
			RIGHT = 2,
			TOP   = 3,
			FRONT = 4,
			BACK  = 5
		};

		// Build frustum from viewproject matrix.
		static Frustum build(const math::mat4& viewprojMatrix);
	};

	class CameraInterface
	{
	public:
		virtual void fillPerframe(GPUPerFrameData& outPerframe);

		// return camera aspect.
		float getAspect() const { return (float)m_width / (float)m_height; }

		// return camera fov y.
		float getFovY() const { return m_fovy; }
		void setFovY(float v) { m_fovy = v; }
		// return camera z near plane.
		float getZNear() const { return m_zNear; }

		// return camera z far plane.
		float getZFar() const { return m_zFar; }

		// return camera view matrix.
		virtual math::mat4 getViewMatrix() const = 0;

		// return camera project matrix.
		virtual math::mat4 getProjectMatrix() const = 0;

		// return camera worldspcae position.
		const math::vec3& getPosition() const { return m_position; }

		Frustum getWorldFrustum() const;

	protected:
		// world space position.
		math::vec3 m_position = { 25.0f, 25.0f, 25.0f};

		// fov y.
		float m_fovy = math::radians(45.0f);

		// z near.
		float m_zNear = 0.001f;

		// z far.
		float m_zFar = 10'000.0f;

		// render width.
		size_t m_width;

		// render height.
		size_t m_height;

		// camera front direction.
		math::vec3 m_front = { -0.5f, -0.6f, 0.6f };

		// camera up direction.
		math::vec3 m_up;

		// camera right direction;
		math::vec3 m_right;
	};
}