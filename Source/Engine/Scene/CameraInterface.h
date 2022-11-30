#pragma once
#include "Pch.h"

namespace Flower
{
	class CameraInterface
	{
		friend class cereal::access;

	protected:
		// world space position.
		glm::vec3 m_position = { 0.0f, 10.0f, 0.0f};

		// fov y.
		float m_fovy = glm::radians(45.0f);

		// z near.
		float m_zNear = 0.1f;

		// z far.
		float m_zFar = 10'000.0f;

		// render width.
		size_t m_width = GMinRenderDim;

		// render height.
		size_t m_height = GMinRenderDim;

		// camera front direction.
		glm::vec3 m_front = { 0.0f, 0.0f, 1.0f };

		// camera up direction.
		glm::vec3 m_up;

		// camera right direction;
		glm::vec3 m_right;

	public:
		float atmosphereHeightOffset = 0.5f; // km.

		float atmosphereMoveScale = 0.0f; // 

		float aperture = 10.0f;        // Size of the lens diaphragm (mm). Controls depth of field and chromatic aberration.
		float shutterSpeed = 12.0f; // Length of time for which the camera shutter is open (sec). Also controls the amount of motion blur.
		float iso = 800.0f;       // Sensitivity to light.
		float exposureCompensation = 0.0f;

		// Reference: https://google.github.io/filament/Filament.md.html#lighting/units/lightunitsvalidation
		float getEv100()    const 
		{ 
			return std::log2((aperture * aperture) / shutterSpeed * 100.0f / iso);
		} 

		// Frostbite: https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
		// https://docs.unrealengine.com/4.27/en-US/RenderingAndGraphics/PostProcessEffects/AutomaticExposure/
		float getExposure() const 
		{ 
			return 1.0f / (std::pow(2.0f, getEv100() + exposureCompensation));
		} 

		struct Frustum
		{
			std::array<glm::vec4, 6> planes;
		};

		Frustum getWorldFrustum() const
		{
			const glm::vec3 forwardVector = glm::normalize(m_front);
			const glm::vec3 camWorldPos = m_position;

			const glm::vec3 nearC = camWorldPos + forwardVector * m_zNear;
			const glm::vec3 farC = camWorldPos + forwardVector * m_zFar;

			const float tanFovyHalf = glm::tan(getFovY() * 0.5f);
			const float aspect = getAspect();

			const float yNearHalf = m_zNear * tanFovyHalf;
			const float yFarHalf = m_zFar * tanFovyHalf;

			const glm::vec3 yNearHalfV = yNearHalf * m_up;
			const glm::vec3 xNearHalfV = yNearHalf * aspect * m_right;

			const glm::vec3 yFarHalfV = yFarHalf * m_up;
			const glm::vec3 xFarHalfV = yFarHalf * aspect * m_right;

			const glm::vec3 NRT = nearC + xNearHalfV + yNearHalfV;
			const glm::vec3 NRD = nearC + xNearHalfV - yNearHalfV;
			const glm::vec3 NLT = nearC - xNearHalfV + yNearHalfV;
			const glm::vec3 NLD = nearC - xNearHalfV - yNearHalfV;

			const glm::vec3 FRT = farC + xFarHalfV + yFarHalfV;
			const glm::vec3 FRD = farC + xFarHalfV - yFarHalfV;
			const glm::vec3 FLT = farC - xFarHalfV + yFarHalfV;
			const glm::vec3 FLD = farC - xFarHalfV - yFarHalfV;

			Frustum ret{};

			// p1 X p2, center is pC.
			auto getNormal = [](const glm::vec3& pC, const glm::vec3& p1, const glm::vec3& p2)
			{
				const glm::vec3 dir0 = p1 - pC;
				const glm::vec3 dir1 = p2 - pC;
				const glm::vec3 crossDir = glm::cross(dir0, dir1);
				return glm::normalize(crossDir);
			};

			// left 
			const glm::vec3 leftN = getNormal(FLD, FLT, NLD);
			ret.planes[0] = glm::vec4(leftN, -glm::dot(leftN, FLD));

			// down
			const glm::vec3 downN = getNormal(FRD, FLD, NRD);
			ret.planes[1] = glm::vec4(downN, -glm::dot(downN, FRD));

			// right
			const glm::vec3 rightN = getNormal(FRT, FRD, NRT);
			ret.planes[2] = glm::vec4(rightN, -glm::dot(rightN, FRT));

			// top
			const glm::vec3 topN = getNormal(FLT, FRT, NLT);
			ret.planes[3] = glm::vec4(topN, -glm::dot(topN, FLT));

			// front
			const glm::vec3 frontN = getNormal(NRT, NRD, NLT);
			ret.planes[4] = glm::vec4(frontN, -glm::dot(frontN, NRT));

			// back
			const glm::vec3 backN = getNormal(FRT, FLT, FRD);
			ret.planes[5] = glm::vec4(backN, -glm::dot(backN, FRT));

			return ret;
		}

		// return camera worldspcae position.
		glm::vec3 getPosition() const
		{
			return m_position;
		}

		// return camera view matrix.
		virtual glm::mat4 getViewMatrix() const = 0;

		// return camera project matrix.
		virtual glm::mat4 getProjectMatrix() const = 0;

		// return camera aspect.
		float getAspect() const
		{
			return (float)m_width / (float)m_height;
		}

		// return camera fov y.
		float getFovY() const
		{
			return m_fovy;
		}

		// return camera z near plane.
		float getZNear() const
		{
			return m_zNear;
		}

		// return camera z far plane.
		float getZFar() const
		{
			return m_zFar;
		}
	};
}