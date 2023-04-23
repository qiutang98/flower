#include "camera_interface.h"

namespace engine
{
	Frustum Frustum::build(const math::mat4& viewprojMatrix)
	{
		Frustum res{};

		res.planes[LEFT].x = viewprojMatrix[0].w + viewprojMatrix[0].x;
		res.planes[LEFT].y = viewprojMatrix[1].w + viewprojMatrix[1].x;
		res.planes[LEFT].z = viewprojMatrix[2].w + viewprojMatrix[2].x;
		res.planes[LEFT].w = viewprojMatrix[3].w + viewprojMatrix[3].x;

		res.planes[RIGHT].x = viewprojMatrix[0].w - viewprojMatrix[0].x;
		res.planes[RIGHT].y = viewprojMatrix[1].w - viewprojMatrix[1].x;
		res.planes[RIGHT].z = viewprojMatrix[2].w - viewprojMatrix[2].x;
		res.planes[RIGHT].w = viewprojMatrix[3].w - viewprojMatrix[3].x;

		res.planes[TOP].x = viewprojMatrix[0].w - viewprojMatrix[0].y;
		res.planes[TOP].y = viewprojMatrix[1].w - viewprojMatrix[1].y;
		res.planes[TOP].z = viewprojMatrix[2].w - viewprojMatrix[2].y;
		res.planes[TOP].w = viewprojMatrix[3].w - viewprojMatrix[3].y;

		res.planes[DOWN].x = viewprojMatrix[0].w + viewprojMatrix[0].y;
		res.planes[DOWN].y = viewprojMatrix[1].w + viewprojMatrix[1].y;
		res.planes[DOWN].z = viewprojMatrix[2].w + viewprojMatrix[2].y;
		res.planes[DOWN].w = viewprojMatrix[3].w + viewprojMatrix[3].y;

		res.planes[BACK].x = viewprojMatrix[0].w + viewprojMatrix[0].z;
		res.planes[BACK].y = viewprojMatrix[1].w + viewprojMatrix[1].z;
		res.planes[BACK].z = viewprojMatrix[2].w + viewprojMatrix[2].z;
		res.planes[BACK].w = viewprojMatrix[3].w + viewprojMatrix[3].z;

		res.planes[FRONT].x = viewprojMatrix[0].w - viewprojMatrix[0].z;
		res.planes[FRONT].y = viewprojMatrix[1].w - viewprojMatrix[1].z;
		res.planes[FRONT].z = viewprojMatrix[2].w - viewprojMatrix[2].z;
		res.planes[FRONT].w = viewprojMatrix[3].w - viewprojMatrix[3].z;

		for (auto i = 0; i < res.planes.size(); i++)
		{
			float length = sqrtf(
				res.planes[i].x * res.planes[i].x + 
				res.planes[i].y * res.planes[i].y + 
				res.planes[i].z * res.planes[i].z);
			res.planes[i] /= length;
		}

		return res;
	}

	Frustum CameraInterface::getWorldFrustum() const
	{
		const math::vec3 forwardVector = math::normalize(m_front);
		const math::vec3 camWorldPos = m_position;

		const math::vec3 nearC = camWorldPos + forwardVector * m_zNear;
		const math::vec3 farC = camWorldPos + forwardVector * m_zFar;

		const float tanFovyHalf = math::tan(getFovY() * 0.5f);
		const float aspect = getAspect();

		const float yNearHalf = m_zNear * tanFovyHalf;
		const float yFarHalf = m_zFar * tanFovyHalf;

		const math::vec3 yNearHalfV = yNearHalf * m_up;
		const math::vec3 xNearHalfV = yNearHalf * aspect * m_right;

		const math::vec3 yFarHalfV = yFarHalf * m_up;
		const math::vec3 xFarHalfV = yFarHalf * aspect * m_right;

		const math::vec3 NRT = nearC + xNearHalfV + yNearHalfV;
		const math::vec3 NRD = nearC + xNearHalfV - yNearHalfV;
		const math::vec3 NLT = nearC - xNearHalfV + yNearHalfV;
		const math::vec3 NLD = nearC - xNearHalfV - yNearHalfV;

		const math::vec3 FRT = farC + xFarHalfV + yFarHalfV;
		const math::vec3 FRD = farC + xFarHalfV - yFarHalfV;
		const math::vec3 FLT = farC - xFarHalfV + yFarHalfV;
		const math::vec3 FLD = farC - xFarHalfV - yFarHalfV;

		Frustum ret{};

		// p1 X p2, center is pC.
		auto getNormal = [](const math::vec3& pC, const math::vec3& p1, const math::vec3& p2)
		{
			const math::vec3 dir0 = p1 - pC;
			const math::vec3 dir1 = p2 - pC;
			const math::vec3 crossDir = math::cross(dir0, dir1);
			return math::normalize(crossDir);
		};

		// left 
		const math::vec3 leftN = getNormal(FLD, FLT, NLD);
		ret.planes[Frustum::LEFT] = math::vec4(leftN, -math::dot(leftN, FLD));

		// down
		const math::vec3 downN = getNormal(FRD, FLD, NRD);
		ret.planes[Frustum::DOWN] = math::vec4(downN, -math::dot(downN, FRD));

		// right
		const math::vec3 rightN = getNormal(FRT, FRD, NRT);
		ret.planes[Frustum::RIGHT] = math::vec4(rightN, -math::dot(rightN, FRT));

		// top
		const math::vec3 topN = getNormal(FLT, FRT, NLT);
		ret.planes[Frustum::TOP] = math::vec4(topN, -math::dot(topN, FLT));

		// front
		const math::vec3 frontN = getNormal(NRT, NRD, NLT);
		ret.planes[Frustum::FRONT] = math::vec4(frontN, -math::dot(frontN, NRT));

		// back
		const math::vec3 backN = getNormal(FRT, FLT, FRD);
		ret.planes[Frustum::BACK] = math::vec4(backN, -math::dot(backN, FRT));

		return ret;
	}

	void CameraInterface::fillPerframe(GPUPerFrameData& outPerframe)
	{
		outPerframe.camWorldPos = { getPosition(), 1.0f };

		outPerframe.camInfo =
		{
			getFovY(),
			getAspect(),
			getZNear(),
			getZFar()
		};

		outPerframe.camView = getViewMatrix();
		outPerframe.camProjNoJitter = getProjectMatrix();

		auto frustum = getWorldFrustum();

		outPerframe.frustumPlanes[0] = frustum.planes[0];
		outPerframe.frustumPlanes[1] = frustum.planes[1];
		outPerframe.frustumPlanes[2] = frustum.planes[2];
		outPerframe.frustumPlanes[3] = frustum.planes[3];
		outPerframe.frustumPlanes[4] = frustum.planes[4];
		outPerframe.frustumPlanes[5] = frustum.planes[5];
	}
}