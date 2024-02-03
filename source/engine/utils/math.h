#pragma once

// glm math.
// 0. glm force compute on radians.
// 1. glm vulkan depth force 0 to 1.
// 2. glm enable experimental.
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL

// Common glm headers
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/matrix_decompose.hpp>

namespace engine
{
	// namespace alias to ensure all glm header under this file's macro control.
	namespace math = glm;

	// From https://github.com/TheCherno/Hazel/blob/master/Hazel/src/Hazel/Math/Math.cpp
	extern bool decomposeTransform(const math::mat4& transform, math::vec3& translation, math::vec3& rotation, math::vec3& scale);

	static inline uint32_t alignUp(uint32_t val, uint32_t alignment)
	{
		return (val + alignment - 1) & ~(alignment - 1);
	}
}

