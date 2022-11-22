#pragma once

#include <atomic>
#include <chrono>
#include <filesystem>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <queue>
#include <list>
#include <vector>
#include <stack>
#include <utility>
#include <type_traits>
#include <thread>
#include <mutex>
#include <fstream>
#include <exception>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include <shared_mutex>

// glm math.
// 0. glm force compute on radians.
// 1. glm vulkan depth force 0 to 1.
// 2. glm enable experimental.
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/euler_angles.hpp>

// cereal serialize.
#define CEREAL_THREAD_SAFE 1
#include <cereal/access.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/array.hpp>
#include <cereal/cereal.hpp> 
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/unordered_set.hpp>
#include <cereal/types/list.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/set.hpp>
#include <cereal/types/map.hpp>

namespace glm
{
	template<class Archive> void serialize(Archive& archive, glm::vec2&  v) { archive(v.x, v.y); }
	template<class Archive> void serialize(Archive& archive, glm::vec3&  v) { archive(v.x, v.y, v.z); }
	template<class Archive> void serialize(Archive& archive, glm::vec4&  v) { archive(v.x, v.y, v.z, v.w); }
	template<class Archive> void serialize(Archive& archive, glm::ivec2& v) { archive(v.x, v.y); }
	template<class Archive> void serialize(Archive& archive, glm::ivec3& v) { archive(v.x, v.y, v.z); }
	template<class Archive> void serialize(Archive& archive, glm::ivec4& v) { archive(v.x, v.y, v.z, v.w); }
	template<class Archive> void serialize(Archive& archive, glm::uvec2& v) { archive(v.x, v.y); }
	template<class Archive> void serialize(Archive& archive, glm::uvec3& v) { archive(v.x, v.y, v.z); }
	template<class Archive> void serialize(Archive& archive, glm::uvec4& v) { archive(v.x, v.y, v.z, v.w); }
	template<class Archive> void serialize(Archive& archive, glm::dvec2& v) { archive(v.x, v.y); }
	template<class Archive> void serialize(Archive& archive, glm::dvec3& v) { archive(v.x, v.y, v.z); }
	template<class Archive> void serialize(Archive& archive, glm::dvec4& v) { archive(v.x, v.y, v.z, v.w); }
	template<class Archive> void serialize(Archive& archive, glm::mat2&  m) { archive(m[0], m[1]); }
	template<class Archive> void serialize(Archive& archive, glm::dmat2& m) { archive(m[0], m[1]); }
	template<class Archive> void serialize(Archive& archive, glm::mat3&  m) { archive(m[0], m[1], m[2]); }
	template<class Archive> void serialize(Archive& archive, glm::mat4&  m) { archive(m[0], m[1], m[2], m[3]); }
	template<class Archive> void serialize(Archive& archive, glm::dmat4& m) { archive(m[0], m[1], m[2], m[3]); }
	template<class Archive> void serialize(Archive& archive, glm::quat&  q) { archive(q.x, q.y, q.z, q.w); }
	template<class Archive> void serialize(Archive& archive, glm::dquat& q) { archive(q.x, q.y, q.z, q.w); }
}

// nlohmann json.
#include <nlohmann/json.hpp>

// Use crc as default hash libraries.
#include <crcpp/crc.h>

#define UUID_SYSTEM_GENERATOR
#include <uuid/uuid.h>

// vulkan
#include <vulkan/vulkan.h>

// We use glfw as window manager library.
#include <glfw/glfw3.h>

// ImGui common.
#include <ImGui/ImGui.h>
#include <ImGui/ImGuiGlfw.h>
#include <ImGui/ImGuiVulkan.h>
#include <ImGui/IconsFontAwesome6.h>
#include <ImGui/IconsFontAwesome6Brands.h>
#include <ImGui/ImGuiExtension.h>

template<typename F, typename... Args>
struct RunOnceObject
{
	RunOnceObject(const F& f, Args... args)
	{
		f(std::forward<Args>(args)...);
	}
};

#include <entt/entt.hpp>