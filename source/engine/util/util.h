#pragma once

#include "allocator.h"
#include "cacheline.h"
#include "cvars.h"
#include "delegate.h"
#include "log.h"
#include "macro.h"
#include "math.h"
#include "noncopyable.h"
#include "uuid.h"
#include "framework.h"
#include "engine.h"
#include "window_data.h"
#include "threadpool.h"
#include "timer.h"
#include "keycode.h"
#include "config.h"
#include "mesh_misc.h"
#include "shader_struct.h"

#include <string>
#include <sstream>
#include <cstdint>
#include <fstream>
#include <map>
#include <set>
#include <vector>
#include <array>
#include <unordered_map>
#include <memory>
#include <stdexcept>
#include <mutex>
#include <thread>
#include <filesystem>

#include <utf8/cpp17.h>

#include <stb/stb_image.h>
#include <stb/stb_image_write.h>
#include <stb/stb_image_resize.h>

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

#define ARCHIVE_DECLARE \
	friend class cereal::access; \
	template<class Archive> \
	void serialize(Archive& archive, std::uint32_t const version);

#define ARCHIVE_NVP(Ar, Member) Ar(cereal::make_nvp(#Member, Member))
#define ARCHIVE_NVP_DEFAULT(Member) archive(cereal::make_nvp(#Member, Member))

#ifndef ARCHIVE_BINARY
	#define ARCHIVE_BINARY_USED 0
#else
	#define ARCHIVE_BINARY_USED 1
#endif

template<typename T>
inline static void archiveFunctionJsonOutput(const std::filesystem::path& path, const T& o)
{
	std::ofstream os(path);
	cereal::JSONOutputArchive archive(os);
	archive(o);
}

template<typename T>
inline static void archiveFunctionBinOutput(const std::filesystem::path& path, const T& o)
{
	std::ofstream os(path, std::ios::binary);
	cereal::BinaryOutputArchive archive(os);
	archive(o);
}

template<typename T>
inline static void archiveFunctionBinInput(const std::filesystem::path& path, T& o)
{
	std::ifstream is(path, std::ios::binary);
	cereal::BinaryInputArchive archive(is);
	archive(o);
}

template<typename T>
inline static void archiveFunctionJsonInput(const std::filesystem::path& path, T& o)
{
	std::ifstream is(path);
	cereal::JSONInputArchive archive(is);
	archive(o);
}

template<typename T>
inline static void archiveFunctionOutput(const std::filesystem::path& path, const T& o)
{
#if ARCHIVE_BINARY_USED
	archiveFunctionBinOutput(path, o);
#else
	archiveFunctionJsonOutput(path, o);
#endif
}

template<typename T>
inline static void archiveFunctionInput(const std::filesystem::path& path, T& o)
{
#if ARCHIVE_BINARY_USED
	archiveFunctionBinInput(path, o);
#else
	archiveFunctionJsonInput(path, o);
#endif
}

namespace glm
{
	template<class Archive> void serialize(Archive& archive, glm::vec2& v) { archive(v.x, v.y); }
	template<class Archive> void serialize(Archive& archive, glm::vec3& v) { archive(v.x, v.y, v.z); }
	template<class Archive> void serialize(Archive& archive, glm::vec4& v) { archive(v.x, v.y, v.z, v.w); }
	template<class Archive> void serialize(Archive& archive, glm::ivec2& v) { archive(v.x, v.y); }
	template<class Archive> void serialize(Archive& archive, glm::ivec3& v) { archive(v.x, v.y, v.z); }
	template<class Archive> void serialize(Archive& archive, glm::ivec4& v) { archive(v.x, v.y, v.z, v.w); }
	template<class Archive> void serialize(Archive& archive, glm::uvec2& v) { archive(v.x, v.y); }
	template<class Archive> void serialize(Archive& archive, glm::uvec3& v) { archive(v.x, v.y, v.z); }
	template<class Archive> void serialize(Archive& archive, glm::uvec4& v) { archive(v.x, v.y, v.z, v.w); }
	template<class Archive> void serialize(Archive& archive, glm::dvec2& v) { archive(v.x, v.y); }
	template<class Archive> void serialize(Archive& archive, glm::dvec3& v) { archive(v.x, v.y, v.z); }
	template<class Archive> void serialize(Archive& archive, glm::dvec4& v) { archive(v.x, v.y, v.z, v.w); }
	template<class Archive> void serialize(Archive& archive, glm::mat2& m) { archive(m[0], m[1]); }
	template<class Archive> void serialize(Archive& archive, glm::dmat2& m) { archive(m[0], m[1]); }
	template<class Archive> void serialize(Archive& archive, glm::mat3& m) { archive(m[0], m[1], m[2]); }
	template<class Archive> void serialize(Archive& archive, glm::mat4& m) { archive(m[0], m[1], m[2], m[3]); }
	template<class Archive> void serialize(Archive& archive, glm::dmat4& m) { archive(m[0], m[1], m[2], m[3]); }
	template<class Archive> void serialize(Archive& archive, glm::quat& q) { archive(q.x, q.y, q.z, q.w); }
	template<class Archive> void serialize(Archive& archive, glm::dquat& q) { archive(q.x, q.y, q.z, q.w); }
}

namespace engine
{
	// Fast way to get next power of two from v.
	static inline uint32_t getNextPOT(uint32_t v)
	{
		v--;
		v |= v >> 1;
		v |= v >> 2;
		v |= v >> 4;
		v |= v >> 8;
		v |= v >> 16;
		v++;
		return v;
	}

	static inline int32_t getNextPOT(int32_t vi)
	{
		uint32_t v = (uint32_t)vi;
		return (int32_t)getNextPOT(v);
	}

	// halton sequence compute.
	static inline float halton(uint64_t index, uint64_t base)
	{
		float f = 1; float r = 0;
		while (index > 0)
		{
			f = f / static_cast<float>(base);
			r = r + f * (index % base);
			index = index / base;
		}
		return r;
	}

	// halton 2d sequence compute.
	static inline math::vec2 halton2D(uint64_t index, uint64_t baseA, uint64_t baseB)
	{
		return math::vec2(halton(index, baseA), halton(index, baseB));
	}

	static inline unsigned char srgbToLinear(unsigned char inSrgb)
	{
		float srgb = inSrgb / 255.0f;
		srgb = math::max(6.10352e-5f, srgb);
		float lin = srgb > 0.04045f ? math::pow(srgb * (1.0f / 1.055f) + 0.0521327f, 2.4f) : srgb * (1.0f / 12.92f);

		return unsigned char(lin * 255.0f);
	}

	static inline unsigned char linearToSrgb(unsigned char inlin)
	{
		float lin = inlin / 255.0f;
		if (lin < 0.00313067f) return unsigned char(lin * 12.92f * 255.0f);
		float srgb = math::pow(lin, (1.0f / 2.4f)) * 1.055f - 0.055f;

		return unsigned char(srgb * 255.0f);
	}

	// Safe div 2 and never smaller than 1.
	static inline uint32_t getSafeDiv2(uint32_t src)
	{
		return math::max(1u, src / 2);
	}

	template<typename T> inline T divideRoundingUp(T x, T y)
	{
		return (x + y - (T)1) / y;
	}

	// Boost hash combine.
	static inline size_t hashCombine(size_t lhs, size_t rhs)
	{
		lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
		return lhs;
	}
	
	// Boost hash combine.
	template <class T>
	inline void hashCombine(std::size_t& seed, const T& v)
	{
		std::hash<T> hasher;
		seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
	}

	// Find "true" in the str and return bool result, no accurate.
	static inline bool stringToBoolApprox(const std::string& str)
	{
		return str.find("true") != std::string::npos;
	}

	static inline bool isPOT(uint32_t n)
	{
		return (n & (n - 1)) == 0;
	}

	struct EnumClassHash
	{
		template <typename T>
		std::size_t operator()(T t) const
		{
			return static_cast<std::size_t>(t);
		}
	};
}