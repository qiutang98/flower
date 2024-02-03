#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <cstdint>
#include <filesystem>
#include <string>

#include "allocator.h"
#include "cacheline.h"
#include "cvars.h"
#include "delegate.h"
#include "log.h"
#include "math.h"
#include "noncopyable.h"
#include "threadpool.h"
#include "uuid.h"
#include "lru.h"
#include "glfw.h"
#include "base.h"
#include "timer.h"
#include "crc.h"

#pragma warning(disable : 4996)

#include <utfcpp/utf8.h>
#include <utfcpp/utf8/cpp17.h>

#include <rttr/registration.h>
#include <rttr/type.h>

#define ENABLE_LOG

#if defined(_DEBUG) || defined(DEBUG)
	#define APP_DEBUG
#endif

#ifdef ENABLE_LOG
	#define LOG_TRACE(...) { ::engine::LoggerSystem::get()->getDefaultLogger()->trace   (__VA_ARGS__); }
	#define LOG_INFO(...)  { ::engine::LoggerSystem::get()->getDefaultLogger()->info    (__VA_ARGS__); }
	#define LOG_WARN(...)  { ::engine::LoggerSystem::get()->getDefaultLogger()->warn    (__VA_ARGS__); }
	#define LOG_ERROR(...) { ::engine::LoggerSystem::get()->getDefaultLogger()->error   (__VA_ARGS__); }
	#define LOG_FATAL(...) { ::engine::LoggerSystem::get()->getDefaultLogger()->critical(__VA_ARGS__); throw std::runtime_error("Utils fatal!"); }
#else
	#define LOG_TRACE(...)   
	#define LOG_INFO (...)    
	#define LOG_WARN(...)   
	#define LOG_ERROR(...)    
	#define LOG_FATAL(...) { throw std::runtime_error("Utils fatal!"); }
#endif

#ifdef APP_DEBUG
	#define CHECK(x) { if(!(x)) { LOG_FATAL("Check error, at line {0} on file {1}.", __LINE__, __FILE__); __debugbreak(); } }
	#define ASSERT(x, ...) { if(!(x)) { LOG_FATAL("Assert failed: {2}, at line {0} on file {1}.", __LINE__, __FILE__, __VA_ARGS__); __debugbreak(); } }
#else
	#define CHECK(x) { if(!(x)) { LOG_FATAL("Check error."); } }
	#define ASSERT(x, ...) { if(!(x)) { LOG_FATAL("Assert failed: {0}.", __VA_ARGS__); } }
#endif

#define  CHECK_ENTRY() ASSERT(false, "Should not entry here, fix me!")
#define UN_IMPLEMENT() ASSERT(false, "Un-implement yet, fix me!")

struct __engine_ConstructOnceObject_Log
{
	enum class ELogType
	{
		Trace,
		Info,
		Warn,
		Error,
	};

	explicit __engine_ConstructOnceObject_Log(const std::string& in, ELogType type)
	{
		switch (type)
		{
		case ELogType::Trace:
		{
			LOG_TRACE(in);
			return;
		}
		case ELogType::Info:
		{
			LOG_INFO(in);
			return;
		}
		case ELogType::Warn:
		{
			LOG_WARN(in);
		}
		case ELogType::Error:
		{
			LOG_ERROR(in);
#ifdef APP_DEBUG
			__debugbreak();
#endif
			return;
		}
		}

		CHECK_ENTRY();
	}
};

#define LOG_TRACE_ONCE(str) { static __engine_ConstructOnceObject_Log __local_trace(str, __engine_ConstructOnceObject_Log::ELogType::Trace); }
#define LOG_INFO_ONCE(str)  { static __engine_ConstructOnceObject_Log __local_info(str,  __engine_ConstructOnceObject_Log::ELogType::Info);  }
#define LOG_WARN_ONCE(str)  { static __engine_ConstructOnceObject_Log __local_warn(str,  __engine_ConstructOnceObject_Log::ELogType::Warn);  }
#define LOG_ERROR_ONCE(str) { static __engine_ConstructOnceObject_Log __local_error(str, __engine_ConstructOnceObject_Log::ELogType::Error); }

#define UN_IMPLEMENT_WARN() LOG_ERROR_ONCE("Logic still un-implement!")

namespace engine
{
	using u8str = std::string;

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

	// Convert light temperature to bt.709 linear rgb color gamut, also suitable for linear srgb color space.
	static inline math::vec3 temperature2Color(float temp)
	{
		temp = math::clamp(temp, 1000.0f, 15000.0f);

		// Approximate Planckian locus in CIE 1960 UCS
		float u =
			(0.860117757f + 1.54118254e-4f * temp + 1.28641212e-7f * temp * temp) /
			(1.0f + 8.42420235e-4f * temp + 7.08145163e-7f * temp * temp);

		float v =
			(0.317398726f + 4.22806245e-5f * temp + 4.20481691e-8f * temp * temp) /
			(1.0f - 2.89741816e-5f * temp + 1.61456053e-7f * temp * temp);

		float x = 3.0f * u / (2.0f * u - 8.0f * v + 4.0f);
		float y = 2.0f * v / (2.0f * u - 8.0f * v + 4.0f);
		float z = 1.0f - x - y;

		float Y = 1.0f;
		float X = Y / y * x;
		float Z = Y / y * z;

		// XYZ to RGB with BT.709 primaries
		float R = 3.2404542f * X + -1.5371385f * Y + -0.4985314f * Z;
		float G = -0.9692660f * X + 1.8760108f * Y + 0.0415560f * Z;
		float B = 0.0556434f * X + -0.2040259f * Y + 1.0572252f * Z;

		return math::vec3(R, G, B);
	}

	inline static std::string buildRelativePathUtf8(
		const std::filesystem::path& shortPath,
		const std::filesystem::path& longPath)
	{
		const std::u16string shortPath16 = std::filesystem::absolute(shortPath).u16string();
		const std::u16string longPath16 = std::filesystem::absolute(longPath).u16string();

		auto result = utf8::utf16to8(longPath16.substr(shortPath16.size()));

		if (result.starts_with("\\") || result.starts_with("/"))
		{
			result = result.erase(0, 1);
		}

		return result;
	}

	inline static std::filesystem::path buildStillNonExistPath(const std::filesystem::path& p)
	{
		const std::u16string rawPath = p.u16string();
		std::u16string pUnique = rawPath;
		size_t num = 1;

		while (std::filesystem::exists(pUnique))
		{
			pUnique = rawPath + utf8::utf8to16(std::format("_{}", num));
			num ++;
		}

		return pUnique;
	}

	template<typename T>
	class RegisterManager
	{
	public:
		void add(T& in) 
		{ 
			m_registers.push_back(in); 
		}

		bool remove(const T& in)
		{
			size_t i = 0;
			for (auto& iter : m_registers)
			{
				if (iter == in)
				{
					break;
				}

				i++;
			}

			if (i >= m_registers.size())
			{
				return false;
			}

			m_registers[i] = std::move(m_registers.back());
			m_registers.pop_back();

			return true;
		}

		void loop(std::function<void(T& r)>&& f)
		{
			for (auto& iter : m_registers)
			{
				f(iter);
			}
		}

		void clear()
		{
			m_registers.clear();
		}

	private:
		std::vector<T> m_registers;
	};

	inline static math::vec3 convertSRGBColorSpace(math::vec3 color)
	{
		static const math::mat3 sRGB_2_XYZ_MAT = math::mat3
		(
			math::vec3(0.4124564, 0.3575761, 0.1804375),
			math::vec3(0.2126729, 0.7151522, 0.0721750),
			math::vec3(0.0193339, 0.1191920, 0.9503041)
		);

		static const math::mat3 XYZ_2_AP1_MAT = math::mat3
		(
			math::vec3(1.6410233797, -0.3248032942, -0.2364246952),
			math::vec3(-0.6636628587, 1.6153315917, 0.0167563477),
			math::vec3(0.0117218943, -0.0082844420, 0.9883948585)
		);

		// D65 to D60 White Point
		static const math::mat3 D65_2_D60_CAT = math::mat3
		(
			math::vec3(1.01303, 0.00610531, -0.014971),
			math::vec3(0.00769823, 0.998165, -0.00503203),
			math::vec3(-0.00284131, 0.00468516, 0.924507)
		);

		static const math::mat3 sRGB_2_AP1 = (sRGB_2_XYZ_MAT * D65_2_D60_CAT) * XYZ_2_AP1_MAT;

		return color * sRGB_2_AP1;
	}
}