#pragma once

#include "Core.h"

namespace Flower
{
	// CPU cache line size, set 64 bytes here.
	constexpr size_t cCPUCacheLineSize = 64;

	constexpr size_t cFloat32Size = sizeof(float);
	static_assert(cFloat32Size == 4);

	constexpr size_t cFloat16Size = sizeof(float) / 2;
	static_assert(cFloat16Size == 2);

	constexpr size_t cFloat64Size = sizeof(double);
	static_assert(cFloat64Size == 8);

	template<typename T> concept StringStreamable = requires(std::stringstream & ss, const T & value) { ss << value; };

	// Fast way to get next power of two from v.
	inline uint32_t getNextPOT(uint32_t v)
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

	// halton sequence compute.
	inline float halton(uint64_t index, uint64_t base)
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
	inline glm::vec2 halton2D(uint64_t index, uint64_t baseA, uint64_t baseB)
	{
		return glm::vec2(halton(index, baseA), halton(index, baseB));
	}

	template <class T> requires StringStreamable<T>
	inline std::string toString(const T& value)
	{
		std::stringstream ss;

		ss << std::fixed << value;
		return ss.str();
	}

	inline unsigned char srgbToLinear(unsigned char inSrgb)
	{
		float srgb = inSrgb / 255.0f;
		srgb = glm::max(6.10352e-5f, srgb);
		float lin = srgb > 0.04045f ? glm::pow(srgb * (1.0f / 1.055f) + 0.0521327f, 2.4f) : srgb * (1.0f / 12.92f);

		return unsigned char(lin * 255.0f);
	}

	inline unsigned char linearToSrgb(unsigned char inlin)
	{
		float lin = inlin / 255.0f;
		if (lin < 0.00313067f) return unsigned char(lin * 12.92f * 255.0f);
		float srgb = glm::pow(lin, (1.0f / 2.4f)) * 1.055f - 0.055f;

		return unsigned char(srgb * 255.0f);
	}

	inline uint32_t getSafeWidthDiv2(uint32_t srcWidth)
	{
		return glm::max(1u, srcWidth / 2);
	}

	template<typename T> inline T divideRoundingUp(T x, T y)
	{
		return (x + y - (T)1) / y;
	}

	template<typename T>
	size_t CRCHash(const T& v)
	{
		return CRC::Calculate(&v, sizeof(T), CRC::CRC_32());
	}

	template<typename T>
	struct CRCHasher
	{
		inline size_t operator()(const T& v) const 
		{
			return CRC::Calculate(&v, sizeof(T), CRC::CRC_32());
		}
	};

	inline size_t hashCombine(size_t lhs, size_t rhs) 
	{
		lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
		return lhs;
	}

	inline void sizeSafeCheck(size_t in, size_t max)
	{
		if (in >= max)
		{
			LOG_WARN("Too much element here, exist {0} elements, but only {1} is safe range.", in, max);
		}
	}

	template<typename T, size_t cLazyFrame>
	class LazyDestroyObject
	{
	private:
		size_t m_tickCount = 0;
		std::array<std::unordered_set<std::shared_ptr<T>>, cLazyFrame> m_container;



	public:
		void insert(std::shared_ptr<T> object)
		{
			m_container[m_tickCount].insert(object);
		}

		void tick()
		{
			m_tickCount ++;
			if (m_tickCount >= cLazyFrame)
			{
				m_tickCount = 0;
			}

			m_container[m_tickCount].clear();
		}

		void releaseAll()
		{
			for (auto& container : m_container)
			{
				container.clear();
			}
		}

		~LazyDestroyObject()
		{
			releaseAll();
		}

	
	};
}