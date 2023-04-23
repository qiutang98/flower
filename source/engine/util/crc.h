#pragma once
#include <cstdint>

namespace engine::crc
{
	extern uint32_t crc32(const void* data, uint32_t Length, uint32_t crc = 0);

	template <typename T>
	inline static uint32_t crc32(const T& data, uint32_t crc = 0)
	{
		return crc32(&data, sizeof(T), crc);
	}
}