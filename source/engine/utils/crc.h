#pragma once
#include <cstdint>

namespace engine::crc
{
	// Crc memory hash based on data's memory, so, you must **ensure** struct init with ** T a = {}; **.
	// 
	extern uint32_t crc32(const void* data, uint32_t length, uint32_t crc = 0);
}