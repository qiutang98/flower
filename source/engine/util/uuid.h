#pragma once

#include <string>
#include <cstdint>

namespace engine
{
	// Standard string uuid.
	// String uuid maybe slow than uint128, but current speed is enough.
	using UUID = std::string;
	[[nodiscard]] extern UUID buildUUID();

	// Random device guid, faster than UUID.
	using UUID64u = uint64_t;
	[[nodiscard]] extern UUID64u buildRuntimeUUID64u();
}