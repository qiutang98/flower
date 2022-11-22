#pragma once
#include "../Pch.h"

namespace Flower
{
	// Standard string uuid.
	// String uuid maybe slow than uint128, but current speed is enough.
	using UUID = std::string;
	inline UUID buildUUID()
	{
		return uuids::to_string(uuids::uuid_system_generator{}());
	}

	// Random device guid.
	using UUID_64u = uint64_t;
	inline UUID_64u buildRuntimeUUID_64u()
	{
		static std::random_device s_randomDevice;
		static std::mt19937_64 s_engine(s_randomDevice());
		static std::uniform_int_distribution<uint64_t> s_uniformDistribution;

		return s_uniformDistribution(s_engine);
	}
}