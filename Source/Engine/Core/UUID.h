#pragma once
#include "../Pch.h"

namespace Flower
{
	using UUID = std::string;

	inline UUID buildUUID()
	{
		return uuids::to_string(uuids::uuid_system_generator{}());
	}
}