#pragma once
#include "Pch.h"

namespace Flower
{
	// bloom prefilter cureve.
	inline glm::vec4 getBloomPrefilter(float threshold, float thresholdSoft)
	{
		float knee = threshold * thresholdSoft;
		glm::vec4 prefilter{ };

		prefilter.x = threshold;
		prefilter.y = prefilter.x - knee;
		prefilter.z = 2.0f * knee;
		prefilter.w = 0.25f / (knee + 0.00001f);

		return prefilter;
	}
}