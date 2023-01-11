#pragma once

#include <glm/glm.hpp>
#include <array>
#include <iostream>

static inline glm::vec2 vogelDiskSample(uint32_t sampleIndex, uint32_t sampleCount, float angle)
{
	const float goldenAngle = 2.399963f;

	float r = sqrt(float(sampleIndex) + 0.5f) / sqrt(float(sampleCount));
	float theta = sampleIndex * goldenAngle + angle;

	float sine = sin(theta);
	float cosine = cos(theta);

	return glm::vec2(cosine, sine) * r;
}

static inline void dofGenerateSamplePoints80()
{
    std::array<glm::vec2, 64> p64 = {};
    std::array<glm::vec2, 16> p16 = {};

    int32_t idx64 = 0;
    int32_t idx16 = 0;

    for (int32_t j = 0; j < 80; ++j)
    {
        glm::vec2 sample = vogelDiskSample(j, 80, 0.0f);

        if (j % 5 == 0)
        {
            p16[idx16] = sample;
            ++idx16;
        }
        else
        {
            p64[idx64] = sample;
            ++idx64;
        }
    }

    for (int32_t j = 0; j < 64; ++j)
    {
        std::cout << "vec2(" << p64[j].x << "," << p64[j].y << ")," << std::endl;
    }

    std::cout << std::endl;

    for (int32_t j = 0; j < 16; ++j)
    {
        std::cout << "vec2(" << p16[j].x << "," << p16[j].y << ")," << std::endl;
    }
}