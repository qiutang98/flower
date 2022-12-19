#ifndef BLUE_NOISE_COMMON_GLSL
#define BLUE_NOISE_COMMON_GLSL

/*
** Physical based render code, develop by engineer: qiutanguu.
*/

const uint kBlueNoiseDim = 128;

#ifdef BLUE_NOISE_TEXTURE_SET
    layout(set = BLUE_NOISE_TEXTURE_SET, binding = 0) uniform texture2D spp_1_blueNoise; // 1 spp rotate by frame index and golden radio.
#endif

#ifdef BLUE_NOISE_BUFFER_SET
    layout(set = BLUE_NOISE_BUFFER_SET, binding = 0) readonly buffer sobolBuffer { uint sobol_256spp_256d[]; };
    layout(set = BLUE_NOISE_BUFFER_SET, binding = 1) readonly buffer rankingTileBuffer { uint rankingTile[]; };
    layout(set = BLUE_NOISE_BUFFER_SET, binding = 2) readonly buffer scramblingTileBuffer { uint scramblingTile[]; };

    float samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d(uint pixel_i, uint pixel_j, uint sampleIndex, uint sampleDimension)
    {
        // wrap arguments
        pixel_i = pixel_i & 127u;
        pixel_j = pixel_j & 127u;
        sampleIndex = sampleIndex & 255u;
        sampleDimension = sampleDimension & 255u;

        // xor index based on optimized ranking
        uint rankedSampleIndex = sampleIndex ^ rankingTile[sampleDimension + (pixel_i + pixel_j * 128u) * 8u];

        // fetch value in sequence
        uint value = sobol_256spp_256d[sampleDimension + rankedSampleIndex * 256u];

        // If the dimension is optimized, xor sequence value based on optimized scrambling
        value = value ^ scramblingTile[(sampleDimension%8) + (pixel_i + pixel_j * 128u) * 8u];

        // convert to float and return
        float v = (0.5f + value) / 256.0f;

        return v;
    }
#endif

#endif