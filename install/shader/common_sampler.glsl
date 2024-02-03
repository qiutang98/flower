#ifndef COMMON_SAMPLER_GLSL_202309272219
#define COMMON_SAMPLER_GLSL_202309272219

#ifdef SHARED_SAMPLER_SET

// These samplers' mipmapMode are point. Commonly use for RT read.
layout(set = SHARED_SAMPLER_SET, binding = 0) uniform sampler pointClampEdgeSampler;
layout(set = SHARED_SAMPLER_SET, binding = 1) uniform sampler pointClampBorder0000Sampler;
layout(set = SHARED_SAMPLER_SET, binding = 2) uniform sampler pointRepeatSampler;
layout(set = SHARED_SAMPLER_SET, binding = 3) uniform sampler linearClampEdgeSampler;
layout(set = SHARED_SAMPLER_SET, binding = 4) uniform sampler linearClampBorder0000Sampler;
layout(set = SHARED_SAMPLER_SET, binding = 5) uniform sampler linearRepeatSampler;
layout(set = SHARED_SAMPLER_SET, binding = 6) uniform sampler linearClampBorder1111Sampler;
layout(set = SHARED_SAMPLER_SET, binding = 7) uniform sampler pointClampBorder1111Sampler;

// With mip filter
layout(set = SHARED_SAMPLER_SET, binding = 8) uniform sampler linearClampEdgeMipFilterSampler;
layout(set = SHARED_SAMPLER_SET, binding = 9) uniform sampler linearRepeatMipFilterSampler;

#endif // SHARED_SAMPLER_SET

const uint kBlueNoiseDim = 128;

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

#endif // COMMON_SAMPLER_GLSL_202309272219