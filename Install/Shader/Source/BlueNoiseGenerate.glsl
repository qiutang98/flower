#version 460

/*
** Physical based render code, develop by engineer: qiutanguu.
*/

#extension GL_GOOGLE_include_directive : enable

// Smapler blue noise from Eric Heitz at Siggraph 2019 unity course.

layout(set = 0, binding = 0, rg8) uniform image2D blueNoiseImage; // Output image.

#define BLUE_NOISE_BUFFER_SET 1
#include "BlueNoiseCommon.glsl"

#include "Common.glsl"
layout (set = 2, binding = 0) uniform UniformView { ViewData viewData; };
layout (set = 3, binding = 0) uniform UniformFrame { FrameData frameData; };

#define COMMON_SAMPLER_SET 4
#include "CommonSamplerSet.glsl"

layout(push_constant) uniform constants
{   
    uint sampleId; // sample id for spp, 1 spp is always 0. maybe shader don't care.
                   //                    2 spp is 0 1
                   //                    4 spp is 0 1 2 3
                   //                    8 spp is 0 1 2 3 4 5 6 7
};


//
// 1d 128 x 128 tile blue noise generator. rotate by frame index and golden radio.
layout (local_size_x = 8, local_size_y = 8) in;
void main() 
{
    float x = samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y, 0, 0u);
    float y = samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y, 0, 1u);

    const float rotate = (frameData.frameIndex.x & 0xFFu) * 1.61803398875;
    x = fract(x + rotate);
    y = fract(y + rotate);

	imageStore(blueNoiseImage, ivec2(gl_GlobalInvocationID.xy), vec4(x, y, 0, 0));
}