#version 460

/*
** Physical based render code, develop by engineer: qiutanguu.
*/

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable

#include "IBL_Common.glsl"

// Slow IBL irradiance filter, commonly used for environment ibl pre-filter.

layout (set = 0, binding = 0, rgba16f) uniform imageCube imageCubeEnv;
layout (set = 0, binding = 1) uniform textureCube inHdrCube;

// Common sampler set.
#define COMMON_SAMPLER_SET 1
#include "CommonSamplerSet.glsl"

layout(push_constant) uniform PushConstants
{
    uint convolutionSampleCount; // 4096
    int  updateFaceIndex;
};

shared vec4 sharedColorWeight[64];

// 
layout (local_size_x = 1, local_size_y = 1, local_size_z = 64) in;
void main()
{
    ivec2 cubeSize = imageSize(imageCubeEnv);
    ivec3 cubeCoord = ivec3(ivec2(gl_WorkGroupID.xy), updateFaceIndex);

    const vec2 pixPos = vec2(cubeCoord) + vec2(0.5f);
    const vec2 uv = pixPos / vec2(cubeSize);

    // Get sample direction.
    vec3 N = getSamplingVector(cubeCoord.z, uv);

    // Just accumulate radiance in hem-sphere.
    {
        const uvec2 cubeSize = textureSize(inHdrCube, 0);
        const uint maxCubeDim = max(cubeSize.x, cubeSize.y); // Use max dim to compute texel size.

        // Compute Lod using inverse solid angle and pdf.
        // From Chapter 20.4 Mipmap filtered samples in GPU Gems 3.
        // https://developer.nvidia.com/gpugems/gpugems3/part-iii-rendering/chapter-20-gpu-based-importance-sampling
        const float omegaP = 4 * kPI / (6 * maxCubeDim * maxCubeDim);
        const float K = 4.0f;

        float w = 0.0;
        vec3 color = vec3(0.0);

        const uint kConvolutionSampleCount = convolutionSampleCount;
        int threadSampleCount  = int(kConvolutionSampleCount) / 64;
        int currentThreadStart = int(gl_LocalInvocationIndex) * threadSampleCount;
        int currentThreadEnd   = currentThreadStart + threadSampleCount;

        sharedColorWeight[gl_LocalInvocationIndex] = vec4(0.0f);

        for(uint i = currentThreadStart; i < currentThreadEnd; i++)
        {
            vec2 Xi = hammersley2d(i, kConvolutionSampleCount);
            vec3 sampleDirection = importanceSampleCosine(Xi, N); // sample in Hemisphere

            float NoL = dot(N, sampleDirection);
            if (NoL > 0.0)
            {
                // Compute Lod using inverse solid angle and pdf.
                // From Chapter 20.4 Mipmap filtered samples in GPU Gems 3.
                // https://developer.nvidia.com/gpugems/gpugems3/part-iii-rendering/chapter-20-gpu-based-importance-sampling
                float pdf = NoL / kPI;
                float omegaS = 1.0 / pdf;

                float lod = max(0.0, 0.5 * log2(K * omegaS / omegaP));

                sharedColorWeight[gl_LocalInvocationIndex].xyz += textureLod(samplerCube(inHdrCube, linearClampEdgeSampler), sampleDirection, lod).rgb;
                sharedColorWeight[gl_LocalInvocationIndex].w += 1.0; // NoL
            }
        }
    }

    groupMemoryBarrier();
    barrier();
    if(gl_LocalInvocationIndex < 32)
    {
        sharedColorWeight[gl_LocalInvocationIndex] = sharedColorWeight[gl_LocalInvocationIndex] + sharedColorWeight[gl_LocalInvocationIndex + 32];
    }
    groupMemoryBarrier();
    barrier();
    if(gl_LocalInvocationIndex < 16)
    {
        sharedColorWeight[gl_LocalInvocationIndex] = sharedColorWeight[gl_LocalInvocationIndex] + sharedColorWeight[gl_LocalInvocationIndex + 16];
    }
    if(gl_LocalInvocationIndex < 8)
    {
        sharedColorWeight[gl_LocalInvocationIndex] = sharedColorWeight[gl_LocalInvocationIndex] + sharedColorWeight[gl_LocalInvocationIndex + 8];
    }
    if(gl_LocalInvocationIndex < 4)
    {
        sharedColorWeight[gl_LocalInvocationIndex] = sharedColorWeight[gl_LocalInvocationIndex] + sharedColorWeight[gl_LocalInvocationIndex + 4];
    }
    if(gl_LocalInvocationIndex < 2)
    {
        sharedColorWeight[gl_LocalInvocationIndex] = sharedColorWeight[gl_LocalInvocationIndex] + sharedColorWeight[gl_LocalInvocationIndex + 2];
    }
    if(gl_LocalInvocationIndex < 1)
    {
        sharedColorWeight[gl_LocalInvocationIndex] = sharedColorWeight[gl_LocalInvocationIndex] + sharedColorWeight[gl_LocalInvocationIndex + 1];
        vec3 irradiance = (sharedColorWeight[0].w > 0.0) ? (sharedColorWeight[0].xyz / sharedColorWeight[0].w) : vec3(0);
        imageStore(imageCubeEnv, cubeCoord, vec4(irradiance, 1.0));
    }
}