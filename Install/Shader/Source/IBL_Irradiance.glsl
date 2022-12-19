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

// 
layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main()
{
    ivec3 cubeCoord = ivec3(gl_GlobalInvocationID);
    ivec2 cubeSize = imageSize(imageCubeEnv);

    if(cubeCoord.x >= cubeSize.x || cubeCoord.y >= cubeSize.y || cubeCoord.z >= 6)
    {
        return;
    }

    const vec2 pixPos = vec2(cubeCoord) + vec2(0.5f);
    const vec2 uv = pixPos / vec2(cubeSize);

    // Get sample direction.
    vec3 N = getSamplingVector(cubeCoord.z, uv);

    // Just accumulate radiance in hem-sphere.
    vec3 irradiance = vec3(0.0);
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

        const uint kConvolutionSampleCount = 4096; // offline accumulate count, XD.
        for(uint i = 0; i < kConvolutionSampleCount; i++)
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

                color += textureLod(samplerCube(inHdrCube, linearClampEdgeSampler), sampleDirection, lod).rgb;
                w += 1.0; // NoL
            }
        }

        irradiance = (w > 0) ? (color / w) : vec3(0.0);
        // irradiance *= kPi; // pre-multi pi for diffuse. Set as one optional.
    }

    imageStore(imageCubeEnv, cubeCoord, vec4(irradiance, 1.0));
    return;
}