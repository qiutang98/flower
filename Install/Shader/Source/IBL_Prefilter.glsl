#version 460

/*
** Physical based render code, develop by engineer: qiutanguu.
*/

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable

#include "IBL_Common.glsl"

// Slow IBL specular filter, commonly used for environment ibl pre-filter.

layout (set = 0, binding = 0, rgba16f) uniform imageCube imageCubeEnv;
layout (set = 0, binding = 1) uniform textureCube inHdrCube;

// Common sampler set.
#define COMMON_SAMPLER_SET 1
#include "CommonSamplerSet.glsl"

layout(push_constant) uniform PushConstants
{
	float alphaRoughness;
};

const int kSamplesCount = 1024;
const float kMaxBrightValue = 10.0f; // maximum value avoids fireflies caused by very bright pixels, .r .g .b compare.
const float kFilterRoughnessMin = 0.05f;


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

    /////////////////////////////////////////////////////////////////////////////////////////////
    // Filter start.
    vec3 V = getSamplingVector(cubeCoord.z, uv);
    vec3 N = V;

    // Pre-return by sample src if
    if(alphaRoughness < kFilterRoughnessMin)
    {
        vec3 sampleColor = textureLod(samplerCube(inHdrCube, linearClampEdgeSampler), N, 0).xyz;
        imageStore(imageCubeEnv, cubeCoord, vec4(sampleColor, 1.0));
        return;
    }

    const uvec2 maxHdrCubeSize = textureSize(inHdrCube, 0);

    // Solid angle associated with a single cubemap texel at zero mipmap level.
	// This will come in handy for importance sampling below.
	float maxCubeSize = max(float(maxHdrCubeSize.x), float(maxHdrCubeSize.y));
    float omegaP = 4.f * kPI / (6.f * maxCubeSize * maxCubeSize);

    // Filter surface roughness.
    float r = alphaRoughness; 
    

    vec3 color = vec3(0.f);
    float weight = 0.f;

    for(int i = 0; i < kSamplesCount; i++)
    {
        vec2 xi = hammersley2d(i, kSamplesCount); 
        vec3 H = importanceSampleGGX(xi, r, N);
        vec3 L = normalize(2.f * dot(V, H ) * H - V); 

        float NoL = max(dot(N, L), 0.f);
        float NoH = max(dot(N, H), 0.f);
        float VoH = max(dot(V, H), 0.f);

        if(NoL > 0.f)
        {
            float K = 4.f;
            float pdf = max(D_GGX(NoH, r) * NoH / (VoH * 4.f), 0.001f);
            float omegaS = 1.f / (pdf * kSamplesCount);
            
            // log_4(2) = 2, as log_2(x) / log_2(y) = log_y(x) we can use log2(x) * 0.5 = log4(x) 
            float lod = max(0.0, log2(K * omegaS / omegaP) * 0.5f);

            color += min(textureLod(samplerCube(inHdrCube, linearClampEdgeSampler), L, lod).rgb, kMaxBrightValue) * NoL;
            weight += NoL; 
        }
    }
    
    vec3 irradiance = (weight > 0.0) ? (color / weight) : vec3(0);

    imageStore(imageCubeEnv, cubeCoord, vec4(irradiance, 1.0));
    return;
}