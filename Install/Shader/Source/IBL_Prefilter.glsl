#version 460

/*
** Physical based render code, develop by engineer: qiutanguu.
*/

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_shader_subgroup_basic : enable


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
    int samplesCount;         // 1024
    float maxBrightValue;     // 1000.0f
    float filterRoughnessMin; // 0.05f
    int updateFaceIndex;
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

    /////////////////////////////////////////////////////////////////////////////////////////////
    // Filter start.
    vec3 V = getSamplingVector(cubeCoord.z, uv);
    vec3 N = V;

    // Pre-return by sample src if
    const bool bFirstThreadInGroup = (gl_LocalInvocationIndex == 0);
    if(alphaRoughness < filterRoughnessMin)
    {
        vec3 sampleColor = textureLod(samplerCube(inHdrCube, linearClampEdgeSampler), N, 0).xyz;

        // NOTE: Smooth surface accumulate may cause flicking hightlight, so don't evaluate this.
        if(bFirstThreadInGroup)
        {
            imageStore(imageCubeEnv, cubeCoord, vec4(sampleColor, 1.0));
        }
        return;
    }

    const uvec2 maxHdrCubeSize = textureSize(inHdrCube, 0);

    // Solid angle associated with a single cubemap texel at zero mipmap level.
	// This will come in handy for importance sampling below.
	float maxCubeSize = max(float(maxHdrCubeSize.x), float(maxHdrCubeSize.y));
    float omegaP = 4.f * kPI / (6.f * maxCubeSize * maxCubeSize);

    // Filter surface roughness.
    float r = alphaRoughness; 
    
    sharedColorWeight[gl_LocalInvocationIndex] = vec4(0.0);

    // Per thread sample count.
    int threadSampleCount  = samplesCount / 64;
    int currentThreadStart = int(gl_LocalInvocationIndex) * threadSampleCount;
    int currentThreadEnd   = currentThreadStart + threadSampleCount;
    for(int i = currentThreadStart; i < threadSampleCount; i++)
    {
        vec2 xi = hammersley2d(i, samplesCount); 
        vec3 H = importanceSampleGGX(xi, r, N);
        vec3 L = normalize(2.f * dot(V, H ) * H - V); 

        float NoL = max(dot(N, L), 0.f);
        float NoH = max(dot(N, H), 0.f);
        float VoH = max(dot(V, H), 0.f);

        if(NoL > 0.f)
        {
            float K = 4.f;
            float pdf = max(D_GGX(NoH, r) * NoH / (VoH * 4.f), 0.001f);
            float omegaS = 1.f / (pdf * samplesCount);
            
            // log_4(2) = 2, as log_2(x) / log_2(y) = log_y(x) we can use log2(x) * 0.5 = log4(x) 
            float lod = max(0.0, log2(K * omegaS / omegaP) * 0.5f);

            sharedColorWeight[gl_LocalInvocationIndex].xyz += min(textureLod(samplerCube(inHdrCube, linearClampEdgeSampler), L, lod).rgb, maxBrightValue) * NoL;
            sharedColorWeight[gl_LocalInvocationIndex].w   += NoL; 
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