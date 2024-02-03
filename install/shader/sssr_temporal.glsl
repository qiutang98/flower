#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_ballot : enable

#include "sssr_common.glsl"

vec3 sampleAverageRadiance(vec2 uv) 
{ 
    return texture(sampler2D(inSSRAverageRadiance, linearClampBorder0000Sampler), uv).xyz;
}

vec3 loadRadiance(ivec2 coord) 
{ 
    return texelFetch(inSSRPrefilterRadiance, coord, 0).xyz;
}

vec3 loadRadianceReprojected(ivec2 coord) 
{ 
    return texelFetch(inSSRReprojectedRadiance, coord, 0).xyz;
}

float loadVariance(ivec2 coord) 
{ 
    return texelFetch(inSSRPrefilterVariance, coord, 0).x;
}

float loadNumSamples(ivec2 coord) 
{ 
    return texelFetch(inPrevSampleCount, coord, 0).x;
}

void storeTemporalAccumulation(ivec2 coord, vec3 radiance, float variance) 
{
    imageStore(SSRTemporalFilterRadiance, coord, vec4(radiance, 0.0));
    imageStore(SSRTemporalfilterVariance, coord, vec4(variance, 0.0, 0.0, 0.0));
}

shared vec3 sharedData[16][16];

vec3 loadFromGroupSharedMemory(ivec2 idx) 
{
    return sharedData[idx.y][idx.x];
}

void storeInGroupSharedMemory(ivec2 idx, vec3 radiance) 
{
    sharedData[idx.y][idx.x] = radiance;
}

struct Moments 
{
    vec3 mean;
    vec3 variance;
};

Moments estimateLocalNeighborhoodInGroup(ivec2 groupThreadId) 
{
    Moments estimate;
    estimate.mean = vec3(0);
    estimate.variance = vec3(0);

    float accumulatedWeight = 0.0;

    // 9x9 Tent.
    for (int j = -kLocalNeighborhoodRadius; j <= kLocalNeighborhoodRadius; ++j) 
    {
        for (int i = -kLocalNeighborhoodRadius; i <= kLocalNeighborhoodRadius; ++i) 
        {
            ivec2 newIdx  = groupThreadId + ivec2(i, j);

            vec3 radiance = loadFromGroupSharedMemory(newIdx);
            float weight  = localNeighborhoodKernelWeight(i) * localNeighborhoodKernelWeight(j);

            accumulatedWeight += weight;
            estimate.mean     += radiance * weight;
            estimate.variance += radiance * radiance * weight;
        }
    }
    estimate.mean     /= accumulatedWeight;
    estimate.variance /= accumulatedWeight;

    estimate.variance = abs(estimate.variance - estimate.mean * estimate.mean);
    return estimate;
}

void loadNeighborhood(ivec2 coord, out vec3 radiance) 
{ 
    radiance = texelFetch(inSSRPrefilterRadiance, coord, 0).xyz;
}

void initializeGroupSharedMemory(ivec2 dispatchThreadId, ivec2 groupThreadId, ivec2 screenSize) 
{
    // Load 16x16 region into shared memory using 4 8x8 blocks.
    ivec2 offset[4] = {ivec2(0, 0), ivec2(8, 0), ivec2(0, 8), ivec2(8, 8)};

    // Intermediate storage registers to cache the result of all loads
    vec3 radiance[4];

    // Start in the upper left corner of the 16x16 region.
    dispatchThreadId -= 4;

    // First store all loads in registers
    for (int i = 0; i < 4; ++i) 
    {
        loadNeighborhood(dispatchThreadId + offset[i], radiance[i]);
    }

    // Then move all registers to groupshared memory
    for (int j = 0; j < 4; ++j) 
    {
        storeInGroupSharedMemory(groupThreadId + offset[j], radiance[j]);
    }
}

vec3 clipAABB(vec3 aabbMin, vec3 aabbMax, vec3 prevSample) 
{
    // Main idea behind clipping - it prevents clustering when neighbor color space
    // is distant from history sample

    // Here we find intersection between color vector and aabb color box

    // Note: only clips towards aabb center
    vec3 aabbCenter = 0.5 * (aabbMax + aabbMin);
    vec3 extentClip = 0.5 * (aabbMax - aabbMin) + 0.001;

    // Find color vector
    vec3 colorVector = prevSample - aabbCenter;

    // Transform into clip space
    vec3 colorVectorClip = colorVector / extentClip;
    // Find max absolute component
    colorVectorClip = abs(colorVectorClip);

    float maxAbsUnit = max(max(colorVectorClip.x, colorVectorClip.y), colorVectorClip.z);

    if (maxAbsUnit > 1.0) 
    {
        return aabbCenter + colorVector / maxAbsUnit; // clip towards color vector
    } 
    else 
    {
        return prevSample; // point is inside aabb
    }
}

void resolveTemporal(ivec2 dispatchThreadId, ivec2 groupThreadId, uvec2 screenSize, float historyClipWeight) 
{
    initializeGroupSharedMemory(dispatchThreadId, groupThreadId, ivec2(screenSize));

    groupMemoryBarrier();
    barrier();

    // Center threads in groupshared memory
    groupThreadId += 4; 

    vec3  newSignal = loadFromGroupSharedMemory(groupThreadId);

    float roughness  = texelFetch(inSSRExtractRoughness, dispatchThreadId, 0).x;
    float newVariance = loadVariance(dispatchThreadId);
    
    if (isGlossyReflection(roughness)) 
    {
        float numSamples = loadNumSamples(dispatchThreadId);
        vec2 uv8 = (vec2(dispatchThreadId.xy) + 0.5) / roundUp8(screenSize);
        vec3 avgRadiance = sampleAverageRadiance(uv8);

        vec3 oldSignal = loadRadianceReprojected(dispatchThreadId);

        Moments localNeighborhood = estimateLocalNeighborhoodInGroup(groupThreadId);

        // Clip history based on the curren local statistics
        vec3 colorStd = (sqrt(localNeighborhood.variance.xyz) + length(localNeighborhood.mean.xyz - avgRadiance)) * historyClipWeight * 1.4;
        localNeighborhood.mean.xyz = mix(localNeighborhood.mean.xyz, avgRadiance, 0.2);

        vec3 radianceMin = localNeighborhood.mean.xyz - colorStd;
        vec3 radianceMax = localNeighborhood.mean.xyz + colorStd;

        vec3 clippedOldSignal = clipAABB(radianceMin, radianceMax, oldSignal.xyz);

        float accumulationSpeed = 1.0 / max(numSamples, 1.0);

        float weight  = (1.0 - accumulationSpeed);

        // Blend with average for small sample count
        newSignal.xyz = mix(newSignal.xyz, avgRadiance, 1.0 / max(numSamples + 1.0f, 1.0));

        // Clip outliers
        {
            vec3 radianceMin = avgRadiance.xyz - colorStd * 1.0;
            vec3 radianceMax = avgRadiance.xyz + colorStd * 1.0;
            newSignal.xyz  = clipAABB(radianceMin, radianceMax, newSignal.xyz);
        }

        // Blend with history
        newSignal = mix(newSignal, clippedOldSignal, weight);
        newVariance = mix(computeTemporalVariance(newSignal.xyz, clippedOldSignal.xyz), newVariance, weight);

        if (any(isinf(newSignal)) || any(isnan(newSignal)) || isinf(newVariance) || isnan(newVariance)) 
        {
            newSignal   = vec3(0.0);
            newVariance = 0.0;
        }
    }
    storeTemporalAccumulation(dispatchThreadId, newSignal, newVariance);
}

layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    uint packedCoords = ssboDenoiseTileList.data[int(gl_WorkGroupID)];

    ivec2 dispatchThreadId = ivec2(packedCoords & 0xffffu, (packedCoords >> 16) & 0xffffu) + ivec2(gl_LocalInvocationID.xy);
    ivec2 dispatchGroupId = dispatchThreadId / 8;

    uvec2 remappedGroupThreadId = remap8x8(gl_LocalInvocationIndex);
    uvec2 remappedDispatchThreadId = dispatchGroupId * 8 + remappedGroupThreadId;

    uvec2 screenSize = textureSize(inDepth, 0);
    resolveTemporal(ivec2(remappedDispatchThreadId), ivec2(remappedGroupThreadId), screenSize, kTemporalStableFactor);
}