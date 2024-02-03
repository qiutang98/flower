#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_ballot : enable

#include "sssr_common.glsl"

vec3 loadRadiance(ivec2 coords)
{
    return vec3(texelFetch(inSSRIntersection, coords, 0).xyz);
}

float loadDepth(ivec2 coords) 
{
    return texelFetch(inDepth, coords, 0).x;
}

float sampleDepthHistory(vec2 uv) 
{ 
    return texture(sampler2D(inPrevDepth, linearClampBorder1111Sampler), uv).r; 
}

float loadDepthHistory(ivec2 coords) 
{ 
    return texelFetch(inPrevDepth, coords, 0).x;
}

vec3 sampleRadianceHistory(vec2 uv)
{
    return vec3(texture(sampler2D(inPrevSSRRadiance, linearClampBorder0000Sampler), uv).rgb);
}

vec3 loadRadianceHistory(ivec2 coords) 
{ 
    return vec3(texelFetch(inPrevSSRRadiance, coords, 0).rgb); 
}


vec3 sampleWorldSpaceNormalHistory(vec2 uv)
{
    return normalize(vec3(texture(sampler2D(inPrevGbufferB, linearClampBorder0000Sampler), uv).rgb));
}

vec3 loadWorldSpaceNormalHistory(ivec2 coords) 
{
    return vec3(texelFetch(inPrevGbufferB, coords, 0).rgb);
}

float sampleVarianceHistory(vec2 uv)
{
    return texture(sampler2D(inSSRVarianceHistory, linearClampBorder0000Sampler), uv).r; 
}

float sampleNumSamplesHistory(vec2 uv)
{
    return texture(sampler2D(inPrevSampleCount, linearClampBorder0000Sampler), uv).r; 
}

void storeRadianceReprojected(ivec2 coord, vec3 value) 
{ 
    imageStore(SSRReprojectedRadiance, coord, vec4(value, 1.0));
}

void storeAverageRadiance(ivec2 coord, vec3 value) 
{ 
    imageStore(SSRAverageRadiance, coord, vec4(value, 1.0));
}

void storeVariance(ivec2 coord, float value) 
{ 
    imageStore(SSRVariance, coord, vec4(value, 0.0, 0.0, 0.0));
}

void storeNumSamples(ivec2 coord, float value) 
{ 
    imageStore(SSRSampleCount, coord, vec4(value, 0.0, 0.0, 0.0));
}

// 16x16 tile in 8x8 group.
shared vec4 sharedData[16][16];

// Radiance load.
vec3 loadFromGroupSharedMemory(ivec2 idx) 
{
    return (sharedData[idx.y][idx.x]).xyz;
}

// Radiance store.
void storeInGroupSharedMemory(ivec2 idx, vec3 radiance) 
{
    (sharedData[idx.y][idx.x]).xyz = radiance;
}

// Radiance and variance store.
void storeInGroupSharedMemory(ivec2 idx, vec4 radianceVariance) 
{
    sharedData[idx.y][idx.x] = radianceVariance;
}

vec4 loadFromGroupSharedMemoryRaw(ivec2 idx) 
{
    return sharedData[idx.y][idx.x];
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
        radiance[i] = loadRadiance(dispatchThreadId + offset[i]);
    }

    // Then move all registers to groupshared memory
    for (int j = 0; j < 4; ++j) 
    {
        storeInGroupSharedMemory(groupThreadId + offset[j], radiance[j]);
    }
}

// 8x8 downsample luminance weight.
float getLuminanceWeight(vec3 val) 
{
    float luma = luminanceSSR(val.xyz);
    float weight = max(exp(-luma * kAverageRadianceLuminanceWeight), 1.0e-2);

    return weight;
}

vec2 getSurfaceReprojection(vec2 uv, vec2 motionVector) 
{
    // See staticMeshGbuffer.glsl
    vec2 historyUv = uv + motionVector;

    return historyUv;
}

float getDisocclusionFactor(vec3 normal, vec3 historyNormal, float linearDepth, float historyLinearDepth) 
{
    return 
        exp(-abs(1.0 - max(0.0, dot(normal, historyNormal))) * kDisocclusionNormalWeight) *
        exp(-abs(historyLinearDepth - linearDepth) / linearDepth * kDisocclusionDepthWeight);
}

vec2 getHitPositionReprojection(ivec2 dispatchThreadId, vec2 uv, float reflectedRayLength) 
{
    float z = loadDepth(dispatchThreadId);

    // Viewspace ray position.
    vec3 viewSpaceRay = getViewPos(uv, z, frameData);

    // We start out with reconstructing the ray length in view space.
    // This includes the portion from the camera to the reflecting surface as well as the portion from the surface to the hit position.
    float surfaceDepth = length(viewSpaceRay);
    float rayLength = surfaceDepth + reflectedRayLength;

    // We then perform a parallax correction by shooting a ray
    // of the same length "straight through" the reflecting surface
    // and reprojecting the tip of that ray to the previous frame.
    viewSpaceRay /= surfaceDepth; // == normalize(viewSpaceRay)
    viewSpaceRay *= rayLength;

    // This is the "fake" hit position if we would follow the ray straight through the surface.
    vec3 worldHitPosition = (frameData.camInvertView * vec4(viewSpaceRay, 1.0)).xyz; 
    
    // Project to prev frame position.
    vec3 prevHitPosition = projectPos(worldHitPosition, frameData.camViewProjPrev);

    return prevHitPosition.xy;
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

    // 9x9 tent.
    float accumulatedWeight = float(0);
    for (int j = -kLocalNeighborhoodRadius; j <= kLocalNeighborhoodRadius; ++j) 
    {
        for (int i = -kLocalNeighborhoodRadius; i <= kLocalNeighborhoodRadius; ++i) 
        {
            // TODO: Optimize. Can pre-compute.
            float weight = localNeighborhoodKernelWeight(i) * localNeighborhoodKernelWeight(j);

            ivec2 newIdx  = groupThreadId + ivec2(i, j);
            vec3 radiance = loadFromGroupSharedMemory(newIdx);

            // Accumulate.
            accumulatedWeight += weight;
            estimate.mean += radiance * weight;
            estimate.variance += radiance * radiance * weight;
        }
    }

    // Weight mean.
    estimate.mean /= accumulatedWeight;

    // Variance compute.
    estimate.variance /= accumulatedWeight;
    estimate.variance = abs(estimate.variance - estimate.mean * estimate.mean);

    return estimate;
}

void pickReprojection(
    ivec2 dispatchThreadId,
    ivec2 groupThreadId, 
    uvec2 screenSize,   
    float roughness,
    float rayLength,
    out float disocclusionFactor,
    out vec2 reprojectionUV, 
    out vec3 reprojection) 
{
    Moments localNeighborhood = estimateLocalNeighborhoodInGroup(groupThreadId);
    vec2 uv = vec2(dispatchThreadId.x + 0.5, dispatchThreadId.y + 0.5) / screenSize;

    vec3 normal = unpackWorldNormal(texelFetch(inGbufferB, dispatchThreadId, 0).rgb);

    vec3 historyNormal;
    float historyLinearDepth;
    {
        const vec2 motionVector = texelFetch(inGbufferV, dispatchThreadId, 0).rg; 

        // Then get surface prev-frame uv.
        const vec2 surfaceReprojectionUV = getSurfaceReprojection(uv, motionVector);

        // Compute prev-frame hit uv.
        const vec2 hitReprojectionUV = getHitPositionReprojection(dispatchThreadId, uv, rayLength);

        // linear sample surface normal and hit normal. from prev-frame.
        const vec3 surfaceNormal = sampleWorldSpaceNormalHistory(surfaceReprojectionUV);
        const vec3 hitNormal = sampleWorldSpaceNormalHistory(hitReprojectionUV);

        // linear sample radiance from prev-frame.
        const vec3 surfaceHistory = sampleRadianceHistory(surfaceReprojectionUV);
        const vec3 hitHistory = sampleRadianceHistory(hitReprojectionUV);

        // Compute normal similarity.
        const float surfaceNormalSimilarity = dot(normalize(vec3(surfaceNormal)), normalize(vec3(normal)));
        const float hitNormalSimilarity = dot(normalize(vec3(hitNormal)), normalize(vec3(normal)));
        
        // linear sample roughness from prev-frame.
        const float surfaceRoughness = float(texture(sampler2D(inPrevSSRExtractRoughness, linearClampBorder0000Sampler), surfaceReprojectionUV).r);
        const float hitRoughness = float(texture(sampler2D(inPrevSSRExtractRoughness, linearClampBorder0000Sampler), hitReprojectionUV).r);
        
        // Choose reprojection uv based on similarity to the local neighborhood.
        if (hitNormalSimilarity > kReprojectionNormalSimilarityThreshold  // Candidate for mirror reflection parallax
            && (hitNormalSimilarity + 1.0e-3) > surfaceNormalSimilarity    
            && abs(hitRoughness - roughness) < abs(surfaceRoughness - roughness) + 1.0e-3
        ) 
        {
            historyNormal = hitNormal;

            float hitHistoryDepth = sampleDepthHistory(hitReprojectionUV);
            float hitHistoryLinearDepth = linearizeDepthPrev(hitHistoryDepth, frameData);

            historyLinearDepth = hitHistoryLinearDepth;
            reprojectionUV = hitReprojectionUV;
            reprojection = hitHistory;
        } 
        else 
        {
            // Reject surface reprojection based on simple distance
            vec3 surfaceHistoryDiff = surfaceHistory - localNeighborhood.mean;
            if (dot(surfaceHistoryDiff, surfaceHistoryDiff) < kReprojectSurfaceDiscardVarianceWeight * length(localNeighborhood.variance)) 
            {
                historyNormal = surfaceNormal;

                float surfaceHistoryDepth = sampleDepthHistory(surfaceReprojectionUV);
                float surfaceHistoryLinearDepth = linearizeDepthPrev(surfaceHistoryDepth, frameData);

                historyLinearDepth = surfaceHistoryLinearDepth;
                reprojectionUV = surfaceReprojectionUV;
                reprojection = surfaceHistory;
            } 
            else 
            {
                disocclusionFactor = 0.0;
                return;
            }
        }
    }

    float depth = loadDepth(dispatchThreadId);
    float linearDepth = linearizeDepth(depth, frameData);

    // Determine disocclusion factor based on history
    disocclusionFactor = getDisocclusionFactor(normal, historyNormal, linearDepth, historyLinearDepth);

    if (disocclusionFactor > kDisocclusionThreshold) // Early out, good enough
    {
        return;
    }

    // Try to find the closest sample in the vicinity if we are not convinced of a disocclusion
    if (disocclusionFactor < kDisocclusionThreshold) 
    {
        vec2 closestUv = reprojectionUV;
        vec2 dudv = 1.0 / vec2(screenSize);

        const int kSearchRadius = 1;
        for (int y = -kSearchRadius; y <= kSearchRadius; y++) 
        {
            for (int x = -kSearchRadius; x <= kSearchRadius; x++) 
            {
                vec2 uv = reprojectionUV + vec2(x, y) * dudv;

                vec3 historyNormal = sampleWorldSpaceNormalHistory(uv);
                float historyDepth = sampleDepthHistory(uv);

                float historyLinearDepth = linearizeDepthPrev(historyDepth, frameData);

                float weight = getDisocclusionFactor(normal, historyNormal, linearDepth, historyLinearDepth);
                if (weight > disocclusionFactor) 
                {
                    disocclusionFactor = weight;
                    closestUv = uv;
                    reprojectionUV = closestUv;
                }
            }
        }
        reprojection = sampleRadianceHistory(reprojectionUV);
    }

    // Rare slow path - triggered only on the edges.
    // Try to get rid of potential leaks at bilinear interpolation level.
    if (disocclusionFactor < kDisocclusionThreshold)
    {
        // If we've got a discarded history, try to construct a better sample out of 2x2 interpolation neighborhood
        // Helps quite a bit on the edges in movement
        float uvx = fract(float(screenSize.x) * reprojectionUV.x + 0.5);
        float uvy = fract(float(screenSize.y) * reprojectionUV.y + 0.5);

        ivec2 reprojectTexelCoords = ivec2(screenSize * reprojectionUV - 0.5);

        vec3 reprojection00 = loadRadianceHistory(reprojectTexelCoords + ivec2(0, 0));
        vec3 reprojection10 = loadRadianceHistory(reprojectTexelCoords + ivec2(1, 0));
        vec3 reprojection01 = loadRadianceHistory(reprojectTexelCoords + ivec2(0, 1));
        vec3 reprojection11 = loadRadianceHistory(reprojectTexelCoords + ivec2(1, 1));

        vec3 normal00 = loadWorldSpaceNormalHistory(reprojectTexelCoords + ivec2(0, 0));
        vec3 normal10 = loadWorldSpaceNormalHistory(reprojectTexelCoords + ivec2(1, 0));
        vec3 normal01 = loadWorldSpaceNormalHistory(reprojectTexelCoords + ivec2(0, 1));
        vec3 normal11 = loadWorldSpaceNormalHistory(reprojectTexelCoords + ivec2(1, 1));

        float depth00 = linearizeDepthPrev(loadDepthHistory(reprojectTexelCoords + ivec2(0, 0)), frameData);
        float depth10 = linearizeDepthPrev(loadDepthHistory(reprojectTexelCoords + ivec2(1, 0)), frameData);
        float depth01 = linearizeDepthPrev(loadDepthHistory(reprojectTexelCoords + ivec2(0, 1)), frameData);
        float depth11 = linearizeDepthPrev(loadDepthHistory(reprojectTexelCoords + ivec2(1, 1)), frameData);

        vec4 w = vec4(1.0);

        // Initialize with occlusion weights
        w.x = getDisocclusionFactor(normal, normal00, linearDepth, depth00) > (kDisocclusionThreshold / 2.0) ? 1.0 : 0.0;
        w.y = getDisocclusionFactor(normal, normal10, linearDepth, depth10) > (kDisocclusionThreshold / 2.0) ? 1.0 : 0.0;
        w.z = getDisocclusionFactor(normal, normal01, linearDepth, depth01) > (kDisocclusionThreshold / 2.0) ? 1.0 : 0.0;
        w.w = getDisocclusionFactor(normal, normal11, linearDepth, depth11) > (kDisocclusionThreshold / 2.0) ? 1.0 : 0.0;
        
        // And then mix in bilinear weights
        w.x = w.x * (1.0 - uvx) * (1.0 - uvy);
        w.y = w.y * (uvx) * (1.0 - uvy);
        w.z = w.z * (1.0 - uvx) * (uvy);
        w.w = w.w * (uvx) * (uvy);

        // Get final max weight.
        float ws = max(w.x + w.y + w.z + w.w, 1.0e-3);

        // normalize
        w /= ws;

        vec3 historyNormal;
        float historyLinearDepth;

        reprojection       = reprojection00 * w.x + reprojection10 * w.y + reprojection01 * w.z + reprojection11 * w.w;
        historyLinearDepth = depth00 * w.x + depth10 * w.y + depth01 * w.z + depth11 * w.w;
        historyNormal      = normal00 * w.x + normal10 * w.y + normal01 * w.z + normal11 * w.w;
        disocclusionFactor = getDisocclusionFactor(normal, historyNormal, linearDepth, historyLinearDepth);
    }
    disocclusionFactor = disocclusionFactor < kDisocclusionThreshold ? 0.0 : disocclusionFactor;
}

void reproject(ivec2 dispatchThreadId, ivec2 groupThreadId, uvec2 screenSize, float temporalStabilityFactor, int maxSamples) 
{
    initializeGroupSharedMemory(dispatchThreadId, groupThreadId, ivec2(screenSize));

    groupMemoryBarrier();
    barrier();

    // Center threads in groupshared memory
    groupThreadId += ivec2(4);

    float variance   = 1.0;
    float numSamples = 0.0;
    float roughness  = float(texelFetch(inSSRExtractRoughness, dispatchThreadId, 0).r);
    
    vec3 normal = unpackWorldNormal(texelFetch(inGbufferB, dispatchThreadId, 0).rgb);

    vec4 intersectResult = texelFetch(inSSRIntersection, dispatchThreadId, 0);
    vec3 radiance = vec3(intersectResult.xyz);
    const float rayLength = float(intersectResult.w);

    if (isGlossyReflection(roughness)) 
    {
        float disocclusionFactor;
        vec2 reprojectionUV;
        vec3 reprojection;

        pickReprojection(
            /* in  */ dispatchThreadId,
            /* in  */ groupThreadId,
            /* in  */ screenSize,
            /* in  */ roughness,
            /* in  */ rayLength,
            /* out */ disocclusionFactor,
            /* out */ reprojectionUV,
            /* out */ reprojection
        );

        if (reprojectionUV.x > 0.0 && reprojectionUV.y > 0.0 && reprojectionUV.x < 1.0 && reprojectionUV.y < 1.0) 
        {
            float prevVariance = sampleVarianceHistory(reprojectionUV);
            numSamples = sampleNumSamplesHistory(reprojectionUV) * disocclusionFactor;

            // Config sample nums.
            float sMaxSamples = max(8.0, float(maxSamples) * (1.0 - exp(-roughness * 100.0)));
            numSamples = min(sMaxSamples, numSamples + 1);

            float newVariance  = computeTemporalVariance(radiance.xyz, reprojection.xyz);
            if (disocclusionFactor < kDisocclusionThreshold) 
            {
                storeRadianceReprojected(dispatchThreadId, vec3(0.0));
                storeVariance(dispatchThreadId, 1.0);
                storeNumSamples(dispatchThreadId, 1.0);
            } 
            else 
            {
                float varianceMix = mix(newVariance, prevVariance, 1.0 / numSamples);

                storeRadianceReprojected(dispatchThreadId, reprojection);
                storeVariance(dispatchThreadId, varianceMix);
                storeNumSamples(dispatchThreadId, numSamples);
                
                // Mix in reprojection for radiance mip computation 
                radiance = mix(radiance, reprojection, 0.3);
            }
        } 
        else 
        {
            storeRadianceReprojected(dispatchThreadId, vec3(0.0));
            storeVariance(dispatchThreadId, 1.0);
            storeNumSamples(dispatchThreadId, 1.0);
        }
    }
    
    // Downsample 8x8 -> 1 radiance using groupshared memory
    // Initialize groupshared array for downsampling
    float weight = getLuminanceWeight(radiance.xyz);
    radiance.xyz *= weight;

    if (
        any(bvec2(dispatchThreadId.x >= screenSize.x, dispatchThreadId.y >= screenSize.y))
     || any(isinf(radiance)) 
     || any(isnan(radiance)) 
     || weight > 1.0e3) 
    {
        radiance = vec3(0.0);
        weight   = 0.0;
    }

    groupThreadId -= 4; // Center threads in groupshared memory

    storeInGroupSharedMemory(groupThreadId, vec4(radiance.xyz, weight));

    groupMemoryBarrier();
    barrier();

    for (int i = 2; i <= 8; i = i * 2) 
    {
        int ox = groupThreadId.x * i;
        int oy = groupThreadId.y * i;
        int ix = groupThreadId.x * i + i / 2;
        int iy = groupThreadId.y * i + i / 2;
        if (ix < 8 && iy < 8) 
        {
            vec4 rad_weight00 = loadFromGroupSharedMemoryRaw(ivec2(ox, oy));
            vec4 rad_weight10 = loadFromGroupSharedMemoryRaw(ivec2(ox, iy));
            vec4 rad_weight01 = loadFromGroupSharedMemoryRaw(ivec2(ix, oy));
            vec4 rad_weight11 = loadFromGroupSharedMemoryRaw(ivec2(ix, iy));

            vec4 sum = rad_weight00 + rad_weight01 + rad_weight10 + rad_weight11;
            storeInGroupSharedMemory(ivec2(ox, oy), sum);
        }

        groupMemoryBarrier();
        barrier();
    }

    if (groupThreadId.x == 0 && groupThreadId.y == 0) 
    {
        vec4 sum = loadFromGroupSharedMemoryRaw(ivec2(0, 0));
        float weightAcc = max(sum.w, 1.0e-3);

        vec3 radianceAvg = sum.xyz / weightAcc;
        storeAverageRadiance(dispatchThreadId.xy / 8, radianceAvg);
    }
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
    reproject(ivec2(remappedDispatchThreadId), ivec2(remappedGroupThreadId), screenSize, kTemporalStableReprojectFactor, kTemporalPeriod);
}