#version 460

////////// Config start.

// Current don't use depth gather. Fetch is enough.
#define SHADOW_DEPTH_GATHER 0

// We use blue noise offset sample position.
#define BLUE_NOISE_OFFSET   1

///////// Config end.

#extension GL_EXT_samplerless_texture_functions : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_shader_subgroup_basic : enable

#include "sdsm_common.glsl"
#include "../../common/shared_shadow.glsl"

#define SHARED_SAMPLER_SET 1
#include "../../common/shared_sampler.glsl"

#define BLUE_NOISE_BUFFER_SET 2
#include "../../common/shared_bluenoise.glsl"
#include "../../common/shared_poisson.glsl"

#if BLUE_NOISE_OFFSET
    // 8 tap taa blue noise. maybe 4 or 2 is enough.
    const uint kShadowSampleCount = 8;
#else
    // 12 tap poisson disk.
    const uint kShadowSampleCount = 12;
    #define poissonDisk kPoissonDisk_12
#endif

float shadowPcf(
    texture2D shadowDpeth,
    in const CascadeShadowConfig config,
    vec3 shadowCoord, 
    vec2 texelSize, 
    uint cascadeId, 
    float perCascadeEdge,
    vec2 screenPos,
    ivec2 colorSize,
    uvec2 offsetId)
{
    const float compareDepth = shadowCoord.z;

    vec2 scaleRange = 1.0f - 1.0f / cascadeInfos[0].cascadeScale.xy;
    scaleRange = (cascadeInfos[cascadeId].cascadeScale.xy / cascadeInfos[0].cascadeScale.xy - 1.0f / cascadeInfos[0].cascadeScale.xy) / scaleRange;

    vec2 filterSize = smoothstep(0.0f, 1.0f, scaleRange) * (config.maxFilterSize - config.shadowFilterSize) + config.shadowFilterSize;
    
    // When cacade increment, shadow map texel mapping size also increase.
    // We need to reduce soft shadow size to keep shading result same.
    const vec2 scaleOffset = texelSize * filterSize;
    
    float occluders = 0.0;
    float occluderDistSum = 0.0;

    float taaOffset = interleavedGradientNoise(screenPos, frameData.frameIndex.x % frameData.jitterPeriod);
    float taaAngle  = taaOffset * 3.14159265359 * 2.0f;


    for (uint i = 0; i < kShadowSampleCount; i++)
    {
        #if BLUE_NOISE_OFFSET
            vec2 offsetUv;
            offsetUv.x = -1.0 + 2.0 * samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d(offsetId.x, offsetId.y, i, 0u);
            offsetUv.y = -1.0 + 2.0 * samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d(offsetId.x, offsetId.y, i, 1u);
            offsetUv *= scaleOffset;
        #else
            float s = sin(taaAngle);
            float c = cos(taaAngle);
            vec2 offsetUv = scaleOffset * vec2(
                poissonDisk[i].x * c  + poissonDisk[i].y * s, 
                poissonDisk[i].x * -s + poissonDisk[i].y * c); 
        #endif

        // Build sample uv.
        vec2 sampleUv = shadowCoord.xy + offsetUv;
        sampleUv.x = clamp(sampleUv.x, perCascadeEdge * cascadeId, perCascadeEdge * (cascadeId + 1));
        sampleUv.y = clamp(sampleUv.y, 0.0f, 1.0f);

    #if SHADOW_DEPTH_GATHER
        vec4 depths = textureGather(sampler2D(shadowDpeth, pointClampEdgeSampler), sampleUv, 0);
        for(uint j = 0; j < 4; j ++)
        {
            float dist = depths[j] - compareDepth;
            float occluder = step(0.0, dist); // reverse z.

            // Collect occluders.
            occluders += occluder;
            occluderDistSum += dist * occluder;
        }
    #else
        float depthShadow = texture(sampler2D(shadowDpeth, pointClampEdgeSampler), sampleUv).r;
        {
            float dist = depthShadow - compareDepth;
            float occluder = step(0.0, dist); // reverse z.

            // Collect occluders.
            occluders += occluder;
            occluderDistSum += dist * occluder;
        }
    #endif
            
    }
    
    return contactHardenPCFKernal(occluders, occluderDistSum, compareDepth, kShadowSampleCount);
}



layout(local_size_x = 8, local_size_y = 8) in;
void main()
{
    ivec2 depthSize = textureSize(inDepth, 0);

    uvec2 groupThreadId = remap8x8(gl_LocalInvocationIndex);
    uvec2 dispatchId = groupThreadId + gl_WorkGroupID.xy * 8;
    ivec2 workPos = ivec2(dispatchId);

    if(workPos.x >= depthSize.x || workPos.y >= depthSize.y)
    {
        return;
    }

    // Non shadow-area pre-return.
    if(!isShadingModelValid(texelFetch(inGbufferA, workPos, 0).a))
    {
        imageStore(imageShadowMask, workPos, vec4(1.0f));
        return;
    }
    
    const vec2 uv = (vec2(workPos) + vec2(0.5f)) / vec2(depthSize);

    // Evaluate soft shadow.
    const vec4 inGbufferBValue = texelFetch(inGbufferB, workPos, 0);
    vec3 N = inGbufferBValue.rgb;

    const float deviceZ = texelFetch(inDepth, workPos, 0).r;
    vec3 worldPos = getWorldPos(uv, deviceZ, frameData);

    const SkyInfo sky = frameData.sky;
    float safeNoL = clamp(dot(N, normalize(-sky.direction)), 0.0, 1.0);

    // First find active cascade.
    uint activeCascadeId = 0;
    vec3 shadowCoord;

    // Loop to find suitable cascade.
    for(uint cascadeId = 0; cascadeId < sky.cacsadeConfig.cascadeCount; cascadeId ++)
    {
        // Perspective divide to get ndc position.
        shadowCoord = projectPos(worldPos, cascadeInfos[cascadeId].viewProj);
    
        // Check current cascade is valid in range.
        if(onRange(shadowCoord.xyz, vec3(sky.cacsadeConfig.cascadeBorderAdopt), vec3(1.0f - sky.cacsadeConfig.cascadeBorderAdopt)))
        {
            break;
        }
        activeCascadeId ++;
    }

    // Out of shadow area return lit.
    if(activeCascadeId == sky.cacsadeConfig.cascadeCount)
    {
        imageStore(imageShadowMask, workPos, vec4(1.0f));
        return;
    }

    // Offset retarget for new seeds each frame
    uvec2 offset = uvec2(vec2(0.754877669, 0.569840296) * (frameData.frameIndex.x) * uvec2(depthSize));
    uvec2 offsetId = uvec2(workPos) + offset;
    offsetId.x = offsetId.x % depthSize.x;
    offsetId.y = offsetId.y % depthSize.y;

    const float shadowTexelSize = 1.0f / float(sky.cacsadeConfig.percascadeDimXY);
    const vec3 offsetPos = biasNormalOffset(N, safeNoL, shadowTexelSize); // Offset position align normal direction.
    const float perCascadeOffsetUV = 1.0f / sky.cacsadeConfig.cascadeCount;

    // Final shadow result.
    float shadowResult = 1.0f;
    
    // Main cascsade shadow compute.
    {
        vec3 shadowPosOnAltas = projectPos(worldPos + offsetPos, cascadeInfos[activeCascadeId].viewProj);
        
        // Also add altas bias.
        shadowPosOnAltas.x = (shadowPosOnAltas.x + float(activeCascadeId)) * perCascadeOffsetUV;

        // Apply shadow depth bias.
        shadowPosOnAltas.z += autoBias(safeNoL, activeCascadeId + 1.0f);

        // Final evaluate shadow.
        shadowResult = shadowPcf(inSDSMShadowDepth, sky.cacsadeConfig, shadowPosOnAltas, vec2(shadowTexelSize), activeCascadeId, perCascadeOffsetUV, vec2(workPos), depthSize, offsetId);
    }

    // Cascade edge mix.
    const vec2 ndcPosAbs = abs(shadowCoord.xy);
    float cascadeFadeEdge = (max(ndcPosAbs.x, ndcPosAbs.y) - sky.cacsadeConfig.cascadeEdgeLerpThreshold) * 4.0f;
    if(cascadeFadeEdge > 0.0f && activeCascadeId < sky.cacsadeConfig.cascadeCount - 1)
    {
        // Mix to next cascade.
        const uint lerpCascadeId = activeCascadeId + 1;

        // Project to next cascasde position.
        vec4 lerpShadowProjPos = cascadeInfos[lerpCascadeId].viewProj * vec4(worldPos + offsetPos, 1.0f); 
        lerpShadowProjPos.xyz /= lerpShadowProjPos.w;

        // Clamp to [0,1]
        lerpShadowProjPos.xy = lerpShadowProjPos.xy * 0.5f + 0.5f;
        lerpShadowProjPos.y = 1.0f - lerpShadowProjPos.y;

        // Altas bias.
        lerpShadowProjPos.x = (lerpShadowProjPos.x + float(lerpCascadeId)) * perCascadeOffsetUV;

        // Shadow depth bias.
        lerpShadowProjPos.z += autoBias(safeNoL, lerpCascadeId + 1.0f);

        // Evaluate next cascade shadow value.
        float lerpShadowValue = shadowPcf(inSDSMShadowDepth, sky.cacsadeConfig, lerpShadowProjPos.xyz, vec2(shadowTexelSize),  lerpCascadeId, perCascadeOffsetUV, vec2(workPos), depthSize, offsetId);
        
        // Mix shadow.
        cascadeFadeEdge = smoothstep(0.0f, 1.0f, cascadeFadeEdge);
        shadowResult = mix(shadowResult, lerpShadowValue, cascadeFadeEdge);
    }

#if 0
    // TODO: Hiz heightmap accelerate.
    // Ray cast in world space, and sample height map to know current pixel is occluded or not.
    const bool bShouldRayTraceTerrainShadow = shadowResult > 1e-3f;
    if(bHeightmapValid > 0 && bShouldRayTraceTerrainShadow)
    {
        float occFactor = 0.0f;
        vec2 heightMapSize = textureSize(inHeightmap, 0);

        const vec3 rayStart = worldPos;

        const uint kMaxSampleRayCount = 1; // Current use spp 1.
        const uint kStepCount = 128;
        const float kLodLevel = 1.0;
        const float kAdoptionCount = 128.0;
        for(uint index = 0; index < kMaxSampleRayCount; index ++)
        {
            float jitter = samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d(offsetId.x, offsetId.y, index, 0u);

            // Box intersection.
            const vec3 rayDirection = normalize(-sky.direction);
            const vec3 bboxMin = vec3(0.0f) - vec3(0.5 * heightMapSize.x, 0.0f,            0.5 * heightMapSize.y);
            const vec3 bboxMax = vec3(0.0f) + vec3(0.5 * heightMapSize.x, heightfiledDump, 0.5 * heightMapSize.y);
            float intersectT = boxLineIntersectWS(rayStart, rayDirection, bboxMin, bboxMax);
            if(intersectT > 0.0f)
            {
                const float kRayLen = intersectT;

                const float dt = kRayLen / float(kStepCount);
                float t = dt * jitter;
                float stepDt = dt;
                float adoption = kRayLen / float(kAdoptionCount)  * abs(rayDirection.y);

                vec3 ray;
                for (uint i = 0u ; i < kStepCount ; i++, t += stepDt) 
                {
                    ray = rayStart + rayDirection * t;

                    vec3 rayUvz = vec3(ray.xz + heightMapSize * 0.5, ray.y);
                    rayUvz.y = heightMapSize.y - rayUvz.y;

                    const float heightSample = textureLod(sampler2D(inHeightmap, linearClampEdgeSampler), vec2(rayUvz.xy) / heightMapSize, kLodLevel).r * heightfiledDump;
                    if(rayUvz.z + adoption < heightSample)
                    {
                        occFactor += 1.0f;
                        break;
                    #if 0
                        if(heightSample - rayUvz.z > 0.5 * adoption  && heightSample - rayUvz.z < adoption)
                        {
                            occFactor += 1.0f;
                            break;
                        }

                        t -= stepDt;
                        stepDt *= 0.5;
                    #endif
                    }

                }
            }
        }
        occFactor /= float(kMaxSampleRayCount);
        shadowResult = min(shadowResult, 1.0 - occFactor);
    }
#endif

    // SDSM keep high accurate shadow when camera move near, and may see some visual artifact which cause by contact shadow.
    // So current don't need this tech here.
#if 0
    // Maybe we need these tech in the future, so keep here as one reference.
    // Do screen space ray trace shadow to fill depth bias leaking problem.
    // Note from: https://panoskarabelas.com/posts/screen_space_shadows/
    // Note from: Unreal engine4 contact shadow.
    const bool bShouldRayTraceShadow = shadowResult > 1e-3f;
    if(bShouldRayTraceShadow)
    {
        float rayTraceShadow = 1.0f - screenSpaceContactShadow(
            inDepth,
            pointClampEdgeSampler,
            frameData,
            interleavedGradientNoise(vec2(workPos), frameData.frameIndex.x % frameData.jitterPeriod)
            , 8
            , worldPos
            , normalize(-sky.direction)
            , 0.25
        );

        shadowResult = min(rayTraceShadow, shadowResult);
    }
#endif

    imageStore(imageShadowMask, workPos, vec4(shadowResult));
}