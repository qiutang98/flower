#version 460

/*
** Physical based render code, develop by engineer: qiutanguu.
*/

#extension GL_EXT_samplerless_texture_functions : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_shader_subgroup_basic : enable

#include "SDSM_Common.glsl"
#include "RayCommon.glsl"

layout (set = 1, binding = 0) uniform UniformView { ViewData viewData; };
layout (set = 2, binding = 0) uniform UniformFrame { FrameData frameData; };

#define COMMON_SAMPLER_SET 3
#include "CommonSamplerSet.glsl"

#define BLUE_NOISE_TEXTURE_SET 4
#define BLUE_NOISE_BUFFER_SET 5
#include "BlueNoiseCommon.glsl"

// Surface normal based bias, see https://learn.microsoft.com/en-us/windows/win32/dxtecharts/cascaded-shadow-maps for more details.
vec3 biasNormalOffset(vec3 N, float NoL, float texelSize)
{
    return N * clamp(1.0f - NoL, 0.0f, 1.0f) * texelSize * 30.0f;
}

// Auto bias by cacsade and NoL, some magic number here.
float autoBias(float NoL, float biasMul)
{
    return 6e-4f + (1.0f - NoL) * biasMul * 1e-4f;
}

// Poisson disk sample tech, see https://en.wikipedia.org/wiki/Poisson_distribution for more details.
#define POISSON_DISK_COUNT 12

// TODO: We have temporal blue noise, maybe it work better and faster, but current poisson disk also look good.

#define SHADOW_DEPTH_GATHER 0
#define BLUE_NOISE_OFFSET 1
#if BLUE_NOISE_OFFSET
    const uint kShadowSampleCount = 8; // 8 tap taa blue noise. maybe 4 or 2 is enough.
#else
    const uint kShadowSampleCount = POISSON_DISK_COUNT;
    const vec2 poissonDisk[kShadowSampleCount] = vec2[](
    #if POISSON_DISK_COUNT == 25
        vec2(-0.9786980, -0.08841210),
        vec2(-0.8411210,  0.52116500),
        vec2(-0.7174600, -0.50322000),
        vec2(-0.7029330,  0.90313400),
        vec2(-0.6631980,  0.15482000),
        vec2(-0.4951020, -0.23288700),
        vec2(-0.3642380, -0.96179100),
        vec2(-0.3458660, -0.56437900),
        vec2(-0.3256630,  0.64037000),
        vec2(-0.1827140,  0.32132900),
        vec2(-0.1426130, -0.02273630),
        vec2(-0.0564287, -0.36729000),
        vec2(-0.0185858,  0.91888200),
        vec2( 0.0381787, -0.72899600),
        vec2( 0.1659900,  0.09311200),
        vec2( 0.2536390,  0.71953500),
        vec2( 0.3695490, -0.65501900),
        vec2( 0.4236270,  0.42997500),
        vec2( 0.5307470, -0.36497100),
        vec2( 0.5660270, -0.94048900),
        vec2( 0.6393320,  0.02841270),
        vec2( 0.6520890,  0.66966800),
        vec2( 0.7737970,  0.34501200),
        vec2( 0.9688710,  0.84044900),
        vec2( 0.9918820, -0.65733800)
    #elif POISSON_DISK_COUNT == 16
        vec2(-0.94201624, -0.39906216),
        vec2(0.94558609, -0.76890725),
        vec2(-0.094184101, -0.92938870),
        vec2(0.34495938, 0.29387760),
        vec2(-0.91588581, 0.45771432),
        vec2(-0.81544232, -0.87912464),
        vec2(-0.38277543, 0.27676845),
        vec2(0.97484398, 0.75648379),
        vec2(0.44323325, -0.97511554),
        vec2(0.53742981, -0.47373420),
        vec2(-0.26496911, -0.41893023),
        vec2(0.79197514, 0.19090188),
        vec2(-0.24188840, 0.99706507),
        vec2(-0.81409955, 0.91437590),
        vec2(0.19984126, 0.78641367),
        vec2(0.14383161, -0.14100790)
    #elif POISSON_DISK_COUNT == 12
        vec2(-.326,-.406),
        vec2(-.840,-.074),
        vec2(-.696, .457),
        vec2(-.203, .621),
        vec2( .962,-.195),
        vec2( .473,-.480),
        vec2( .519, .767),
        vec2( .185,-.893),
        vec2( .507, .064),
        vec2( .896, .412),
        vec2(-.322,-.933),
        vec2(-.792,-.598)
    #elif POISSON_DISK_COUNT == 4
        vec2(-0.94201624, -0.39906216),
        vec2( 0.94558609, -0.76890725),
        vec2(-0.094184101,-0.92938870),
        vec2( 0.34495938,  0.29387760)
    #endif
    );

#endif

// Depth Aware Contact harden pcf. See GDC2021: "Shadows of Cold War" for tech detail.
// Use cache occluder dist to fit one curve similar to tonemapper, to get some effect like pcss.
// can reduce tiny acne natively.
float contactHardenPCFKernal(
    in const DirectionalLightInfo light, 
    const float occluders, 
    const float occluderDistSum, 
    const float compareDepth)
{
    // Normalize occluder dist.
    float occluderAvgDist = occluderDistSum / occluders;

#if SHADOW_DEPTH_GATHER
    float w = 1.0f / (4 * kShadowSampleCount); // We gather 4 pixels.
#else
    float w = 1.0f / (1 * kShadowSampleCount); 
#endif
    
    float pcfWeight = clamp(occluderAvgDist / compareDepth, 0.0, 1.0);
    
    // Normalize occluders.
    float percentageOccluded = clamp(occluders * w, 0.0, 1.0);

    // S curve fit.
    percentageOccluded = 2.0f * percentageOccluded - 1.0f;
    float occludedSign = sign(percentageOccluded);
    percentageOccluded = 1.0f - (occludedSign * percentageOccluded);
    percentageOccluded = mix(percentageOccluded * percentageOccluded * percentageOccluded, percentageOccluded, pcfWeight);
    percentageOccluded = 1.0f - percentageOccluded;
    percentageOccluded *= occludedSign;
    percentageOccluded = 0.5f * percentageOccluded + 0.5f;

    return 1.0f - percentageOccluded;
}

float shadowPcf(
    texture2D shadowDpeth,
    in const DirectionalLightInfo light,
    vec3 shadowCoord, 
    vec2 texelSize, 
    uint cascadeId, 
    float perCascadeEdge,
    vec2 screenPos,
    ivec2 colorSize)
{
    const float compareDepth = shadowCoord.z;

    vec2 scaleRange = 1.0f - 1.0f / cascadeInfos[0].cascadeScale.xy;
    scaleRange = (cascadeInfos[cascadeId].cascadeScale.xy / cascadeInfos[0].cascadeScale.xy - 1.0f / cascadeInfos[0].cascadeScale.xy) / scaleRange;

    vec2 filterSize = smoothstep(0.0f, 1.0f, scaleRange) * (light.maxFilterSize - light.shadowFilterSize) + light.shadowFilterSize;
    
    // When cacade increment, shadow map texel mapping size also increase.
    // We need to reduce soft shadow size to keep shading result same.
    const vec2 scaleOffset = texelSize * filterSize;
    
    float occluders = 0.0;
    float occluderDistSum = 0.0;

    // Offset retarget for new seeds each frame
    uvec2 offset = uvec2(vec2(0.754877669, 0.569840296) * frameData.frameIndex.x * uvec2(colorSize));
    uvec2 offsetId = uvec2(screenPos) + offset;
    offsetId.x = offsetId.x % colorSize.x;
    offsetId.y = offsetId.y % colorSize.y;

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
    
    return contactHardenPCFKernal(light, occluders, occluderDistSum, compareDepth);
}

float screenSpaceContactShadow(
    float noise01, 
    uint stepNum, 
    vec3 wsRayStart, 
    vec3 wsRayDirection, 
    float wsRayLength) 
{
    // cast a ray in the direction of the light
    float occlusion = 0.0;
    
    ScreenSpaceRay rayData;
    initScreenSpaceRay(rayData, wsRayStart, wsRayDirection, wsRayLength, viewData);

    // step
    const uint kStepCount = stepNum;
    const float dt = 1.0 / float(kStepCount);

    // tolerance
    const float tolerance = abs(rayData.ssViewRayEnd.z - rayData.ssRayStart.z) * dt;

    // dither the ray with interleaved gradient noise
    const float dither = noise01 - 0.5;

    // normalized position on the ray (0 to 1)
    float t = dt * dither + dt;

    vec3 ray;
    for (uint i = 0u ; i < kStepCount ; i++, t += dt) 
    {
        ray = rayData.uvRayStart + rayData.uvRay * t;
        float z = texture(sampler2D(inDepth, pointClampEdgeSampler), ray.xy).r;
        float dz = z - ray.z;
        if (abs(tolerance - dz) < tolerance) 
        {
            occlusion = 1.0;
            break;
        }
    }

    // we fade out the contribution of contact shadows towards the edge of the screen
    // because we don't have depth data there
    vec2 fade = max(12.0 * abs(ray.xy - 0.5) - 5.0, 0.0);
    occlusion *= saturate(1.0 - dot(fade, fade));

    return occlusion;
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
    vec3 N = normalize(inGbufferBValue.rgb);

    const float deviceZ = texelFetch(inDepth, workPos, 0).r;
    vec3 worldPos = getWorldPos(uv, deviceZ, viewData);

    const DirectionalLightInfo light = frameData.directionalLight;
    float safeNoL = clamp(dot(N, normalize(-light.direction)), 0.0, 1.0);

    // First find active cascade.
    uint activeCascadeId = 0;
    vec3 shadowCoord;

    // Loop to find suitable cascade.
    for(uint cascadeId = 0; cascadeId < light.cascadeCount; cascadeId ++)
    {
        // Perspective divide to get ndc position.
        shadowCoord = projectPos(worldPos, cascadeInfos[cascadeId].viewProj);
    
        // Check current cascade is valid in range.
        if(onRange(shadowCoord.xyz, vec3(light.cascadeBorderAdopt), vec3(1.0f - light.cascadeBorderAdopt)))
        {
            break;
        }
        activeCascadeId ++;
    }

    if(activeCascadeId == light.cascadeCount)
    {
        imageStore(imageShadowMask, workPos, vec4(1.0f));
        return;
    }

    const float shadowTexelSize = 1.0f / float(light.perCascadeXYDim);
    const vec3 offsetPos = biasNormalOffset(N, safeNoL, shadowTexelSize); // Offset position align normal direction.
    const float perCascadeOffsetUV = 1.0f / light.cascadeCount;

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
        shadowResult = shadowPcf(inSDSMShadowDepth, light, shadowPosOnAltas, vec2(shadowTexelSize), activeCascadeId, perCascadeOffsetUV, vec2(workPos), depthSize);
    }

    // Cascade edge mix.
    const vec2 ndcPosAbs = abs(shadowCoord.xy);
    float cascadeFadeEdge = (max(ndcPosAbs.x, ndcPosAbs.y) - light.cascadeEdgeLerpThreshold) * 4.0f;
    if(cascadeFadeEdge > 0.0f && activeCascadeId < light.cascadeCount - 1)
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
        float lerpShadowValue = shadowPcf(inSDSMShadowDepth, light, lerpShadowProjPos.xyz, vec2(shadowTexelSize),  lerpCascadeId, perCascadeOffsetUV, vec2(workPos), depthSize);
        
        // Mix shadow.
        cascadeFadeEdge = smoothstep(0.0f, 1.0f, cascadeFadeEdge);
        shadowResult = mix(shadowResult, lerpShadowValue, cascadeFadeEdge);
    }

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
            interleavedGradientNoise(vec2(workPos), frameData.frameIndex.x % frameData.jitterPeriod)
            , 8
            , worldPos
            , normalize(-light.direction)
            , 0.25
        );

        shadowResult = min(rayTraceShadow, shadowResult);
    }
#endif

    imageStore(imageShadowMask, workPos, vec4(shadowResult));
}