#ifndef SHADOW_COMMON_GLSL
#define SHADOW_COMMON_GLSL

#ifndef SHADOW_DEPTH_GATHER
#define SHADOW_DEPTH_GATHER 0
#endif

#ifndef SHADOW_COLOR
#define SHADOW_COLOR 0.99
#endif

#include "shared_functions.glsl"

// Surface normal based bias, see https://learn.microsoft.com/en-us/windows/win32/dxtecharts/cascaded-shadow-maps for more details.
vec3 biasNormalOffset(vec3 N, float NoL, float texelSize)
{
    return N * clamp(1.0f - NoL, 0.0f, 1.0f) * texelSize * 30.0f;
}

// Auto bias by cacsade and NoL, some magic number here.
float autoBias(float NoL, float biasMul)
{
    return 1e-3f + (1.0f - NoL) * biasMul * 2e-3f;
}

// Depth Aware Contact harden pcf. See GDC2021: "Shadows of Cold War" for tech detail.
// Use cache occluder dist to fit one curve similar to tonemapper, to get some effect like pcss.
// can reduce tiny acne natively.
float contactHardenPCFKernal(
    const float occluders, 
    const float occluderDistSum, 
    const float compareDepth,
    const uint shadowSampleCount)
{
    // Normalize occluder dist.
    float occluderAvgDist = occluderDistSum / occluders;

#if SHADOW_DEPTH_GATHER
    float w = 1.0f / (4 * shadowSampleCount); // We gather 4 pixels.
#else
    float w = 1.0f / (1 * shadowSampleCount); 
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

    return 1.0f - percentageOccluded * SHADOW_COLOR;
}

float screenSpaceContactShadow(
    texture2D inSceneDepth,
    sampler inPointClampEdgeSampler,
    in const PerFrameData inViewData,
    float noise01, 
    uint stepNum, 
    vec3 wsRayStart, 
    vec3 wsRayDirection, 
    float wsRayLength) 
{
    // cast a ray in the direction of the light
    float occlusion = 0.0;
    
    ScreenSpaceRay rayData;
    initScreenSpaceRay(rayData, wsRayStart, wsRayDirection, wsRayLength, inViewData);

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
        float z = texture(sampler2D(inSceneDepth, inPointClampEdgeSampler), ray.xy).r;
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


#endif