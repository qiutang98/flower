#ifndef SSGI_COMMON_GLSL
#define SSGI_COMMON_GLSL

#extension GL_EXT_samplerless_texture_functions : enable

// SSGI is a bad global illumination solution for realtime rendering. :(
// We will use RTX ddgi after first version release.

#include "Common.glsl"
#include "FastMath.glsl"

const uint kMostDetailedMip = 0;
const float kTemporalStableReprojectFactor = .9f; // big value is ghosting, small value is noise.

const int kTemporalPeriod = 32; // 32 is good for keep energy fill.
const float kTemporalStableFactor = kTemporalStableReprojectFactor;

const float kDepthBufferThickness = 50.0f;
const uint kMaxTraversalIterations = 128; // 128;
const uint kMinTraversalOccupancy = 4;

// FIX param.
const float kAverageRadianceLuminanceWeight = 0.3f;
const float kDisocclusionNormalWeight = 1.4f;
const float kDisocclusionDepthWeight = 1.0f;
const float kReprojectionNormalSimilarityThreshold = 0.9999;
const float kReprojectSurfaceDiscardVarianceWeight = 1.5;
const float kDisocclusionThreshold = 0.9;
const float kPrefilterNormalSigma = 512.0;
const float kPrefilterDepthSigma = 4.0;
const float kRadianceWeightBias = 0.6;
const float kRadianceWeightVarianceK = 0.1;
const float kPrefilterVarianceBias = 0.1;
const float kPrefilterVarianceWeight = 4.4;

layout (set = 0, binding = 0)  uniform texture2D inHiz;
layout (set = 0, binding = 1)  uniform texture2D inDepth;
layout (set = 0, binding = 2)  uniform texture2D inGbufferA; // Gbuffer A use for additional light.
layout (set = 0, binding = 3)  uniform texture2D inGbufferB;
layout (set = 0, binding = 4)  uniform texture2D inGbufferS;
layout (set = 0, binding = 5)  uniform texture2D inGbufferV;
layout (set = 0, binding = 6)  uniform texture2D inPrevDepth;
layout (set = 0, binding = 7)  uniform texture2D inPrevGbufferB;
layout (set = 0, binding = 8)  uniform texture2D inHDRSceneColor; // Current frame Hdr scene color lit by direct lighting, it will used for ssgi.

layout (set = 0, binding = 9, rgba16f) uniform image2D HDRSceneColorImage; // HDR output.

layout (set = 0, binding = 10) uniform textureCube inCubeGlobalPrefilter; // SSR fallback env ibl.
layout (set = 0, binding = 11)  uniform texture2D inGTAO; // GTAO for ssr occlusion.

layout (set = 0, binding = 12, rgba16f) uniform image2D SSRIntersection; // ssr intersect result.
layout (set = 0, binding = 13) uniform texture2D inSSRIntersection; // in ssr intersect.

layout (set = 0, binding = 14) uniform texture2D inPrevSSRRadiance; // ssr prevframe radiance result. for reproject pass.
layout (set = 0, binding = 15) uniform texture2D inPrevSampleCount; // ssr prevframe sample count. for reproject and temporal pass.

layout (set = 0, binding = 16, rgba16f) uniform image2D SSRReprojectedRadiance; // ssr reproject output radiance.
layout (set = 0, binding = 17, r11f_g11f_b10f) uniform image2D SSRAverageRadiance;  
layout (set = 0, binding = 18, r16f) uniform image2D SSRVariance;
layout (set = 0, binding = 19, r16f) uniform image2D SSRSampleCount;

layout (set = 0, binding = 20) uniform texture2D inSSRReprojectedRadiance;
layout (set = 0, binding = 21) uniform texture2D inSSRAverageRadiance;
layout (set = 0, binding = 22) uniform texture2D inSSRVariance;
layout (set = 0, binding = 23) uniform texture2D inSSRVarianceHistory; // SSR variance help to filter.

layout (set = 0, binding = 24, rgba16f) uniform image2D SSRPrefilterRadiance;
layout (set = 0, binding = 25, r16f) uniform image2D SSRPrefilterVariance;
layout (set = 0, binding = 26) uniform texture2D inSSRPrefilterRadiance;
layout (set = 0, binding = 27) uniform texture2D inSSRPrefilterVariance;

layout (set = 0, binding = 28, rgba16f) uniform image2D SSRTemporalFilterRadiance;
layout (set = 0, binding = 29, r16f) uniform image2D SSRTemporalfilterVariance;

layout(push_constant) uniform PushConsts
{   
    float intensity;
} SSGIPush;

layout (set = 1, binding = 0) uniform UniformView { ViewData viewData; };
layout (set = 2, binding = 0) uniform UniformFrame { FrameData frameData; };

#define COMMON_SAMPLER_SET 3
#include "CommonSamplerSet.glsl"

// Temporal blue noise jitter is hard to stable resolve. :(
// Maybe simple blue noise is better.
#define BLUE_NOISE_TEXTURE_SET 4
#define BLUE_NOISE_BUFFER_SET 5
#include "BlueNoiseCommon.glsl"

vec3 envIBLReflectionCallback(uvec2 dispatchId, vec2 uv, float roughness)
{
    vec3 n = normalize(texelFetch(inGbufferB, ivec2(dispatchId), 0).xyz); 
    float deviceZ = texelFetch(inDepth, ivec2(dispatchId), 0).r;
    vec3 worldPos = getWorldPos(uv, deviceZ, viewData);
    vec3 v = normalize(viewData.camWorldPos.xyz - worldPos);

    vec3 reflection = normalize(reflect(-v, n));
    float NdotV = clamp(dot(n, v), 0.0, 1.0);

    // Compute roughness's lod.
    uvec2 prefilterCubeSize = textureSize(inCubeGlobalPrefilter, 0);
    float mipCount = float(log2(max(prefilterCubeSize.x, prefilterCubeSize.y)));
    float lod = clamp(roughness * float(mipCount), 0.0, float(mipCount));
    
    // Load environment's color from prefilter color.
    vec3 specularLight = textureLod(samplerCube(inCubeGlobalPrefilter, linearClampEdgeSampler), reflection, lod).rgb;
    return specularLight;
}

const int   kLocalNeighborhoodRadius = 4;
const float kGaussianK = 3.0;

float localNeighborhoodKernelWeight(float i) 
{
    const float radius = kLocalNeighborhoodRadius + 1.0;
    return exp(-kGaussianK * (i * i) / (radius * radius));
}

float luminanceSSGI(vec3 color) 
{ 
    return float(max(luminance(color), 0.001)); 
}

float computeTemporalVariance(vec3 radiance, vec3 historyRadiance) 
{
    float historyLuminance = luminanceSSGI(historyRadiance);
    float luminanceCurrent = luminanceSSGI(radiance);

    float diff  = abs(historyLuminance - luminanceCurrent) / max(max(historyLuminance, luminanceCurrent), 0.5);
    return diff * diff;
}

#endif