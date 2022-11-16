#ifndef SSR_COMMON_GLSL
#define SSR_COMMON_GLSL

#extension GL_EXT_samplerless_texture_functions : enable

// Screen space reflection for global illumination.
// It generate ray from depth buffer and normal buffer, then trace the hiz buffer, find the hit position, and return it's color.

// Ray hit by hiz.
// Spatial filter by à trous.
// Reproject with recurrent blur.

// Bad news is that it still ghosting.

#include "Common.glsl"
#include "FastMath.glsl"

const float kTemporalStableReprojectFactor = .21f; // big value is ghosting, small value is noise.

const int kTemporalPeriod = 32; // 32 is good for keep energy fill.
const float kTemporalStableFactor = kTemporalStableReprojectFactor;

const float kDepthBufferThickness = 0.45f;
const uint kMaxTraversalIterations = 64; // 128;
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
layout (set = 0, binding = 2)  uniform texture2D inGbufferA;
layout (set = 0, binding = 3)  uniform texture2D inGbufferB;
layout (set = 0, binding = 4)  uniform texture2D inGbufferS;
layout (set = 0, binding = 5)  uniform texture2D inGbufferV;
layout (set = 0, binding = 6)  uniform texture2D inPrevDepth;
layout (set = 0, binding = 7)  uniform texture2D inPrevGbufferB;
layout (set = 0, binding = 8)  uniform texture2D inHDRSceneColor; // Current frame Hdr scene color lit by direct lighting, it will used for reflection src.
layout (set = 0, binding = 9)  uniform texture2D inBRDFLut;

layout (set = 0, binding = 10) buffer SSRRayCounterSSBO
{ 
    uint rayCount; 
    uint denoiseTileCount;
} ssboRayCounter; // SSR ray counter, use for intersection and denoise dispatch indirect args.

layout (set = 0, binding = 11) buffer SSRRayListSSBO
{ 
    uint data[]; 
} ssboRayList; // SSR ray list, use for ray list info cache.

layout (set = 0, binding = 12) buffer SSRDenoiseTileListSSBO
{ 
    uint data[]; 
} ssboDenoiseTileList; // SSR denoise tile data, store some denoise info.

layout (set = 0, binding = 13, rgba16f) uniform image2D HDRSceneColorImage; // HDR output.

layout (set = 0, binding = 14) uniform textureCube inCubeGlobalPrefilter; // SSR fallback env ibl.
layout (set = 0, binding = 15)  uniform texture2D inGTAO; // GTAO for ssr occlusion.

layout (set = 0, binding = 16) buffer SSRIntersectCmdSSBO
{ 
    DispatchIndirectCommand args; 
} ssboIntersectCommand;

layout (set = 0, binding = 17) buffer SSRDenoiseCmdSSBO
{ 
    DispatchIndirectCommand args; 
} ssboDenoiseCommand;

layout (set = 0, binding = 18, r8) uniform image2D SSRExtractRoughness; // ssr roughness extract.
layout (set = 0, binding = 19) uniform texture2D inSSRExtractRoughness; // current frame ssr roughness.

layout (set = 0, binding = 20, rgba16f) uniform image2D SSRIntersection; // ssr intersect result.
layout (set = 0, binding = 21) uniform texture2D inSSRIntersection; // in ssr intersect.

layout (set = 0, binding = 22) uniform texture2D inPrevSSRExtractRoughness; // ssr prevframe roughness.
layout (set = 0, binding = 23) uniform texture2D inPrevSSRRadiance; // ssr prevframe radiance result. for reproject pass.
layout (set = 0, binding = 24) uniform texture2D inPrevSampleCount; // ssr prevframe sample count. for reproject and temporal pass.

layout (set = 0, binding = 25, rgba16f) uniform image2D SSRReprojectedRadiance; // ssr reproject output radiance.
layout (set = 0, binding = 26, r11f_g11f_b10f) uniform image2D SSRAverageRadiance;  
layout (set = 0, binding = 27, r16f) uniform image2D SSRVariance;
layout (set = 0, binding = 28, r16f) uniform image2D SSRSampleCount;

layout (set = 0, binding = 29) uniform texture2D inSSRReprojectedRadiance;
layout (set = 0, binding = 30) uniform texture2D inSSRAverageRadiance;
layout (set = 0, binding = 31) uniform texture2D inSSRVariance;
layout (set = 0, binding = 32) uniform texture2D inSSRVarianceHistory; // SSR variance help to filter.

layout (set = 0, binding = 33, rgba16f) uniform image2D SSRPrefilterRadiance;
layout (set = 0, binding = 34, r16f) uniform image2D SSRPrefilterVariance;
layout (set = 0, binding = 35) uniform texture2D inSSRPrefilterRadiance;
layout (set = 0, binding = 36) uniform texture2D inSSRPrefilterVariance;

layout (set = 0, binding = 37, rgba16f) uniform image2D SSRTemporalFilterRadiance;
layout (set = 0, binding = 38, r16f) uniform image2D SSRTemporalfilterVariance;



layout (set = 1, binding = 0) uniform UniformView { ViewData viewData; };
layout (set = 2, binding = 0) uniform UniformFrame { FrameData frameData; };

#define COMMON_SAMPLER_SET 3
#include "CommonSamplerSet.glsl"

// Temporal blue noise jitter is hard to stable resolve. :(
// Maybe simple blue noise is better.
#define BLUE_NOISE_TEXTURE_SET 4
#define BLUE_NOISE_BUFFER_SET 5
#include "BlueNoiseCommon.glsl"

layout(push_constant) uniform PushConsts
{   
    uint samplesPerQuad;
    uint temporalVarianceGuidedTracingEnabled;
    uint mostDetailedMip;

    float roughnessThreshold; // Max roughness stop to reflection sample.
    float temporalVarianceThreshold;
} SSRPush;

bool isGlossyReflection(float roughness) 
{
    return roughness < SSRPush.roughnessThreshold;
}

bool isMirrorReflection(float roughness) 
{
    return roughness < 0.0001;
}

uint addRayCount(uint value) 
{
    return atomicAdd(ssboRayCounter.rayCount, value);
}

uint addDenoiseTileCount()
{
    return atomicAdd(ssboRayCounter.denoiseTileCount, 1);
}

uint packRayCoords(uvec2 rayCoord, bool bCopyHorizontal, bool bCopyVertical, bool bCopyDiagonal) 
{
    uint rayX15bit = rayCoord.x & 0x7FFF;
    uint rayY14bit = rayCoord.y & 0x3FFF;

    uint copyHorizontal1bit = bCopyHorizontal ? 1 : 0;
    uint copyVertical1bit   = bCopyVertical   ? 1 : 0;
    uint copyDiagonal1bit   = bCopyDiagonal   ? 1 : 0;

    uint packed = (copyDiagonal1bit << 31) | (copyVertical1bit << 30) | (copyHorizontal1bit << 29) | (rayY14bit << 15) | (rayX15bit << 0);
    return packed;
}

void unpackRayCoords(uint packed, out uvec2 rayCoord, out bool bCopyHorizontal, out bool bCopyVertical, out bool bCopyDiagonal) 
{
    rayCoord.x = (packed >> 0) & 0x7FFF;
    rayCoord.y = (packed >> 15) & 0x3FFF;

    bCopyHorizontal = ((packed >> 29) & 0x1) != 0;
    bCopyVertical   = ((packed >> 30) & 0x1) != 0;
    bCopyDiagonal   = ((packed >> 31) & 0x1) != 0;
}

void addRay(uint index, uvec2 rayCoord, bool bCopyHorizontal, bool bCopyVertical, bool bCopyDiagonal)
{
    ssboRayList.data[index] = packRayCoords(rayCoord, bCopyHorizontal, bCopyVertical, bCopyDiagonal);
}

void addDenoiserTile(uint index, uvec2 tileCoord) 
{
    ssboDenoiseTileList.data[index] = ((tileCoord.y & 0xffffu) << 16) | ((tileCoord.x & 0xffffu) << 0);
}

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

float luminanceSSR(vec3 color) 
{ 
    return float(max(luminance(color), 0.001)); 
}

float computeTemporalVariance(vec3 radiance, vec3 historyRadiance) 
{
    float historyLuminance = luminanceSSR(historyRadiance);
    float luminanceCurrent = luminanceSSR(radiance);

    float diff  = abs(historyLuminance - luminanceCurrent) / max(max(historyLuminance, luminanceCurrent), 0.5);
    return diff * diff;
}

#endif