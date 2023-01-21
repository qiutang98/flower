#ifndef VOLUMETRIC_CLOUD_COMMON_GLSL
#define VOLUMETRIC_CLOUD_COMMON_GLSL

/*
** Physical based render code, develop by engineer: qiutanguu.
*/

// My personal volumetric cloud implement.
// Reference implement from https://www.slideshare.net/guerrillagames/the-realtime-volumetric-cloudscapes-of-horizon-zero-dawn.

#include "Common.glsl"
#include "Bayer.glsl"

layout (set = 0, binding = 0, rgba16f) uniform image2D imageHdrSceneColor;
layout (set = 0, binding = 1) uniform texture2D inHdrSceneColor;

layout (set = 0, binding = 2, rgba16f) uniform image2D imageCloudRenderTexture; // quater resolution.
layout (set = 0, binding = 3) uniform texture2D inCloudRenderTexture; // quater resolution.

layout (set = 0, binding = 4) uniform texture2D inDepth;
layout (set = 0, binding = 5) uniform texture2D inGBufferA;

layout (set = 0, binding = 6) uniform texture3D inBasicNoise;
layout (set = 0, binding = 7) uniform texture3D inWorleyNoise;

layout (set = 0, binding = 8) uniform texture2D inWeatherTexture;
layout (set = 0, binding = 9) uniform texture2D inGradientTexture;

layout (set = 0, binding = 10) uniform texture2D inTransmittanceLut;
layout (set = 0, binding = 11) uniform texture3D inFroxelScatter;

layout (set = 0, binding = 12, r32f) uniform image2D imageCloudShadowDepth;
layout (set = 0, binding = 13) uniform texture2D inCloudShadowDepth;

layout (set = 0, binding = 14, rgba16f) uniform image2D imageCloudReconstructionTexture;  // full resolution.
layout (set = 0, binding = 15) uniform texture2D inCloudReconstructionTexture;  // full resolution.

layout (set = 0, binding = 16, r32f) uniform image2D imageCloudDepthTexture;  // quater resolution.
layout (set = 0, binding = 17) uniform texture2D inCloudDepthTexture;  // quater resolution.

layout (set = 0, binding = 18, r32f) uniform image2D imageCloudDepthReconstructionTexture;  // full resolution.
layout (set = 0, binding = 19) uniform texture2D inCloudDepthReconstructionTexture;  // full resolution.

layout (set = 0, binding = 20) uniform texture2D inCloudReconstructionTextureHistory;
layout (set = 0, binding = 21) uniform texture2D inCloudDepthReconstructionTextureHistory;


layout (set = 0, binding = 22, r8) uniform image2D imageGbufferTranslucentMask;  // full resolution.

// Other common set.
layout (set = 1, binding = 0) uniform UniformView  { ViewData  viewData;  };
layout (set = 2, binding = 0) uniform UniformFrame { FrameData frameData; };

// Common sampler set.
#define COMMON_SAMPLER_SET 3
#include "CommonSamplerSet.glsl"

#define BLUE_NOISE_TEXTURE_SET 4
#define BLUE_NOISE_BUFFER_SET 5
#include "BlueNoiseCommon.glsl"

// Helper header.
#include "RayCommon.glsl"
#include "Sample.glsl"
#include "Phase.glsl"

///////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////
// Cloud shape.

float remap(float value, float orignalMin, float orignalMax, float newMin, float newMax)
{
    return newMin + (saturate((value - orignalMin) / (orignalMax - orignalMin)) * (newMax - newMin));
}

// TODO: change shape function in the future. This should be artist bias.
float cloudMap(vec3 posMeter, float normalizeHeight)  // Meter
{
    // If normalize height out of range, pre-return.
    // May evaluate error value on shadow light.
    if(normalizeHeight < 1e-4f || normalizeHeight > 0.9999f)
    {
        return 0.0f;
    }

    const float kCoverage = 0.50;
    const float kDensity  = 0.10; // 0.05

    const vec3 windDirection = vec3(1.0, 0.0, 0.0);
    const float cloudSpeed = 0.1f;

    posMeter += windDirection * normalizeHeight * 500.0f;
    vec3 posKm = posMeter * 0.001;

    vec3 windOffset = (windDirection + vec3(0.0, 0.1, 0.0)) * frameData.appTime.x * cloudSpeed;

    vec2 sampleUv = posKm.xz * 0.005;
    vec4 weatherValue = texture(sampler2D(inWeatherTexture, linearRepeatSampler), sampleUv);

    float coverage = saturate(kCoverage * weatherValue.x);
	float gradienShape = remap(normalizeHeight, 0.00, 0.10, 0.1, 1.0) * remap(normalizeHeight, 0.10, 0.80, 1.0, 0.2);
    // float gradienShape = remap(normalizeHeight, 0.00, 0.10, 0.1, 1.0) * remap(normalizeHeight, 0.10, 0.90, 1.0, 0.6);

    float basicNoise = texture(sampler3D(inBasicNoise, linearRepeatSampler), (posKm + windOffset) * vec3(0.1)).r;
    
    float basicCloudNoise = gradienShape * basicNoise;
    //    basicCloudNoise = mix(basicCloudNoise, smoothstep(0.1, 0.9, basicCloudNoise), saturate(1.0 - normalizeHeight * 6.0));

	float basicCloudWithCoverage = coverage * remap(basicCloudNoise, 1.0 - coverage, 1, 0, 1);

    vec3 sampleDetailNoise = posKm - windOffset * 0.15 + vec3(basicNoise.x, 0.0, basicCloudNoise) * normalizeHeight;
    float detailNoiseComposite = texture(sampler3D(inWorleyNoise, linearRepeatSampler), sampleDetailNoise * 0.2).r;
	float detailNoiseMixByHeight = 0.2 * mix(detailNoiseComposite, 1 - detailNoiseComposite, saturate(normalizeHeight * 10.0));
    
    float densityShape = saturate(0.01 + normalizeHeight * 1.15) * kDensity *
        remap(normalizeHeight, 0.0, 0.1, 0.0, 1.0) * 
        remap(normalizeHeight, 0.8, 1.0, 1.0, 0.0);

    float cloudDensity = remap(basicCloudWithCoverage, detailNoiseMixByHeight, 1.0, 0.0, 1.0);
	return cloudDensity * densityShape;
 
}

// Cloud shape end.
////////////////////////////////////////////////////////////////////////////////////////

#endif