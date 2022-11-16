#ifndef VOLUMETRIC_CLOUD_COMMON_GLSL
#define VOLUMETRIC_CLOUD_COMMON_GLSL

// My personal volumetric cloud implement.
// Reference implement from https://www.slideshare.net/guerrillagames/the-realtime-volumetric-cloudscapes-of-horizon-zero-dawn.


#include "Common.glsl"

layout (set = 0, binding = 0, rgba16f) uniform image2D imageHdrSceneColor;
layout (set = 0, binding = 1) uniform texture2D inHdrSceneColor;

layout (set = 0, binding = 2, rgba16f) uniform image2D imageCloudRenderTexture;
layout (set = 0, binding = 3) uniform texture2D inCloudRenderTexture;

layout (set = 0, binding = 4) uniform texture2D inDepth;
layout (set = 0, binding = 5) uniform texture2D inGBufferA;

layout (set = 0, binding = 6) uniform texture2D inBasicNoise;
layout (set = 0, binding = 7) uniform texture3D inWorleyNoise;

layout (set = 0, binding = 8) uniform texture2D inWeatherTexture;
layout (set = 0, binding = 9) uniform texture2D inGradientTexture;

layout (set = 0, binding = 10, r8) uniform image2D imageTranslucentMask;

// Other common set.
layout (set = 1, binding = 0) uniform UniformView  { ViewData  viewData;  };
layout (set = 2, binding = 0) uniform UniformFrame { FrameData frameData; };

// Common sampler set.
#define COMMON_SAMPLER_SET 3
#include "CommonSamplerSet.glsl"

// Helper header.
#include "RayCommon.glsl"
#include "Sample.glsl"
#include "Phase.glsl"

#endif