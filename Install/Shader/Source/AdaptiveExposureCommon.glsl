#ifndef ADAPTIVE_EXPOSURE_COMMON_GLSL
#define ADAPTIVE_EXPOSURE_COMMON_GLSL

/*
** Physical based render code, develop by engineer: qiutanguu.
*/

#extension GL_EXT_samplerless_texture_functions : enable

#include "Common.glsl"

layout (set = 0, binding = 0) uniform texture2D inHDRSceneColor;
layout (set = 0, binding = 1, r16f) uniform image2D adaptedLumImage; // 1x1 buffer.
layout (set = 0, binding = 2, r32ui) uniform uimage2D histogramImage; // kHistogramBin x 1 buffer.
layout (set = 0, binding = 3) uniform utexture2D inHistogramImage;
layout (set = 0, binding = 4) uniform texture2D inPrevLumImage; // 1x1 buffer, history frame.

#define COMMON_SAMPLER_SET 1
#include "CommonSamplerSet.glsl"

layout (set = 2, binding = 0) uniform UniformView  { ViewData  viewData;  };
layout (set = 3, binding = 0) uniform UniformFrame { FrameData frameData; };

layout (push_constant) uniform PushConsts 
{  
    float scale;
    float offset;
    float lowPercent;
    float highPercent;
    float minBrightness;
    float maxBrightness;
    float speedDown;
    float speedUp;
    float exposureCompensation;
    float deltaTime;
} autoExposurePush;

const uint kHistogramBin = 128;
const uint kHistogramThreadDim = 16;
const uint kHistogramReductionThreadDimX = kHistogramThreadDim;
const uint kHistogramReductionThreadDimY = kHistogramBin / kHistogramThreadDim;

// Magic filter from unity engine postprocess stack.
// Can smooth lerp scene filter.

float getHistogramBinFromLuminance(float value)
{
    return saturate(log2(value) * autoExposurePush.scale + autoExposurePush.offset);
}

float getLuminanceFromHistogramBin(float bin)
{
    return exp2((bin - autoExposurePush.offset) / autoExposurePush.scale);
}

#endif
