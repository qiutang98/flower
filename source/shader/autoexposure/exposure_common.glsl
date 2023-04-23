#ifndef ADAPTIVE_EXPOSURE_COMMON_GLSL
#define ADAPTIVE_EXPOSURE_COMMON_GLSL

#extension GL_EXT_samplerless_texture_functions : enable

#include "../common/shared_functions.glsl"
#include "../common/shared_struct.glsl"

layout (set = 0, binding = 0) uniform texture2D inHDRSceneColor;
layout (set = 0, binding = 1, r16f) uniform image2D adaptedLumImage; // 1x1 buffer.
layout (set = 0, binding = 2, r32ui) uniform uimage2D histogramImage; // kHistogramBin x 1 buffer.
layout (set = 0, binding = 3) uniform utexture2D inHistogramImage;
layout (set = 0, binding = 4) uniform texture2D inPrevLumImage; // 1x1 buffer, history frame.
layout (set = 0, binding = 5) uniform UniformFrameData { PerFrameData frameData; };

#define SHARED_SAMPLER_SET 1
#include "../common/shared_sampler.glsl"

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

// Magic filter function from unity engine postprocess stack. This can smooth lerp scene filter.
float getHistogramBinFromLuminance(float value)
{
    return saturate(log2(value) * autoExposurePush.scale + autoExposurePush.offset);
}

float getLuminanceFromHistogramBin(float bin)
{
    return exp2((bin - autoExposurePush.offset) / autoExposurePush.scale);
}

#endif
