#ifndef DOF_COMMON_GLSL
#define DOF_COMMON_GLSL

// Depth of field effect.

// Depth of field effect. Original from keijiro san's implement, slightly change to fit my engine.
// https://github.com/keijiro/KinoBokeh


// NOTE: keijiro san blur Near and far together, sometimes looks bad. eg. front blur's bleeding is small, and sometimes the composite looks no good.
//       Other engine like unreal engine/CryEngine, they pre-multiply Coc, and blur near far with different RT, and they also Tile Coc to find Maximum/Minimum gay.
//       So the final near blur will bleeding to src image, which help to natural transition.
//
//       BTW, it still is a good dof for current engine, dof only open when miku dance, we always focus miku chan, so near blur is un-importance.

#extension GL_EXT_samplerless_texture_functions : enable

#include "Common.glsl"
#include "Schedule.glsl"
#include "VogelDisk.glsl"

layout (set = 0, binding = 0) uniform texture2D inDepth;
layout (set = 0, binding = 1) uniform texture2D inHDRSceneColor;

layout (set = 0, binding = 2, rgba16f) uniform image2D downSampleHDRImage; // Half res
layout (set = 0, binding = 3) uniform texture2D inDownSampleHDRImage; // Half res.

layout (set = 0, binding = 4, rgba16f) uniform image2D HDRSceneColorImage; 

layout (set = 0, binding = 5, rgba16f) uniform image2D gatherImage;
layout (set = 0, binding = 6) uniform texture2D inGatherImage;

layout (set = 0, binding = 7, rgba16f) uniform image2D expandFillImage;
layout (set = 0, binding = 8) uniform texture2D inExpandFillImage;

layout (set = 0, binding = 9) uniform texture2D inGbufferA;

struct DofDepthRange
{
    uint minDepth;
    uint maxDepth;
    uint sumPmxDepth;
    uint pmxPixelCount;
};

// Depth range min max buffer
layout(set = 0, binding = 10) buffer SSBODepthRangeBuffer { DofDepthRange depthRange; };


// Other common set.
layout (set = 1, binding = 0) uniform UniformView  { ViewData  viewData;  };
layout (set = 2, binding = 0) uniform UniformFrame { FrameData frameData; };

// Common sampler set.
#define COMMON_SAMPLER_SET 3
#include "CommonSamplerSet.glsl"

#define BLUE_NOISE_TEXTURE_SET 4
#define BLUE_NOISE_BUFFER_SET 5
#include "BlueNoiseCommon.glsl"

layout(push_constant) uniform PushConsts
{   
    int bNearBlur;
    int bFocusPMXCharacter;

    float distanceF; // focus distance
    float lensCoeff;
    float maxCoc;
    float maxCoCRcp;
    float aspectRcp;

    float focusLen;
    float filmHeight;
    float fStop;
} DofPush;

bool shouldSkipRenderDof()
{
    return (DofPush.bFocusPMXCharacter > 0) && (depthRange.pmxPixelCount == 0);
}

#endif