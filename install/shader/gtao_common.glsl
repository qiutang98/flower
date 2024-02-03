#ifndef GTAO_COMMON_GLSL
#define GTAO_COMMON_GLSL

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable

#define SHARED_SAMPLER_SET 1
#include "common_shader.glsl"
#include "common_sampler.glsl"

layout (set = 0, binding = 0)  uniform texture2D inHiz;
layout (set = 0, binding = 1)  uniform texture2D inDepth;
layout (set = 0, binding = 2)  uniform texture2D inGbufferA;
layout (set = 0, binding = 3)  uniform texture2D inGbufferB;
layout (set = 0, binding = 4)  uniform texture2D inGbufferS;
layout (set = 0, binding = 5, r8) uniform image2D GTAOImage;
layout (set = 0, binding = 6) uniform texture2D inGTAO;
layout (set = 0, binding = 7, r8) uniform image2D GTAOFilterImage;
layout (set = 0, binding = 8) uniform texture2D inGTAOFilterImage;
layout (set = 0, binding = 9, r8) uniform image2D GTAOTempFilter; // current frame temporal filter result, 
layout (set = 0, binding = 10) uniform texture2D inGTAOTempFilter;  
layout (set = 0, binding = 11, r8) uniform image2D GTAOTemporalHistory; // history temporal result.
layout (set = 0, binding = 12) uniform texture2D inGTAOTemporalHistory; // 
layout (set = 0, binding = 13)  uniform texture2D inGbufferV;
layout (set = 0, binding = 14)  uniform texture2D inPrevDepth;
layout (set = 0, binding = 15) uniform UniformFrameData { PerFrameData frameData; };

layout(push_constant) uniform PushConsts
{   
    uint sliceNum;
    uint stepNum;
    float radius;
    float thickness;
    float power;
    float intensity;
    float kFalloffRadius;
} GTAOPush;

#endif