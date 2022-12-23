#ifndef WATER_COMMON_GLSL
#define WATER_COMMON_GLSL

#include "Common.glsl"

layout (set = 0, binding = 0, rgba16f) uniform image2D imageHdrSceneColor;
layout (set = 0, binding = 1) uniform texture2D inHdrSceneColor;

layout (set = 0, binding = 2) uniform texture2D inDepth;
layout (set = 0, binding = 3) uniform texture2D inGBufferA;


#endif