#ifndef GTAO_COMMON_GLSL
#define GTAO_COMMON_GLSL

/*
** Physical based render code, develop by engineer: qiutanguu.
*/

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable

// Screen space ambient occlusion component for global illumination.
// Base on GTAO.
// See Activision GTAO paper: https://www.activision.com/cdn/research/s2016_pbs_activision_occlusion.pptx
// Ref implement:
// #0. https://github.com/EpicGames/UnrealEngine/blob/release/Engine/Shaders/Private/PostProcessAmbientOcclusion.usf
// #1. https://github.com/GameTechDev/XeGTAO
// #2. https://github.com/GPUOpen-Effects/FidelityFX-CACAO
// #3. http://m.cdn.blog.hu/da/darthasylum/tutorials/C++/ch54_gtao.html
// #4. https://github.com/PanosK92/SpartanEngine/blob/master/data/shaders/ssao.hlsl
// #5. https://github.com/GameTechDev/ASSAO

// Most code also ref from unreal engine GTAO impl version.
// TODO: Maybe i should try XeGTAO tech more, it look stable and quick, but i don't want to give up these code which already cost me about three weeks free time.
//       So continue use this version.


#include "Common.glsl"

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
layout (set = 0, binding = 15)  uniform texture2D inPrevGbufferB;

layout (set = 1, binding = 0) uniform UniformView { ViewData viewData; };
layout (set = 2, binding = 0) uniform UniformFrame { FrameData frameData; };

#define COMMON_SAMPLER_SET 3
#include "CommonSamplerSet.glsl"

layout(push_constant) uniform PushConsts
{   
    uint sliceNum;
    uint stepNum;
    float radius;
    float thickness;
    float power;
    float intensity;
} GTAOPush;

#include "FastMath.glsl"

#endif