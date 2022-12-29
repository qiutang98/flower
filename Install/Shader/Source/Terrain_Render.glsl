#version 460

/*
** Physical based render code, develop by engineer: qiutanguu.
*/

#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable

#include "Terrain_Common.glsl"

struct VS2PS
{
    vec2 uv0;
    vec3 worldPos;
    vec4 posNDCPrevNoJitter;
    vec4 posNDCCurNoJitter;
};

layout (set = 0, binding = 0) uniform UniformView{  ViewData viewData; };
layout (set = 1, binding = 0) uniform UniformFrame{ FrameData frameData; };
layout (set = 2, binding = 0) uniform texture2D bindlessTexture2D[];
layout (set = 3, binding = 0) uniform sampler bindlessSampler[];

#ifdef VERTEX_SHADER ///////////// vertex shader start 

layout (location = 0) in vec3 inPosition;
layout (location = 1) in vec2 inUV0;

void main()
{
    
}

#endif ///////////// vertex shader end

#ifdef PIXEL_SHADER ////////////// pixel shader start 

// Scene hdr color. .rgb store emissive color.
layout(location = 0) out vec4 outHDRSceneColor;

// GBuffer A: r8g8b8a8 unorm, .rgb store base color, .a is shading model id.
layout(location = 1) out vec4 outGBufferA;

// GBuffer B: r16g16b16a16 sfloat, .rgb store worldspace normal, .a is object id.
layout(location = 2) out vec4 outGBufferB;

// GBuffer S: r8g8b8a8 unorm, .r is metal, .g is roughness, .b is mesh ao.
layout(location = 3) out vec4 outGBufferS;

// GBuffer V: r16g16 sfloat, store velocity.
layout(location = 4) out vec2 outGBufferV;

void main()
{

}

#endif //////////////////////////// pixel shader end