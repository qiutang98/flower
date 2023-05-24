#version 460
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable

struct VS2PS
{
    vec2 uv0;
};

#include "pmx_common.glsl"

layout (set = 0, binding = 0) uniform UniformFrameData { PerFrameData frameData; };
layout (set = 0, binding = 1) buffer SSBOCascadeInfoBuffer { CascadeInfo cascadeInfos[]; }; // Cascade infos.

#define SHARED_SAMPLER_SET 1
#include "../common/shared_sampler.glsl"

layout (set = 2, binding = 0) uniform  texture2D texture2DBindlessArray[];

layout(push_constant) uniform PushConsts
{   
    mat4 modelMatrix;
    uint colorTexId;
    uint cascadeId;
};

#ifdef VERTEX_SHADER ///////////// vertex shader start 

layout (location = 0) in vec3 inPosition;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inUV0;
layout (location = 3) in vec3 inPositionLast;

layout(location = 0) out VS2PS vsOut;

void main()
{
    // PMX GL style uv flip.
    vsOut.uv0 = inUV0 * vec2(1.0f, -1.0f); 

    // Local vertex position.
    const vec4 localPosition = vec4(inPosition, 1.0f);
    const vec4 worldPosition = modelMatrix * localPosition;

    // Convert to clip space.
    gl_Position = cascadeInfos[cascadeId].viewProj * worldPosition;
}

#endif /////////////////////////// vertex shader end

#ifdef PIXEL_SHADER ////////////// pixel shader start 

vec4 tex(uint texId, vec2 uv)
{
    // PMX file use linear repeat to sample texture.
    return texture(sampler2D(texture2DBindlessArray[nonuniformEXT(texId)], linearRepeatSampler), uv);
}

layout(location = 0) in VS2PS vsIn;

layout(location = 0) out vec4 outColor;

void main()
{
    vec4 baseColor = tex(colorTexId, vsIn.uv0);
    if(baseColor.a < 0.01f)
    {
        // Clip very small mask.
        discard;
    }

    outColor = vec4(0.0f);
}

#endif //////////////////////////// pixel shader end