#version 460

/*
** Physical based render code, develop by engineer: qiutanguu.
*/

#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable

struct VS2PS
{
    vec2 uv0;
};

#include "Common.glsl"
#include "ColorSpace.glsl"
#include "PMX_Common.glsl"

#include "StaticMeshCommon.glsl"
#include "SDSM_Common.glsl"

layout (set = 1, binding = 0) uniform UniformView  { ViewData  viewData;  };
layout (set = 2, binding = 0) uniform UniformFrame { FrameData frameData; };

#define COMMON_SAMPLER_SET 3
#include "CommonSamplerSet.glsl"

#define BLUE_NOISE_TEXTURE_SET 4
#define BLUE_NOISE_BUFFER_SET 5
#include "BlueNoiseCommon.glsl"

layout (set = 6, binding = 0) uniform texture2D bindlessTexture2D[];
layout (set = 7, binding = 0) uniform UniformPMXParam { UniformPMX pmxParam; };

layout(push_constant) uniform PushConsts
{   
    uint cascadeId;
    uint perCascadeMaxCount;
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
    const vec4 worldPosition = pmxParam.modelMatrix * localPosition;

    // Convert to clip space.
    gl_Position = cascadeInfos[cascadeId].viewProj * worldPosition;
}

#endif /////////////////////////// vertex shader end

#ifdef PIXEL_SHADER ////////////// pixel shader start 

vec4 tex(uint texId, vec2 uv)
{
    // PMX file use linear repeat to sample texture.
    return texture(sampler2D(bindlessTexture2D[nonuniformEXT(texId)], linearRepeatSampler), uv);
}

layout(location = 0) in VS2PS vsIn;

layout(location = 0) out vec4 outColor;

void main()
{
    vec4 baseColor = tex(pmxParam.texID, vsIn.uv0);
    if(baseColor.a < 0.01f)
    {
        // Clip very small mask.
        discard;
    }

    outColor = vec4(0.0f);
}

#endif //////////////////////////// pixel shader end