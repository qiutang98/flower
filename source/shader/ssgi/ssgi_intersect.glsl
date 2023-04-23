#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable

#include "ssgi_common.glsl"

layout (set = 0, binding = 0) uniform texture2D inHistoryHdrColor;
layout (set = 0, binding = 1, rgba16f)  uniform writeonly image2D outIntersectColor;
layout (set = 0, binding = 2) uniform texture2D inGbufferB;
layout (set = 0, binding = 3) uniform UniformFrameData { PerFrameData frameData; };

#define SHARED_SAMPLER_SET 1
#include "../common/shared_sampler.glsl"

#define BLUE_NOISE_BUFFER_SET 2
#include "../common/shared_bluenoise.glsl"

layout (push_constant) uniform PushConsts 
{  
    float pad0;
};

layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    ivec2 colorSize = imageSize(outIntersectColor);

    uvec2 groupThreadId = remap8x8(gl_LocalInvocationIndex);
    uvec2 dispatchId = groupThreadId + gl_WorkGroupID.xy * 8;
    ivec2 workPos = ivec2(dispatchId);
    if(workPos.x >= colorSize.x || workPos.y >= colorSize.y)
    {
        return;
    }
    const vec2 uv = (vec2(workPos) + vec2(0.5f)) / vec2(colorSize);

    const vec4 inGbufferBValue = texelFetch(inGbufferB, workPos, 0);
    const vec3 worldNormal = inGbufferBValue.xyz;

    imageStore(outIntersectColor, workPos, vec4(worldNormal * 0.5 + 0.5, 1.0f));
}