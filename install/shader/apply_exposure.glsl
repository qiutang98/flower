#version 460

#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable

#define SHARED_SAMPLER_SET    1
#include "common_shader.glsl"

layout (set = 0, binding = 0) uniform UniformFrameData { PerFrameData frameData; };
layout (set = 0, binding = 1) uniform texture2D inHDRSceneColor;
layout (set = 0, binding = 2, rgba16f) uniform image2D outColor;
layout (set = 0, binding = 3) uniform texture2D inAdaptedLumTex;


layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    ivec2 colorSize = imageSize(outColor);

    uvec2 groupThreadId = remap8x8(gl_LocalInvocationIndex);
    uvec2 dispatchId = groupThreadId + gl_WorkGroupID.xy * 8;
    ivec2 workPos = ivec2(dispatchId);
    if(workPos.x >= colorSize.x || workPos.y >= colorSize.y)
    {
        return;
    }
    const vec2 uv = (vec2(workPos) + vec2(0.5f)) / vec2(colorSize);
    float autoExposure = getExposure(frameData, inAdaptedLumTex);

    // HDR color in acescg color space do tonemapper.
    vec3 hdrColor = texelFetch(inHDRSceneColor, workPos, 0).xyz * autoExposure;



    // Final store.
    imageStore(outColor, workPos, vec4(hdrColor, 1.0f));
}