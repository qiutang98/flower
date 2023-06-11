#version 460

layout(set = 0, binding = 0) uniform texture2D inGaussianUpscale;
layout(set = 0, binding = 1) uniform texture2D inHighRes;
layout(set = 0, binding = 2) uniform writeonly image2D outA;

#define SHARED_SAMPLER_SET 1
#include "../common/shared_sampler.glsl"

#include "post_common.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
void main()
{
    ivec2 colorSize = imageSize(outA);

    uvec2 groupThreadId = remap8x8(gl_LocalInvocationIndex);
    uvec2 dispatchId = groupThreadId + gl_WorkGroupID.xy * 8;
    ivec2 workPos = ivec2(dispatchId);
    if(workPos.x >= colorSize.x || workPos.y >= colorSize.y)
    {
        return;
    }
    const vec2 texelSize = 1.0 / vec2(colorSize);
    const vec2 uv = (vec2(workPos) + vec2(0.5f)) * texelSize;

    vec4 srcColor = texture(sampler2D(inHighRes, pointClampEdgeSampler), uv);
    vec4 upscaleColor = texture(sampler2D(inGaussianUpscale, pointClampEdgeSampler), uv);

    vec4 result = srcColor - upscaleColor;

    imageStore(outA, workPos, result);
}