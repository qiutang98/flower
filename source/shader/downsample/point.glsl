#version 460

layout(set = 0, binding = 0) uniform texture2D inA;
layout(set = 0, binding = 1) uniform writeonly image2D outA;

#define SHARED_SAMPLER_SET 1
#include "../common/shared_sampler.glsl"

#include "../common/shared_functions.glsl"

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
    const vec2 uv = (vec2(workPos) + vec2(0.5f)) / vec2(colorSize);

    imageStore(outA, workPos, texture(sampler2D(inA, pointClampEdgeSampler), uv));
}