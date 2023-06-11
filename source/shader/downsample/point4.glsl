#version 460

layout(set = 0, binding = 0) uniform texture2D in0;
layout(set = 0, binding = 1) uniform texture2D in1;
layout(set = 0, binding = 2) uniform texture2D in2;
layout(set = 0, binding = 3) uniform texture2D in3;
layout(set = 0, binding = 4) uniform writeonly image2D out0;
layout(set = 0, binding = 5) uniform writeonly image2D out1;
layout(set = 0, binding = 6) uniform writeonly image2D out2;
layout(set = 0, binding = 7) uniform writeonly image2D out3;

#define SHARED_SAMPLER_SET 1
#include "../common/shared_sampler.glsl"

#include "../common/shared_functions.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
void main()
{
    ivec2 colorSize = imageSize(out0);

    uvec2 groupThreadId = remap8x8(gl_LocalInvocationIndex);
    uvec2 dispatchId = groupThreadId + gl_WorkGroupID.xy * 8;
    ivec2 workPos = ivec2(dispatchId);
    if(workPos.x >= colorSize.x || workPos.y >= colorSize.y)
    {
        return;
    }
    const vec2 uv = (vec2(workPos) + vec2(0.5f)) / vec2(colorSize);

    imageStore(out0, workPos, texture(sampler2D(in0, pointClampEdgeSampler), uv));
    imageStore(out1, workPos, texture(sampler2D(in1, pointClampEdgeSampler), uv));
    imageStore(out2, workPos, texture(sampler2D(in2, pointClampEdgeSampler), uv));
    imageStore(out3, workPos, texture(sampler2D(in3, pointClampEdgeSampler), uv));
}