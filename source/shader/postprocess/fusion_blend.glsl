#version 460

layout(set = 0, binding = 0) uniform texture2D inLaplace0;
layout(set = 0, binding = 1) uniform texture2D inLaplace1;
layout(set = 0, binding = 2) uniform texture2D inLaplace2;
layout(set = 0, binding = 3) uniform texture2D inLaplace3;
layout(set = 0, binding = 4) uniform texture2D inWeight;
layout(set = 0, binding = 5) uniform writeonly image2D outA;

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

    vec4 weights = texture(sampler2D(inWeight, pointClampEdgeSampler), uv);

    vec4 laplace0 = texture(sampler2D(inLaplace0, pointClampEdgeSampler), uv);
    vec4 laplace1 = texture(sampler2D(inLaplace1, pointClampEdgeSampler), uv);
    vec4 laplace2 = texture(sampler2D(inLaplace2, pointClampEdgeSampler), uv);
    vec4 laplace3 = texture(sampler2D(inLaplace3, pointClampEdgeSampler), uv);

    vec4 result = weights.x * laplace0 + weights.y * laplace1 + weights.z * laplace2 + weights.w * laplace3;

    imageStore(outA, workPos, result);
}