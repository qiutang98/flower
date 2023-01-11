#version 460

#include "KinoBokehDof_Common.glsl"

layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    if(shouldSkipRenderDof())
    {
        return;
    }

    ivec2 colorSize = imageSize(expandFillImage);

    uvec2 groupThreadId = remap8x8(gl_LocalInvocationIndex);
    uvec2 dispatchId = groupThreadId + gl_WorkGroupID.xy * 8;
    ivec2 workPos = ivec2(dispatchId);

    if(workPos.x >= colorSize.x || workPos.y >= colorSize.y)
    {
        return;
    }

    const vec2 texelSize = 1.0f / vec2(colorSize);
    const vec2 uv = (vec2(workPos) + vec2(0.5f)) * texelSize;

    vec4 result;

    // Simple 9x9 tent filter.
    result  = texture(sampler2D(inGatherImage, linearClampEdgeSampler), uv + texelSize * vec2(-1.0, -1.0));
    result += texture(sampler2D(inGatherImage, linearClampEdgeSampler), uv + texelSize * vec2( 0.0, -1.0)) * 2.0;
    result += texture(sampler2D(inGatherImage, linearClampEdgeSampler), uv + texelSize * vec2( 1.0, -1.0));

    result += texture(sampler2D(inGatherImage, linearClampEdgeSampler), uv + texelSize * vec2(-1.0,  0.0)) * 2.0;
    result += texture(sampler2D(inGatherImage, linearClampEdgeSampler), uv + texelSize * vec2( 0.0,  0.0)) * 4.0;
    result += texture(sampler2D(inGatherImage, linearClampEdgeSampler), uv + texelSize * vec2( 1.0,  0.0)) * 2.0;

    result += texture(sampler2D(inGatherImage, linearClampEdgeSampler), uv + texelSize * vec2(-1.0,  1.0));
    result += texture(sampler2D(inGatherImage, linearClampEdgeSampler), uv + texelSize * vec2( 0.0,  1.0)) * 2.0;
    result += texture(sampler2D(inGatherImage, linearClampEdgeSampler), uv + texelSize * vec2( 1.0,  1.0));

    result /= 16.0f;

    imageStore(expandFillImage, workPos, result);
}