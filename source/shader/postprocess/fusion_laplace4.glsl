#version 460

layout(set = 0, binding = 0) uniform texture2D inGaussianUpscale0;
layout(set = 0, binding = 1) uniform texture2D inGaussianUpscale1;
layout(set = 0, binding = 2) uniform texture2D inGaussianUpscale2;
layout(set = 0, binding = 3) uniform texture2D inGaussianUpscale3;
layout(set = 0, binding = 4) uniform texture2D inHighRes0;
layout(set = 0, binding = 5) uniform texture2D inHighRes1;
layout(set = 0, binding = 6) uniform texture2D inHighRes2;
layout(set = 0, binding = 7) uniform texture2D inHighRes3;
layout(set = 0, binding = 8) uniform writeonly image2D out0;
layout(set = 0, binding = 9) uniform writeonly image2D out1;
layout(set = 0, binding = 10) uniform writeonly image2D out2;
layout(set = 0, binding = 11) uniform writeonly image2D out3;

#define SHARED_SAMPLER_SET 1
#include "../common/shared_sampler.glsl"

#include "post_common.glsl"

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
    const vec2 texelSize = 1.0 / vec2(colorSize);
    const vec2 uv = (vec2(workPos) + vec2(0.5f)) * texelSize;

    {
        vec4 srcColor = texture(sampler2D(inHighRes0, pointClampEdgeSampler), uv);
        vec4 upscaleColor = texture(sampler2D(inGaussianUpscale0, pointClampEdgeSampler), uv);
        vec4 result = srcColor - upscaleColor;
        imageStore(out0, workPos, result);
    }

    {
        vec4 srcColor = texture(sampler2D(inHighRes1, pointClampEdgeSampler), uv);
        vec4 upscaleColor = texture(sampler2D(inGaussianUpscale1, pointClampEdgeSampler), uv);
        vec4 result = srcColor - upscaleColor;
        imageStore(out1, workPos, result);
    }

    {
        vec4 srcColor = texture(sampler2D(inHighRes2, pointClampEdgeSampler), uv);
        vec4 upscaleColor = texture(sampler2D(inGaussianUpscale2, pointClampEdgeSampler), uv);
        vec4 result = srcColor - upscaleColor;
        imageStore(out2, workPos, result);
    }

    {
        vec4 srcColor = texture(sampler2D(inHighRes3, pointClampEdgeSampler), uv);
        vec4 upscaleColor = texture(sampler2D(inGaussianUpscale3, pointClampEdgeSampler), uv);
        vec4 result = srcColor - upscaleColor;
        imageStore(out3, workPos, result);
    }
}