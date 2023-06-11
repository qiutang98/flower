#version 460

#include "post_common.glsl"

layout(set = 0, binding = 0) uniform texture2D inColor;
layout(set = 0, binding = 1) uniform writeonly image2D exposureImage0;
layout(set = 0, binding = 2) uniform writeonly image2D exposureImage1;
layout(set = 0, binding = 3) uniform writeonly image2D exposureImage2;
layout(set = 0, binding = 4) uniform writeonly image2D exposureImage3;

layout(push_constant) uniform PushConstants
{
    float black;
    float shadows;
    float highlights;
};

#define SHARED_SAMPLER_SET 1
#include "../common/shared_sampler.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
void main()
{
    ivec2 colorSize = imageSize(exposureImage0);

    uvec2 groupThreadId = remap8x8(gl_LocalInvocationIndex);
    uvec2 dispatchId = groupThreadId + gl_WorkGroupID.xy * 8;
    ivec2 workPos = ivec2(dispatchId);
    if(workPos.x >= colorSize.x || workPos.y >= colorSize.y)
    {
        return;
    }
    const vec2 uv = (vec2(workPos) + vec2(0.5f)) / vec2(colorSize);

    vec3 hdrColor = texture(sampler2D(inColor, pointClampEdgeSampler), uv).xyz;

    vec3 color0 = toneMapperFunction(hdrColor.xyz);
    vec3 color1 = toneMapperFunction(hdrColor.xyz * shadows);
    vec3 color2 = toneMapperFunction(hdrColor.xyz * highlights);
    vec3 color3 = toneMapperFunction(hdrColor.xyz * black);

    imageStore(exposureImage0, workPos, vec4(color0, 1.0));
    imageStore(exposureImage1, workPos, vec4(color1, 1.0));
    imageStore(exposureImage2, workPos, vec4(color2, 1.0));
    imageStore(exposureImage3, workPos, vec4(color3, 1.0));
}