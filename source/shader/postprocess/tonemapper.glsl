#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable

#include "post_common.glsl"

layout (set = 0, binding = 0) uniform texture2D inHDRSceneColor;
layout (set = 0, binding = 1, rgba8)  uniform writeonly image2D outLdrColor;
layout (set = 0, binding = 2) uniform UniformFrameData { PerFrameData frameData; };

#define SHARED_SAMPLER_SET 1
#include "../common/shared_sampler.glsl"

#define BLUE_NOISE_BUFFER_SET 2
#include "../common/shared_bluenoise.glsl"

layout (push_constant) uniform PushConsts 
{  
    uint bExposureFusion;
};

layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    ivec2 colorSize = imageSize(outLdrColor);

    uvec2 groupThreadId = remap8x8(gl_LocalInvocationIndex);
    uvec2 dispatchId = groupThreadId + gl_WorkGroupID.xy * 8;
    ivec2 workPos = ivec2(dispatchId);
    if(workPos.x >= colorSize.x || workPos.y >= colorSize.y)
    {
        return;
    }
    const vec2 uv = (vec2(workPos) + vec2(0.5f)) / vec2(colorSize);

    // load color.
    // HDR color in linear srgb color space.
    vec3 encodeColor = texelFetch(inHDRSceneColor, workPos, 0).xyz;
    if(bExposureFusion == 0)
    {
        encodeColor = toneMapperFunction(encodeColor);
    }

    // Offset retarget for new seeds each frame
    uvec2 offset = uvec2(vec2(0.754877669, 0.569840296) * frameData.frameIndex.x * uvec2(colorSize));
    uvec2 offsetId = dispatchId.xy + offset;
    offsetId.x = offsetId.x % colorSize.x;
    offsetId.y = offsetId.y % colorSize.y;

    // Display is 8bit, so jitter with blue noise with [-1, 1] / 255.0.
    // Current also looks good in hdr display even it under 11bit.
    encodeColor.x += 1.0 / 255.0 * (-1.0 + 2.0 * samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d(offsetId.x, offsetId.y, 0, 0u));
    encodeColor.y += 1.0 / 255.0 * (-1.0 + 2.0 * samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d(offsetId.x, offsetId.y, 0, 1u));
    encodeColor.z += 1.0 / 255.0 * (-1.0 + 2.0 * samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d(offsetId.x, offsetId.y, 0, 2u));
    
    // Safe color.
    encodeColor.xyz = max(encodeColor.xyz, vec3(0.0));

    // Final store.
    imageStore(outLdrColor, workPos, vec4(encodeColor, 1.0f));
}