#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable

layout(set = 0, binding = 0) uniform texture2D inputTexture;
layout(set = 0, binding = 1) uniform texture2D inputCurTexture;
layout(set = 0, binding = 2, rgba16f)  uniform image2D hdrUpscale;

#define COMMON_SAMPLER_SET 1
#include "CommonSamplerSet.glsl"

layout (push_constant) uniform PushConsts 
{  
    float blurRadius;
};

#include "BasicBloomCommon.glsl"

layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    ivec2 upscaleSize = imageSize(hdrUpscale);
    ivec2 workPos = ivec2(gl_GlobalInvocationID.xy);
    if(workPos.x >= upscaleSize.x || workPos.y >= upscaleSize.y)
    {
        return;
    }

    vec2 uv = (vec2(workPos) + vec2(0.5)) / vec2(upscaleSize);
    vec3 outColor = vec3(0.0);

    // Upscale
    {
        vec3 currentColor = texture(sampler2D(inputCurTexture, linearClampEdgeSampler), uv).rgb;

#if MIX_BLOOM_UPSCALE 
        outColor = mix(currentColor, upscampleTentFilter(uv, inputTexture, linearClampEdgeSampler, blurRadius), vec3(blurRadius));
#else
        outColor = currentColor + upscampleTentFilter(uv, inputTexture, linearClampEdgeSampler, blurRadius);
#endif

    }
    imageStore(hdrUpscale, workPos, vec4(outColor, 1.0f));
}