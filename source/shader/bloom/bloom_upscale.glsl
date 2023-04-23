#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable

#include "bloom_common.glsl"

layout(set = 0, binding = 0) uniform texture2D inputTexture;
layout(set = 0, binding = 1) uniform texture2D inputCurTexture;
layout(set = 0, binding = 2, rgba16f)  uniform image2D hdrUpscale;

#define SHARED_SAMPLER_SET 1
#include "../common/shared_sampler.glsl"

layout (push_constant) uniform PushConsts 
{  
    uint  bBlurX;
    uint  bFinalBlur;
    uint  upscaleTime;
    float blurRadius;
};

const float kRadius = 16.;
const float kSigma = 0.996416;
float gaussianWeight(float x) 
{
    const float mu = 0; // From center.
    const float dx = x - mu;
    const float sigma2 = kSigma * kSigma;
    return 0.398942280401 / kSigma * exp(- (dx * dx) * 0.5 / sigma2);
}

const float kBlenWeight[6] = { 0.25, 0.75, 1.5, 2.0, 2.5, 3.0}; // Sum is 10.0, so when composite, need *
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

    vec2 inputTexelSize = 1.0 / vec2(upscaleSize);
    
    const bool bBlurXDirection = bBlurX > 0;
    vec2 blurDirection = bBlurXDirection ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    blurDirection *= inputTexelSize;

    vec4 sum = vec4(0.0);
    for(float i = -kRadius; i <= kRadius; i++)
    {
        float weight = gaussianWeight(i);
        sum.xyz += weight * texture(sampler2D(inputTexture, linearClampBorder0000Sampler), uv + i * blurDirection).xyz;
        sum.w += weight;
    }
    sum.xyz /= sum.w;

    if((!bBlurXDirection) && (bFinalBlur == 0))
    {
        vec3 currentColor = texture(sampler2D(inputCurTexture, linearClampEdgeSampler), uv).rgb;

#if MIX_BLOOM_UPSCALE 
        sum.xyz = mix(currentColor, sum.xyz, vec3(blurRadius));
#else
        sum.xyz = currentColor + sum.xyz * kBlenWeight[upscaleTime];
#endif
    }

#if !MIX_BLOOM_UPSCALE 
    sum.xyz *= (bFinalBlur == 1) ? 0.1 : 1.0;
#endif

    imageStore(hdrUpscale, workPos, vec4(sum.xyz, 1.0f));
}