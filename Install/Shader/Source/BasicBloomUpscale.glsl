#version 460

/*
** Physical based render code, develop by engineer: qiutanguu.
*/

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable

layout(set = 0, binding = 0) uniform texture2D inputTexture;
layout(set = 0, binding = 1) uniform texture2D inputCurTexture;
layout(set = 0, binding = 2, rgba16f)  uniform image2D hdrUpscale;

#define COMMON_SAMPLER_SET 1
#include "CommonSamplerSet.glsl"

layout (push_constant) uniform PushConsts 
{  
    uint  bBlurX;
    uint  bFinalBlur;
    uint  upscaleTime;
    float blurRadius;
};

#include "BasicBloomCommon.glsl"

/*
    //int lenght = 3;  float coeffs[] = { 0.382971, 0.241842, 0.060654, }; // norm = 0.987962
    //int lenght = 4;  float coeffs[] = { 0.292360, 0.223596, 0.099952, 0.026085, }; // norm = 0.991625
    //int lenght = 5;  float coeffs[] = { 0.235833, 0.198063, 0.117294, 0.048968, 0.014408, }; // norm = 0.993299
    //int lenght = 6;  float coeffs[] = { 0.197419, 0.174688, 0.121007, 0.065615, 0.027848, 0.009250, }; // norm = 0.994236
    //int lenght = 7;  float coeffs[] = { 0.169680, 0.155018, 0.118191, 0.075202, 0.039930, 0.017692, 0.006541, }; // norm = 0.994827
    //int lenght = 8;  float coeffs[] = { 0.148734, 0.138756, 0.112653, 0.079592, 0.048936, 0.026183, 0.012191, 0.004939, }; // norm = 0.995231
    //int lenght = 9;  float coeffs[] = { 0.132370, 0.125285, 0.106221, 0.080669, 0.054877, 0.033440, 0.018252, 0.008924, 0.003908, }; // norm = 0.995524
    //int lenght = 10; float coeffs[] = { 0.119237, 0.114032, 0.099737, 0.079779, 0.058361, 0.039045, 0.023890, 0.013368, 0.006841, 0.003201, }; // norm = 0.995746
    //int lenght = 11; float coeffs[] = { 0.108467, 0.104534, 0.093566, 0.077782, 0.060053, 0.043061, 0.028677, 0.017737, 0.010188, 0.005435, 0.002693, }; // norm = 0.995919
    //int lenght = 12; float coeffs[] = { 0.099477, 0.096435, 0.087850, 0.075206, 0.060500, 0.045736, 0.032491, 0.021690, 0.013606, 0.008021, 0.004443, 0.002313, }; // norm = 0.996058
    //int lenght = 13; float coeffs[] = { 0.091860, 0.089459, 0.082622, 0.072368, 0.060113, 0.047355, 0.035378, 0.025065, 0.016842, 0.010732, 0.006486, 0.003717, 0.002020, }; // norm = 0.996172
    //int lenght = 14; float coeffs[] = { 0.085325, 0.083397, 0.077868, 0.069455, 0.059181, 0.048172, 0.037458, 0.027824, 0.019744, 0.013384, 0.008667, 0.005361, 0.003168, 0.001789, }; // norm = 0.996267
    //int lenght = 15; float coeffs[] = { 0.079656, 0.078085, 0.073554, 0.066578, 0.057908, 0.048399, 0.038870, 0.029997, 0.022245, 0.015852, 0.010854, 0.007142, 0.004516, 0.002743, 0.001602, }; // norm = 0.996347
    //int lenght = 16; float coeffs[] = { 0.074693, 0.073396, 0.069638, 0.063796, 0.056431, 0.048197, 0.039746, 0.031648, 0.024332, 0.018063, 0.012947, 0.008961, 0.005988, 0.003864, 0.002407, 0.001448, }; // norm = 0.996416
*/
#define FIX_KERNAL 0
#if FIX_KERNAL
    // const int kLenght = 9; const float coeffs[9] = { 0.132370, 0.125285, 0.106221, 0.080669, 0.054877, 0.033440, 0.018252, 0.008924, 0.003908 }; // norm = 0.995524
    const int kLenght = 16;  const float coeffs[16] = { 0.074693, 0.073396, 0.069638, 0.063796, 0.056431, 0.048197, 0.039746, 0.031648, 0.024332, 0.018063, 0.012947, 0.008961, 0.005988, 0.003864, 0.002407, 0.001448 };// norm = 0.996416 
#else
    const float rad = 16.;
    const float sigma = 0.996416;
    float gaussianWeight(float x) 
    {
        const float mu = 0; // From center.
        const float dx = x - mu;
        const float sigma2 = sigma * sigma;
        return 0.398942280401 / sigma * exp(- (dx * dx) * 0.5 / sigma2);
    }
#endif

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


#if FIX_KERNAL
    vec4 sum = coeffs[0] * texture(sampler2D(inputTexture, linearClampBorder0000Sampler), uv );   
    for(int i = 1; i < kLenght; i++)
    {
        sum += coeffs[i] * texture(sampler2D(inputTexture, linearClampBorder0000Sampler), uv + i * blurDirection);
        sum += coeffs[i] * texture(sampler2D(inputTexture, linearClampBorder0000Sampler), uv - i * blurDirection);
    }
#else
    vec4 sum = vec4(0.0);
    for(float i = -rad; i <= rad; i++)
    {
        float weight = gaussianWeight(i);
        sum.xyz += weight * texture(sampler2D(inputTexture, linearClampBorder0000Sampler), uv + i * blurDirection).xyz;
        sum.w += weight;
    }
    sum.xyz /= sum.w;
#endif

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