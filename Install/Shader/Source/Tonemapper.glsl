#version 460

/*
** Physical based render code, develop by engineer: qiutanguu.
*/

#extension GL_EXT_samplerless_texture_functions : enable
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable
#extension GL_ARB_gpu_shader_fp64 : enable

layout (set = 0, binding = 0) uniform texture2D inHDRSceneColor;
layout (set = 0, binding = 1, rgba8)  uniform writeonly image2D outLdrColor;
layout (set = 0, binding = 2) uniform texture2D inBloomTexture;
layout (set = 0, binding = 3) uniform texture2D inAdaptedLumTex;

#define COMMON_SAMPLER_SET 1
#include "CommonSamplerSet.glsl"
#include "Common.glsl"

layout (set = 2, binding = 0) uniform UniformView  { ViewData  viewData;  };
layout (set = 3, binding = 0) uniform UniformFrame { FrameData frameData; };

// Temporal blue noise jitter is hard to stable resolve. :(
// Maybe simple blue noise is better.
#define BLUE_NOISE_TEXTURE_SET 4
#define BLUE_NOISE_BUFFER_SET 5
#include "BlueNoiseCommon.glsl"
#include "Deband16.glsl"

layout (push_constant) uniform PushConsts 
{  
    vec4 prefilterFactor;
    float bloomIntensity;
    float bloomBlur;

    float tonemapper_P;  // Max brightness.
    float tonemapper_a;    // contrast
    float tonemapper_m;   // linear section start
    float tonemapper_l;    // linear section length
    float tonemapper_c;   // black
    float tonemapper_b;    // pedestal
    float tonemmaper_s;  // scale 
    uint bDisplayHDR_rec2020_PQ;     // HDR display?
    float saturation;
};

#include "TonemapperFunction.glsl"
#include "BasicBloomCommon.glsl"
#include "ColorSpace.glsl"

// Gamma curve encode to srgb.
vec3 encodeSRGB(vec3 linearRGB)
{
    // Most PC Monitor is 2.2 Gamma, this function is enough.
    return pow(linearRGB, vec3(1.0 / 2.2));

    // TV encode Rec709 encode.
    vec3 a = 12.92 * linearRGB;
    vec3 b = 1.055 * pow(linearRGB, vec3(1.0 / 2.4)) - 0.055;
    vec3 c = step(vec3(0.0031308), linearRGB);
    return mix(a, b, c);
}

// PQ curve encode to ST2084.
vec3 encodeST2084(vec3 linearRGB)
{
	const float m1 = 2610. / 4096. * .25;
	const float m2 = 2523. / 4096. *  128;
	const float c1 = 2392. / 4096. * 32 - 2413. / 4096. * 32 + 1;
	const float c2 = 2413. / 4096. * 32;
	const float c3 = 2392. / 4096. * 32;

    // Standard encode 10000 nit, same with standard DaVinci Resolve nit.
    float C = 10000.0;

	   vec3 L = linearRGB / C;
    // vec3 L = linearRGB;

	vec3 Lm = pow(L, vec3(m1));
	vec3 N1 = (c1 + c2 * Lm);
	vec3 N2 = (1.0 + c3 * Lm);
	vec3 N = N1 / N2;

	vec3 P = pow(N, vec3(m2));
	return P;
}

float lumaSRGB(vec3 c)
{
    return 0.212 * c.r + 0.701 * c.g + 0.087 * c.b;
}

vec3 applySaturationSRGB(vec3 c, float saturation)
{
    return mix(vec3(lumaSRGB(c)), c, saturation);
}

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

    // Compute uv basic on out ldr color.
    const vec2 uv = (vec2(workPos) + vec2(0.5f)) / vec2(colorSize);

    // load color.
    vec4 hdrColor = texture(sampler2D(inHDRSceneColor, pointClampEdgeSampler), uv);
     

    // Build bloom color.
    vec3 bloomColor = texture(sampler2D(inBloomTexture, linearClampEdgeSampler), uv).rgb;

    // Auto exposure.
    #if 1
    {
        float adaptiveLum = texelFetch(inAdaptedLumTex, ivec2(0, 0), 0).r;
        hdrColor.xyz *= adaptiveLum, viewData.evCompensation;
        // 
    }
    #elif 0
    {
        hdrColor.xyz *= viewData.exposure;
    }
    #endif

        // Add bloom feed if mix output.
#if MIX_BLOOM_OUTPUT
    bloomColor += hdrColor.xyz - prefilter(hdrColor.xyz, prefilterFactor);
#endif
    
    
    // Bloom composition.
#if MIX_BLOOM_OUTPUT
    hdrColor.xyz = mix(hdrColor.xyz, bloomColor, bloomIntensity);
#else
    hdrColor.xyz += bloomColor * bloomIntensity;
#endif

    // Linear srgb color space.
    vec3 colorSrgb = max(hdrColor.xyz, vec3(0.0));


    
    // TODO: Maybe use ACES color space in the future. then we can do some hdr color gradient in aces color space.
    //

    const bool bSrgb = bDisplayHDR_rec2020_PQ == 0;

    const float P = bSrgb ? min(1.0, tonemapper_P) : tonemapper_P; // max display brightness.
    const float a = tonemapper_a;  // contrast
    const float m = tonemapper_m; // linear section start
    const float l = tonemapper_l;  // linear section length
    const float c = tonemapper_c; // black
    const float b = tonemapper_b;  // pedestal

    // Tonemapper in srgb color space.
    colorSrgb.x = uchimuraTonemapper(colorSrgb.x, P, a, m, l, c, b);
    colorSrgb.y = uchimuraTonemapper(colorSrgb.y, P, a, m, l, c, b);
    colorSrgb.z = uchimuraTonemapper(colorSrgb.z, P, a, m, l, c, b);

    colorSrgb = applySaturationSRGB(colorSrgb, saturation);

    vec3 mappingColor;
    if(bSrgb) // Gamma encode srgb
    {
        // OETF = gamma(1.0/2.2)
        mappingColor.xyz = encodeSRGB(colorSrgb.xyz);


    }   
    else // PQ encode BT2020.
    {
        // Scale factor for converting pixel values to nits. 
        // This value is required for PQ (ST2084) conversions, because PQ linear values are in nits. 
        // The purpose is to make good use of PQ lut entries. A scale factor of 100 conveniently places 
        // about half of the PQ lut indexing below 1.0, with the other half for input values over 1.0.
        // Also, 100nits is the expected monitor brightness for a 1.0 pixel value without a tone curve.
        const float LinearToNitsScale = 100.0 * tonemmaper_s;
        const float LinearToNitsScaleInverse = 1.0 / LinearToNitsScale;

        vec3 colorRec2020 = sRGB_2_Rec2020 * colorSrgb;
        //
        // OETF = inverse pq
        mappingColor.xyz = encodeST2084(colorRec2020.xyz * LinearToNitsScale);
    }



    
    // TODO: Need jitter in R11G10B11 when in HDR display mode, but current looks good, so just use blue noise 8bit jitter.
    // mappingColor.xyz = quantise(mappingColor.xyz, vec2(workPos), frameData);

    // Offset retarget for new seeds each frame
    uvec2 offset = uvec2(vec2(0.754877669, 0.569840296) * frameData.frameIndex.x * uvec2(colorSize));
    uvec2 offsetId = dispatchId.xy + offset;
    offsetId.x = offsetId.x % colorSize.x;
    offsetId.y = offsetId.y % colorSize.y;

    // Display is 8bit, so jitter with blue noise with [-1, 1] / 255.0.
    // Current also looks good in hdr display even it under 11bit.
    mappingColor.x += 1.0 / 255.0 * (-1.0 + 2.0 * samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d(offsetId.x, offsetId.y, 0, 0u));
    mappingColor.y += 1.0 / 255.0 * (-1.0 + 2.0 * samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d(offsetId.x, offsetId.y, 0, 1u));
    mappingColor.z += 1.0 / 255.0 * (-1.0 + 2.0 * samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d(offsetId.x, offsetId.y, 0, 2u));
    
    mappingColor.xyz = max(mappingColor.xyz, vec3(0.0));
    imageStore(outLdrColor, workPos, vec4(mappingColor, 1.0));
}

// NOTE: if want to output hdr to DaVinci Resolve.
//       you need to add one ACES tonemapper fx, and input transform is ST2048. ouput tramsform is Rec709.