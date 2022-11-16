#version 460

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

layout (push_constant) uniform PushConsts 
{  
    vec4 prefilterFactor;
    float bloomIntensity;
    float bloomBlur;
};

#include "TonemapperFunction.glsl"
#include "BasicBloomCommon.glsl"
#include "ColorSpace.glsl"

// Gamma curve encode to srgb.
vec3 encodeSRGB(vec3 linearRGB)
{
    return pow(linearRGB, vec3(1.0 / 2.2));

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

    // Standard encode 10000 nit. XD
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
    vec3 bloomColor = upscampleTentFilter(uv, inBloomTexture, linearClampEdgeSampler, bloomBlur);
    


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

    // Linear rec 2020 color space.
    vec3 colorRec2020 = max(hdrColor.xyz, vec3(0.0));


    
    // TODO: Maybe use ACES color space in the future. then we can do some hdr color gradient in aces color space.
    //

    const bool bSrgb = frameData.toneMapper.displayMode == 0;

    const float P = bSrgb ? min(1.0, frameData.toneMapper.tonemapper_P) : frameData.toneMapper.tonemapper_P; // max display brightness.
    const float a = frameData.toneMapper.tonemapper_a;  // contrast
    const float m = frameData.toneMapper.tonemapper_m; // linear section start
    const float l = frameData.toneMapper.tonemapper_l;  // linear section length
    const float c = frameData.toneMapper.tonemapper_c; // black
    const float b = frameData.toneMapper.tonemapper_b;  // pedestal

    // Tonemapper in rec 2020 color space.
    colorRec2020.x = uchimuraTonemapper(colorRec2020.x, P, a, m, l, c, b);
    colorRec2020.y = uchimuraTonemapper(colorRec2020.y, P, a, m, l, c, b);
    colorRec2020.z = uchimuraTonemapper(colorRec2020.z, P, a, m, l, c, b);

    vec3 mappingColor;
    if(bSrgb) // Gamma encode srgb
    {
        vec3 colorSrgb = Rec2020_2_sRGB * colorRec2020;

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
        const float LinearToNitsScale = 100.0 * frameData.toneMapper.tonemmaper_s;
        const float LinearToNitsScaleInverse = 1.0 / LinearToNitsScale;

        //
        // OETF = inverse pq
        mappingColor.xyz = encodeST2084(colorRec2020.xyz * LinearToNitsScale);
    }
    
    imageStore(outLdrColor, workPos, vec4(mappingColor, 1.0));
}