#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable

#include "../common/shared_functions.glsl"
#include "../bloom/bloom_common.glsl"

layout (set = 0, binding = 0) uniform texture2D inHDRSceneColor;
layout (set = 0, binding = 1, rgba8)  uniform writeonly image2D outLdrColor;
layout (set = 0, binding = 2) uniform texture2D inAdaptedLumTex;
layout (set = 0, binding = 3) uniform UniformFrameData { PerFrameData frameData; };
layout (set = 0, binding = 4) uniform texture2D inBloomTexture;

#define SHARED_SAMPLER_SET 1
#include "../common/shared_sampler.glsl"

#define BLUE_NOISE_BUFFER_SET 2
#include "../common/shared_bluenoise.glsl"

layout (push_constant) uniform PushConsts 
{  
    vec4 prefilterFactor;
    float bloomIntensity;
    float bloomBlur;
};

vec3 acesFilmFit(vec3 x)
{
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return saturate((x*(a*x+b))/(x*(c*x+d)+e));
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
    const vec2 uv = (vec2(workPos) + vec2(0.5f)) / vec2(colorSize);

    // load color.
    // HDR color in linear srgb color space.
    vec4 hdrColor = texelFetch(inHDRSceneColor, workPos, 0);
    hdrColor.xyz *= getExposure(frameData, inAdaptedLumTex).r;

    // Composite bloom color.
    vec3 bloomColor = texture(sampler2D(inBloomTexture, linearClampEdgeSampler), uv).rgb;
#if MIX_BLOOM_OUTPUT
    // Add bloom feed if mix output.
    bloomColor += hdrColor.xyz - prefilter(hdrColor.xyz, prefilterFactor);
    hdrColor.xyz = mix(hdrColor.xyz, bloomColor, bloomIntensity);
#else
    hdrColor.xyz += bloomColor * bloomIntensity;
#endif

    vec3 ldrColor;
#if 1
    {
        vec3 colorAP0 = hdrColor.xyz * sRGB_2_AP0;
        #if 1
            ldrColor = acesFilm(colorAP0);
        #else
            ldrColor = ACESOutputTransformsRGBD65(colorAP0);
        #endif
    }
#else
    {
        ldrColor = acesFilmFit(hdrColor.xyz);
    }
#endif

    ldrColor = WhiteBalance(ldrColor);
    
    // Encode ldr srgb or hdr.
    vec3 encodeColor;
    encodeColor = encodeSRGB(ldrColor.xyz);


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
    
    encodeColor.xyz = max(encodeColor.xyz, vec3(0.0));

    imageStore(outLdrColor, workPos, vec4(encodeColor, 1.0f));
}