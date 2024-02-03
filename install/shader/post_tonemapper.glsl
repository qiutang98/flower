#version 460

#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable

#define SHARED_SAMPLER_SET    1
#define BLUE_NOISE_BUFFER_SET 2
#include "common_shader.glsl"

layout (set = 0, binding = 0) uniform UniformFrameData { PerFrameData frameData; };
layout (set = 0, binding = 1) uniform texture2D inHDRSceneColor;
layout (set = 0, binding = 2, rgba8)  uniform writeonly image2D outLdrColor;
layout (set = 0, binding = 3) uniform texture2D inAdaptedLumTex;
layout (set = 0, binding = 4) uniform texture2D inBloomTexture;
layout (set = 0, binding = 5) buffer SSBOLensFlare { float ssboLensFlareDatas[]; };

#include "common_tonemapper.glsl"

#include "lens.glsl"

const float kACESExpandGamut = 1.0;

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

    // HDR color in acescg color space do tonemapper.
    vec3 hdrColor = texelFetch(inHDRSceneColor, workPos, 0).xyz;
    vec3 bloomColor = texture(sampler2D(inBloomTexture, linearClampEdgeSampler), uv).rgb;

    // Apply bloom.
    hdrColor += bloomColor * frameData.postprocessing.bloomIntensity;

    if(ssboLensFlareDatas[3] > 0.0f && -frameData.sunLightInfo.direction.y > 0.0f)
    {
        vec2 sunUv = projectPos(frameData.sunLightInfo.direction * 65535.0f, frameData.camViewProjNoJitter).xy;
        vec3 lensColor;
        vec3 ghostColor = lensFlare(uv, sunUv, lensColor);

        hdrColor.xyz += (ghostColor + lensColor) * vec3(
            ssboLensFlareDatas[0],
            ssboLensFlareDatas[1],
            ssboLensFlareDatas[2])  * ssboLensFlareDatas[3] * getExposure(frameData, inAdaptedLumTex).r;
    }

#if AP1_COLOR_SPACE
    // Already in AP1 color space, no need to fake expand color gamut.
    vec3 colorAP1 = hdrColor;
#else 
    vec3 colorAP1 = hdrColor * sRGB_2_AP1_MAT;
    if(frameData.postprocessing.expandGamutFactor > 0.0f)
    {
        // NOTE: Expand to wide gamut.
        //       We render in linear srgb color space, and tonemapper in acescg.
        //       acescg gamut larger than srgb, we use this expand matrix to disguise rendering in acescg color space.
        float lumaAP1 = dot(colorAP1, AP1_RGB2Y);
        vec3 chromaAP1 = colorAP1 / lumaAP1;

        float chromaDistSqr = dot(chromaAP1 - 1.0, chromaAP1 - 1.0);
        float expandAmount = 
            (1.0 - exp2(-4.0 * chromaDistSqr)) * 
            (1.0 - exp2(-4.0 * frameData.postprocessing.expandGamutFactor * lumaAP1 * lumaAP1));

        const mat3 Wide_2_XYZ_MAT = mat3(
            vec3( 0.5441691,  0.2395926,  0.1666943),
            vec3( 0.2394656,  0.7021530,  0.0583814),
            vec3(-0.0023439,  0.0361834,  1.0552183)
        );

        const mat3 Wide_2_AP1 = Wide_2_XYZ_MAT * XYZ_2_AP1_MAT;
        const mat3 expandMat = AP1_2_sRGB_MAT * Wide_2_AP1;

        vec3 colorExpand = colorAP1 * expandMat;
        colorAP1 = mix(colorAP1, colorExpand, expandAmount);
    }
#endif 

    // Apply tonemmapper.
    vec3 toneAp1 = tonemapperAp1(colorAP1, frameData);

    // Convert to srgb.
    vec3 srgbColor = toneAp1 * AP1_2_sRGB_MAT;

    // OETF part.
    // Encode to fit monitor gamma curve, current default use Rec.709 gamma encode. 
    vec3 encodeColor = encodeSRGB(srgbColor);

    // Dither RGB.
    {
        // Offset retarget for new seeds each frame
        uvec2 offsetId = jitterSequence(frameData.frameIndex.x, uvec2(colorSize), dispatchId);

        // Display is 8bit, so jitter with blue noise with [-1, 1] / 255.0.
        // Current also looks good in hdr display even it under 11bit.
        encodeColor.x += 1.0 / 255.0 * (-1.0 + 2.0 * samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d(offsetId.x, offsetId.y, 0, 0u));
        encodeColor.y += 1.0 / 255.0 * (-1.0 + 2.0 * samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d(offsetId.x, offsetId.y, 0, 1u));
        encodeColor.z += 1.0 / 255.0 * (-1.0 + 2.0 * samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d(offsetId.x, offsetId.y, 0, 2u));
        
        // Safe color.
        encodeColor.xyz = max(encodeColor.xyz, vec3(0.0));
    }

    // Final store.
    imageStore(outLdrColor, workPos, vec4(encodeColor, 1.0f));
}