#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable

#define SHARED_SAMPLER_SET    1
#include "common_shader.glsl"

layout (set = 0, binding = 0, rgba16f)  uniform image2D hdrSceneColor;
layout (set = 0, binding = 1)  uniform texture2D inDepth;
layout (set = 0, binding = 2)  uniform texture2D inGbufferA;
layout (set = 0, binding = 3)  uniform texture2D inGbufferB;
layout (set = 0, binding = 4)  uniform texture2D inGbufferS;
layout (set = 0, binding = 5)  uniform UniformFrameData { PerFrameData frameData; };
layout (set = 0, binding = 6)  uniform texture2D inBRDFLut;
layout (set = 0, binding = 7)  uniform texture2D inTransmittanceLut;
layout (set = 0, binding = 8)  uniform texture2D inSunShadowMask;
layout (set = 0, binding = 9)  uniform texture2D inBentNormalSSAO;
layout (set = 0, binding = 10)  uniform texture2D inAdaptedLumTex;
#include "common_lighting.glsl"

layout (local_size_x = 8, local_size_y = 8) in;
void main()
{   
    ivec2 sceneColorSize = imageSize(hdrSceneColor);

    uvec2 groupThreadId = remap8x8(gl_LocalInvocationIndex);
    uvec2 dispatchId = groupThreadId + gl_WorkGroupID.xy * 8;
    ivec2 workPos = ivec2(dispatchId);

    if(workPos.x >= sceneColorSize.x || workPos.y >= sceneColorSize.y)
    {
        return;
    }
    const vec2 uv = (vec2(workPos) + vec2(0.5f)) / vec2(sceneColorSize);

    // Load value from Gbuffer.
    const vec4 inSceneColorValue = imageLoad(hdrSceneColor, workPos);
    const vec4 inGbufferAValue = texelFetch(inGbufferA, workPos, 0);
    const vec4 inGbufferBValue = texelFetch(inGbufferB, workPos, 0);
    const vec4 inGbufferSValue = texelFetch(inGbufferS, workPos, 0);

    vec4 inSSAOBentNormal = loadBentNormalSSAO(
        texture(sampler2D(inBentNormalSSAO, pointClampEdgeSampler), uv), 
        vec4(inGbufferBValue.xyz, 1.0), frameData);

    const float deviceZ = texelFetch(inDepth, workPos, 0).r;
    float autoExposure = getExposure(frameData, inAdaptedLumTex);
    // Start basic lighting parameter prepare.
    const vec3 emissiveColor = inSceneColorValue.rgb / autoExposure; // exposure scale.

    const vec3 f0 = vec3(0.04);
    const vec3 baseColor = inGbufferAValue.rgb;
    float metallic = inGbufferSValue.r;
    float perceptualRoughness = inGbufferSValue.g;
    float meshAo = inGbufferSValue.b;
    vec3 diffuseColor = baseColor * (vec3(1.0) - f0) * (1.0 - metallic);
    vec3 specularColor = mix(f0, baseColor.rgb, metallic);
    perceptualRoughness = clamp(perceptualRoughness, 0.0, 1.0);

    // On Physically Based Shading at Disney. http://blog.selfshadow.com/publications/s2012-shading-course/burley/s2012_pbs_disney_brdf_notes_v3.pdf
    // Roughness is authored as perceptual roughness, convert to material roughness by squaring the perceptual roughness.
    float alphaRoughness = perceptualRoughness * perceptualRoughness;

    // Compute reflectance.
    // Reference from AMD's physical based rendering sample on https://github.com/GPUOpen-Effects/FidelityFX-SSSR
    float reflectance = max(max(specularColor.r, specularColor.g), specularColor.b);
    vec3 specularEnvironmentR0 = specularColor.rgb;

    // Anything less than 2% is physically impossible and is instead considered to be shadowing. Compare to "Real-Time-Rendering" 4th editon on page 325.
    vec3 specularEnvironmentR90 = vec3(clamp(reflectance * 50.0, 0.0, 1.0));



    vec3 color = vec3(0.0);
    vec3 normal = unpackWorldNormal(inGbufferBValue.rgb);
    vec3 worldPos = getWorldPos(uv, deviceZ, frameData);
    vec3 view = normalize(frameData.camWorldPos.xyz - worldPos);

    float intervalNoise = interleavedGradientNoise(workPos.xy, frameData.frameIndex.x % frameData.jitterPeriod);

    vec3 specularTerm = vec3(0.0);
    vec3 diffuseTerm = vec3(0.0);

    // PBR material build.
    PBRMaterial material;
    material.perceptualRoughness = perceptualRoughness;
    material.alphaRoughness = alphaRoughness;
    material.diffuseColor = diffuseColor;
    material.specularColor = specularColor;
    material.reflectance0 = specularEnvironmentR0;
    material.reflectance90 = specularEnvironmentR90;
    material.shadingModel = unpackShadingModelId(inGbufferAValue.a);
    material.baseColor = baseColor;
    material.curvature = inGbufferSValue.w;

    // Importance lights direct lighting evaluate.
    vec3 directColor = vec3(0.0f);

    float ao = inSSAOBentNormal.w;
    vec3 multiBounceAO = AoMultiBounce(ao, baseColor);

    // sky light shading.
    if(frameData.bSkyComponentValid != 0)
    {
        // Sun.
        {
            vec3 atmosphereTransmittance = vec3(1.0);
            // Second evaluate transmittance due to participating media
            {
                AtmosphereParameters atmosphere = getAtmosphereParameters(frameData);
                vec3 P0 = worldPos * 0.001 + vec3(0.0, atmosphere.bottomRadius, 0.0); // meter -> kilometers.
                float viewHeight = length(P0);
                const vec3 upVector = P0 / viewHeight;

                float viewZenithCosAngle = dot(-normalize(frameData.sunLightInfo.direction), upVector);
                vec2 sampleUv;
                lutTransmittanceParamsToUv(atmosphere, viewHeight, viewZenithCosAngle, sampleUv);
                atmosphereTransmittance = texture(sampler2D(inTransmittanceLut, linearClampEdgeSampler), sampleUv).rgb;
            }

            vec4 sunMask = texelFetch(inSunShadowMask, workPos, 0);
            float sunShadowMaskValue = sunMask.x;

            vec3 sunVisibility = vec3(sunShadowMaskValue);

            if(sunShadowMaskValue > 0.0f && sunShadowMaskValue < 1.0f)
            {
                sunVisibility = mix(
                    sunShadowMaskValue * frameData.sunLightInfo.shadowColor * frameData.sunLightInfo.shadowColorIntensity,
                    vec3(sunShadowMaskValue), 
                    vec3(sunShadowMaskValue));
            } 

            sunVisibility = min(sunVisibility, vec3(sunMask.y));

            ShadingResult sunShadeResult = evaluateSkyDirectLight(frameData.sunLightInfo, material, normal, view);

            specularTerm += sunShadeResult.specularTerm * atmosphereTransmittance * sunVisibility;
            diffuseTerm  += sunShadeResult.diffuseTerm  * atmosphereTransmittance * sunVisibility;
        }

    }

    // Emissive color.
    color += emissiveColor;

    if(material.shadingModel == EShadingModelType_DefaultLit)
    {
        // Specular and diffuse term.
        color += (specularTerm + diffuseTerm) * multiBounceAO;

        // Store in scene color.
        imageStore(hdrSceneColor, workPos, vec4(color, inSceneColorValue.a));
    }
}