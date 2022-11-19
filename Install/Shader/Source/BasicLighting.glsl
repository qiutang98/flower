#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable

// Directional lighting pass.
// Shadow. direct diffuse light. direct specular light.
#include "Common.glsl"

#include "LightingCommon.glsl"

layout (set = 0, binding = 0, rgba16f)  uniform image2D hdrSceneColor;
layout (set = 0, binding = 1)  uniform texture2D inDepth;
layout (set = 0, binding = 2)  uniform texture2D inGbufferA;
layout (set = 0, binding = 3)  uniform texture2D inGbufferB;
layout (set = 0, binding = 4)  uniform texture2D inGbufferS;
layout (set = 0, binding = 5)  uniform texture2D inSDSMShadowMask;
layout (set = 0, binding = 6)  uniform texture2D inBRDFLut;
layout (set = 0, binding = 7)  uniform texture2D inTransmittanceLut;
layout (set = 0, binding = 8)  uniform texture2D inMultiScatterLut;
layout (set = 0, binding = 9)  uniform texture2D inSkyViewLut;
layout (set = 0, binding = 10) uniform textureCube inCubeEnv;
layout (set = 0, binding = 11) uniform textureCube inCubeGlobalIrradiance;
layout (set = 0, binding = 12) uniform textureCube inCubeGlobalPrefilter;
layout (set = 0, binding = 13)  uniform texture2D inGTAO;

layout (set = 1, binding = 0) uniform UniformView { ViewData viewData; };
layout (set = 2, binding = 0) uniform UniformFrame { FrameData frameData; };

#define COMMON_SAMPLER_SET 3
#include "CommonSamplerSet.glsl"

layout(push_constant) uniform PushConsts
{   
    uint directionalLightShadowValid; // 0 is unvalid.
};

#include "TonemapperFunction.glsl"



#include "Schedule.glsl"

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
    const float deviceZ = texelFetch(inDepth, workPos, 0).r;

    // Start basic lighting parameter prepare.
    const vec3 emissiveColor = inSceneColorValue.rgb;
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
    vec3 normal = normalize(inGbufferBValue.rgb);
    vec3 worldPos = getWorldPos(uv, deviceZ, viewData);
    vec3 view = normalize(viewData.camWorldPos.xyz - worldPos);

    // PBR material build.
    PBRMaterial material;
    material.perceptualRoughness = perceptualRoughness;
    material.alphaRoughness = alphaRoughness;
    material.diffuseColor = diffuseColor;
    material.specularColor = specularColor;
    material.reflectance0 = specularEnvironmentR0;
    material.reflectance90 = specularEnvironmentR90;

    // Importance lights direct lighting evaluate.
    vec3 directColor = vec3(0.0f);

    // Directional light shading.
    vec3 atmosphereTransmittance = vec3(1.0);
    if(frameData.directionalLightCount > 0)
    {
        const DirectionalLightInfo evaluateLight = frameData.directionalLight;

        float shadowFactor = 1.0f; 
        if(directionalLightShadowValid > 0)
        {
            shadowFactor = texelFetch(inSDSMShadowMask, workPos, 0).r;
        }

        // Second evaluate transmittance due to participating media
        {
            AtmosphereParameters atmosphere = getAtmosphereParameters(frameData);
            vec3 P0 = worldPos + vec3(0.0, atmosphere.bottomRadius, 0.0);
            float viewHeight = length(P0);
            const vec3 upVector = P0 / viewHeight;

            float viewZenithCosAngle = dot(-normalize(evaluateLight.direction), upVector);
            vec2 sampleUv;
            lutTransmittanceParamsToUv(atmosphere, viewHeight, viewZenithCosAngle, sampleUv);
            atmosphereTransmittance = texture(sampler2D(inTransmittanceLut, linearClampEdgeSampler), sampleUv).rgb;
        }

        directColor += atmosphereTransmittance * shadowFactor * evaluateDirectionalLight(evaluateLight, material, normal, view);
    }

    // Other type light shading.
    // TODO....

    color += directColor;
    {
        vec3 diffuseLight = texture(samplerCube(inCubeGlobalIrradiance, linearClampEdgeSampler), normal).rgb  * frameData.globalIBLIntensity;// ;
        color += diffuseLight * diffuseColor;
    }

    float ao = texture(sampler2D(inGTAO, linearClampEdgeSampler), uv).r * inGbufferSValue.b;
    vec3 multiBounceAO = AoMultiBounce(ao, baseColor);

    color *= multiBounceAO; // Also add ao to direct light, though it's no correct in physics.

    // Emissive color.
    color += emissiveColor;

    // Store in scene color.
    imageStore(hdrSceneColor, workPos, vec4(color, 1.0f));
}