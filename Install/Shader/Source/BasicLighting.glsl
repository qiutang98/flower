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

// Activision GTAO paper: https://www.activision.com/cdn/research/s2016_pbs_activision_occlusion.pptx
vec3 AoMultiBounce(float AO, vec3 baseColor)
{
    vec3 a =  2.0404 * baseColor - 0.3324;
    vec3 b = -4.7951 * baseColor + 0.6417;
    vec3 c =  2.7552 * baseColor + 0.6903;

    vec3 x  = vec3(AO);

    return max(x, ((x * a + b) * x + c) * x);
}

// IBL term compute.
// Reference from https://google.github.io/filament/Filament.md.html 
// 5.3.4.6 IBL evaluation implementation
vec3 getIBLContribution(PBRMaterial materialInfo, vec3 n, vec3 v, vec3 ao, float so)
{
    vec3 diffuse = vec3(0.0);
    vec3 specular = vec3(0.0);

    // Irradiance diffuse compute.
    {
        vec3 diffuseLight = texture(samplerCube(inCubeGlobalIrradiance, linearClampEdgeSampler), n).rgb;

        // No physical correct, but look better.
        // diffuseLight = granTurismoTonemapper(diffuseLight);

        diffuse = diffuseLight * materialInfo.diffuseColor;// * kPI;
    }

    // Environment specular compute.
    {
        vec3 reflection = normalize(reflect(-v, n));
        float NdotV = clamp(dot(n, v), 0.0, 1.0);

        // NOTE:
        // alphaRoughness = perceptualRoughness * perceptualRoughness
        // Use perceptualRoughness to lut and prefilter search to get better view.

        // Load precompute brdf texture value.
        vec2 brdfSamplePoint = clamp(vec2(NdotV, materialInfo.perceptualRoughness), vec2(0.0), vec2(1.0));
        vec2 brdf = texture(sampler2D(inBRDFLut, linearClampEdgeSampler), brdfSamplePoint).rg;

        // Compute roughness's lod.
        uvec2 prefilterCubeSize = textureSize(inCubeGlobalPrefilter, 0);
        float mipCount = float(log2(max(prefilterCubeSize.x, prefilterCubeSize.y)));
        float lod = clamp(materialInfo.perceptualRoughness * float(mipCount), 0.0, float(mipCount));
    
        // Load environment's color from prefilter color.
        vec3 specularLight = textureLod(samplerCube(inCubeGlobalPrefilter, linearClampEdgeSampler), reflection, lod).rgb;

        // No physical correct, but look better.
        // specularLight = granTurismoTonemapper(specularLight);

        // Final get the specular.
        specular = specularLight * (materialInfo.specularColor * brdf.x + brdf.y);
    }

    vec3 singleScatter = diffuse * ao; // Two times

    // TODO: Try some white furnace test. multi-scatter feed.
    // See https://google.github.io/filament/Filament.md.html 4.7.2
    // https://bruop.github.io/ibl/
    // But in my test, looks same with single scatter result.
    return singleScatter;
}

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

    if(frameData.globalIBLEnable > 0)
    {
        float ao = texture(sampler2D(inGTAO, linearClampEdgeSampler), uv).r * meshAo;
        vec3 aoMultiBounceColor = AoMultiBounce(ao, baseColor);

        // GTSO use bent normal, but looks just slightly diff with this, XD.
        // So kill GTSO and save filter cost, keep this so as an approximate.
        // float so = specularAOLagarde(max(0.0, dot(normal, view)), ao, material.alphaRoughness);
        float so = ao;

        color += getIBLContribution(material, normal, view, aoMultiBounceColor, so) * frameData.globalIBLIntensity;
    }
    

    // Emissive color.
    color += emissiveColor;

    // Store in scene color.
    imageStore(hdrSceneColor, workPos, vec4(color, 1.0f));
}