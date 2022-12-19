#version 460

/*
** Physical based render code, develop by engineer: qiutanguu.
*/

#extension GL_GOOGLE_include_directive : enable

#include "AMD_SSRCommon.glsl"

// AMD SSSR looks good but still pool on performances, current implement it here.

// Important bits from the PBR shader
vec3 getIBLContribution(float perceptualRoughness, vec3 specularColor, vec3 specularLight, vec3 n, vec3 v)
{
    vec3 reflection = normalize(reflect(-v, n));
    float NdotV = clamp(dot(n, v), 0.0, 1.0);

    // Compute roughness's lod.
    uvec2 prefilterCubeSize = textureSize(inCubeGlobalPrefilter, 0);
    float mipCount = float(log2(max(prefilterCubeSize.x, prefilterCubeSize.y)));
    float lod = clamp(perceptualRoughness * float(mipCount), 0.0, float(mipCount));
    
    // NOTE:
    // alphaRoughness = perceptualRoughness * perceptualRoughness
    // Use perceptualRoughness to lut and prefilter search to get better view.

    // Load precompute brdf texture value.
    vec2 brdfSamplePoint = clamp(vec2(NdotV, perceptualRoughness), vec2(0.0, 0.0), vec2(1.0, 1.0));

    // retrieve a scale and bias to F0. See [1], Figure 3
    vec2 brdf = texture(sampler2D(inBRDFLut, linearClampEdgeSampler), brdfSamplePoint).rg;

    vec3 specularLightEnv = textureLod(samplerCube(inCubeGlobalPrefilter, linearClampEdgeSampler), reflection, lod).rgb;
    
    // Also apply ibl intensity.
    specularLightEnv = specularLightEnv * frameData.globalIBLIntensity;

    // Add env ibl specular light, also scale ssr radiance.
    specularLight = max(specularLight, vec3(0.0)) * 2.0 + specularLightEnv;

    vec3 specular = specularLight * (specularColor * brdf.x + brdf.y); 

    return specular;
}

// Only for reference.

layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    uvec2 groupThreadId = remap8x8(gl_LocalInvocationIndex);
    uvec2 dispatchId = groupThreadId + gl_WorkGroupID.xy * 8;
    ivec2 workPos = ivec2(dispatchId);

    ivec2 workSize = imageSize(HDRSceneColorImage);
    if(workPos.x >= workSize.x || workPos.y >= workSize.y)
    {
        return;
    }

    const vec2 uv = (vec2(workPos) + vec2(0.5f)) / vec2(workSize);

    const vec4 inGbufferSValue = texelFetch(inGbufferS, workPos, 0);
    const vec4 inGbufferAValue = texelFetch(inGbufferA, workPos, 0);

    float metallic = inGbufferSValue.r;
    const vec3 f0 = vec3(0.04);
    const vec3 baseColor = inGbufferAValue.rgb;
    vec3 specularColor = mix(f0, baseColor.rgb, metallic);

    float perceptualRoughness = texelFetch(inGbufferS, workPos, 0).g;
    vec3 n = normalize(texelFetch(inGbufferB, workPos, 0).xyz); 
    float deviceZ = texelFetch(inDepth, workPos, 0).r;
    vec3 worldPos = getWorldPos(uv, deviceZ, viewData);
    vec3 v = normalize(viewData.camWorldPos.xyz - worldPos);

    float ao = texture(sampler2D(inGTAO, linearClampEdgeSampler), uv).r * inGbufferSValue.b;
    
    vec4 ssrResult = texelFetch(inSSRIntersection, workPos, 0);

    
    ssrResult.xyz =  getIBLContribution(perceptualRoughness, specularColor, ssrResult.xyz, n, v) * ao;


    vec4 resultColor;
    vec4 srcHdrSceneColor = imageLoad(HDRSceneColorImage, workPos);

    resultColor = max(ssrResult, vec4(0.0)) + srcHdrSceneColor.xyzw;

    resultColor.w = 1.0;

    imageStore(HDRSceneColorImage, workPos, resultColor);
}