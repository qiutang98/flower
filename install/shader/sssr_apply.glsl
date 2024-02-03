#version 460
#extension GL_GOOGLE_include_directive : enable

#include "sssr_common.glsl"

// Important bits from the PBR shader
vec3 computeIBLContribution(float perceptualRoughness, vec3 specularColor, vec3 specularLight,  vec3 n, vec3 v)
{
    float NdotV = clamp(dot(n, v), 0.0, 1.0);

    // NOTE:
    // alphaRoughness = perceptualRoughness * perceptualRoughness
    // Use perceptualRoughness to lut and prefilter search to get better view.

    // Load precompute brdf texture value.
    vec2 brdfSamplePoint = clamp(vec2(NdotV, perceptualRoughness), vec2(0.0, 0.0), vec2(1.0, 1.0));

    // retrieve a scale and bias to F0. See [1], Figure 3
    vec2 brdf = texture(sampler2D(inBRDFLut, linearClampEdgeSampler), brdfSamplePoint).rg;

    // Add env ibl specular light, also scale ssr radiance.
    specularLight = max(specularLight, vec3(0.0));

    vec3 specular = specularLight * (specularColor * brdf.x + brdf.y); 

    return specular;
}

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
    float deviceZ = texelFetch(inDepth, workPos, 0).r;

    if(deviceZ <= 0.0f)
    {
        return;
    }
    const vec4 inGbufferSValue = texelFetch(inGbufferS, workPos, 0);
    const vec4 inGbufferAValue = texelFetch(inGbufferA, workPos, 0);

    float metallic = inGbufferSValue.r;
    const vec3 f0 = vec3(0.04);
    const vec3 baseColor = inGbufferAValue.rgb;
    vec3 specularColor = mix(f0, baseColor.rgb, metallic);

    float perceptualRoughness = texelFetch(inGbufferS, workPos, 0).g;
    vec3 n = unpackWorldNormal(texelFetch(inGbufferB, workPos, 0).xyz); 

    vec3 worldPos = getWorldPos(uv, deviceZ, frameData);
    vec3 v = normalize(frameData.camWorldPos.xyz - worldPos);

    float ao = min(texture(sampler2D(inSSAO, linearClampEdgeSampler), uv).r, inGbufferSValue.b);
    vec3 multiBounceAO = AoMultiBounce(ao, baseColor);



    vec3 ssrResult = texelFetch(inSSRIntersection, workPos, 0).xyz;
    if(ssrResult.x + ssrResult.y + ssrResult.z <= 0.0f)
    {
        vec3 reflection = normalize(reflect(-v, n));
        ssrResult += getIBLContribution(perceptualRoughness, reflection, v, worldPos);
    }


    ssrResult =  computeIBLContribution(perceptualRoughness, specularColor, ssrResult, n, v) * multiBounceAO;

    vec4 resultColor;
    vec4 srcHdrSceneColor = imageLoad(HDRSceneColorImage, workPos);

    resultColor.xyz = max(ssrResult, vec3(0.0)) + srcHdrSceneColor.xyz;
    resultColor.w = 1.0;

    imageStore(HDRSceneColorImage, workPos, resultColor);
}