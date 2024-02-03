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
layout (set = 0, binding = 7)  uniform textureCube inSkyReflection;

#include "common_lighting.glsl"


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

vec3 getIBLContribution(float perceptualRoughness, vec3 n, vec3 v)
{
    vec3 reflection = normalize(reflect(-v, n));
    float NdotV = clamp(dot(n, v), 0.0, 1.0);

    // Compute roughness's lod.
    uvec2 prefilterCubeSize = textureSize(inSkyReflection, 0);
    float mipCount = float(log2(max(prefilterCubeSize.x, prefilterCubeSize.y)));
    float lod = clamp(perceptualRoughness * float(mipCount), 0.0, float(mipCount));
    
    return textureLod(samplerCube(inSkyReflection, linearClampEdgeSampler), reflection, lod).rgb;
}

#ifdef REFLECTION_COMPOSITE_PASS

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

    const vec4 inSceneColorValue = imageLoad(hdrSceneColor, workPos);
    const vec4 inGbufferSValue = texelFetch(inGbufferS, workPos, 0);
    const vec4 inGbufferAValue = texelFetch(inGbufferA, workPos, 0);
    const vec4 inGbufferBValue = texelFetch(inGbufferB, workPos, 0);
    const float deviceZ = texelFetch(inDepth, workPos, 0).r;

    if(deviceZ <= 0.0)
    {
        return;
    }

    EShadingModelType shadingModel = unpackShadingModelId(inGbufferAValue.a);
    if(shadingModel != EShadingModelType_DefaultLit)
    {
        return;
    }

    vec3 sceneColor = inSceneColorValue.rgb;

    const float meshAo = inGbufferSValue.b;
    float metallic = inGbufferSValue.r;
    const vec3 f0 = vec3(0.04);
    const vec3 baseColor = inGbufferAValue.rgb;
    vec3 specularColor = mix(f0, baseColor.rgb, metallic);


    float perceptualRoughness = texelFetch(inGbufferS, workPos, 0).g;
    vec3 n = texelFetch(inGbufferB, workPos, 0).xyz; 
    vec3 worldPos = getWorldPos(uv, deviceZ, frameData);
    vec3 v = normalize(frameData.camWorldPos.xyz - worldPos);
    vec3 normal = unpackWorldNormal(inGbufferBValue.rgb);

    {
        vec3 envFallback = getIBLContribution(perceptualRoughness, normal, normalize(frameData.camWorldPos.xyz - worldPos));
        sceneColor += computeIBLContribution(perceptualRoughness, specularColor, envFallback, normal, v);
    }

    imageStore(hdrSceneColor, workPos, vec4(sceneColor, inSceneColorValue.a));
}

#endif