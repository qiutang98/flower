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
layout (set = 0, binding = 6) uniform textureCube inSkyIrradiance;
layout (set = 0, binding = 7)  uniform texture2D inSSAO;
layout (set = 0, binding = 8)  uniform texture2D inSSGI;

layout(push_constant) uniform PushConsts
{   
    uint bSSGIValid;
};

#include "common_lighting.glsl"

#ifdef SKY_LIGHT_ONLY_PASS

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

    const vec4 inGbufferBValue = texelFetch(inGbufferB, workPos, 0);
    const vec4 inGbufferSValue = texelFetch(inGbufferS, workPos, 0);

    // Start basic lighting parameter prepare.
    vec3 sceneColor = inSceneColorValue.rgb;
    const vec3 f0 = vec3(0.04);
    const vec3 baseColor = inGbufferAValue.rgb;

    float metallic    = inGbufferSValue.r;
    float meshAo      = inGbufferSValue.b;
    vec3 diffuseColor = baseColor * (vec3(1.0) - f0) * (1.0 - metallic);

    vec3 normal = unpackWorldNormal(inGbufferBValue.rgb);

    // Get AO
    float aoValid = min(meshAo, texture(sampler2D(inSSAO, pointClampEdgeSampler), uv).r);
    vec3 multiBounceAO = AoMultiBounce(aoValid, baseColor);

    bool bSSGIValidFinal = (bSSGIValid != 0);
    if (bSSGIValidFinal)
    {
        // Sky neighbor area detect.

        float DeviceZ0 = texelFetch(inDepth, workPos + ivec2(-1, 0), 0).r;
        float DeviceZ1 = texelFetch(inDepth, workPos + ivec2( 1, 0), 0).r;
        float DeviceZ2 = texelFetch(inDepth, workPos + ivec2( 0, 1), 0).r;
        float DeviceZ3 = texelFetch(inDepth, workPos + ivec2( 0,-1), 0).r;

        if(DeviceZ0 <= 0.0 || DeviceZ1 <= 0.0 || DeviceZ2 <= 0.0 || DeviceZ3 <= 0.0)
        {
            bSSGIValidFinal = false;
        }
    }

    vec3 skylight;
    if(bSSGIValidFinal)
    {
        skylight = texture(sampler2D(inSSGI, linearClampEdgeSampler), uv).xyz;
    }
    else
    {
        skylight = texture(samplerCube(inSkyIrradiance, linearClampEdgeSampler), normal).rgb;// ;
    }

    sceneColor += skylight * diffuseColor * multiBounceAO;

    imageStore(hdrSceneColor, workPos, vec4(sceneColor, inSceneColorValue.a));
}

#endif