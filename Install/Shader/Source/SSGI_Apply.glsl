#version 460

#extension GL_GOOGLE_include_directive : enable

#include "SSGI_Common.glsl"


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

    const vec3 baseColor = inGbufferAValue.rgb;
    float metallic = inGbufferSValue.r;
    const vec3 f0 = vec3(0.04);
    vec3 diffuseColor = baseColor * (vec3(1.0) - f0) * (1.0 - metallic);

    vec3 n = normalize(texelFetch(inGbufferB, workPos, 0).xyz); 

    float ao = texture(sampler2D(inGTAO, linearClampEdgeSampler), uv).r * inGbufferSValue.b;
    vec3 multiBounceAO = AoMultiBounce(ao, baseColor);

    vec3 diffuseLight = texture(samplerCube(inCubeGlobalPrefilter, linearClampEdgeSampler), n).rgb  * frameData.globalIBLIntensity;// ;

    diffuseLight += texture(sampler2D(inSSRIntersection, linearClampEdgeSampler), uv).rgb;

    diffuseLight = diffuseLight * diffuseColor;

    vec4 srcHdrSceneColor = imageLoad(HDRSceneColorImage, workPos);

    srcHdrSceneColor = max(vec4(diffuseLight, 0.0), vec4(0.0)) + srcHdrSceneColor.xyzw;
  
    imageStore(HDRSceneColorImage, workPos, srcHdrSceneColor);
}