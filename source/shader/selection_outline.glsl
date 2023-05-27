#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable

#include "common/shared_functions.glsl"

layout (set = 0, binding = 0, rgba8) uniform image2D inoutLdrColor;
layout (set = 0, binding = 1) uniform texture2D inMask;
layout (set = 0, binding = 2) uniform utexture2D inSceneNodeIdTexture;
layout (set = 0, binding = 3) uniform UniformFrameData { PerFrameData frameData; };
#define SHARED_SAMPLER_SET 1
#include "common/shared_sampler.glsl"

layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    uvec2 groupThreadId = remap8x8(gl_LocalInvocationIndex);
    uvec2 dispatchId = groupThreadId + gl_WorkGroupID.xy * 8;
    ivec2 workPos = ivec2(dispatchId);
    ivec2 workSize = imageSize(inoutLdrColor);

    // border check.
    if(workPos.x >= workSize.x || workPos.y >= workSize.y)
    {
        return;
    }


    vec2 uvDt = 1.0f / vec2(frameData.displayWidth, frameData.displayHeight);
    const vec2 cancelJitter = -frameData.jitterData.xy * uvDt;

    vec4 resultColor = imageLoad(inoutLdrColor, workPos);

    const int distanceDiff = 3;

    float TL = texture(sampler2D(inMask, linearClampEdgeSampler), uvDt * (workPos + 0.5 + distanceDiff * ivec2(-1,  1)) - cancelJitter).r;
    float TM = texture(sampler2D(inMask, linearClampEdgeSampler), uvDt * (workPos + 0.5 + distanceDiff * ivec2( 0,  1)) - cancelJitter).r;
    float TR = texture(sampler2D(inMask, linearClampEdgeSampler), uvDt * (workPos + 0.5 + distanceDiff * ivec2( 1,  1)) - cancelJitter).r;
    float ML = texture(sampler2D(inMask, linearClampEdgeSampler), uvDt * (workPos + 0.5 + distanceDiff * ivec2(-1,  0)) - cancelJitter).r;
    float MR = texture(sampler2D(inMask, linearClampEdgeSampler), uvDt * (workPos + 0.5 + distanceDiff * ivec2( 1,  0)) - cancelJitter).r;
    float BL = texture(sampler2D(inMask, linearClampEdgeSampler), uvDt * (workPos + 0.5 + distanceDiff * ivec2(-1, -1)) - cancelJitter).r;
    float BM = texture(sampler2D(inMask, linearClampEdgeSampler), uvDt * (workPos + 0.5 + distanceDiff * ivec2( 0, -1)) - cancelJitter).r;
    float BR = texture(sampler2D(inMask, linearClampEdgeSampler), uvDt * (workPos + 0.5 + distanceDiff * ivec2( 1, -1)) - cancelJitter).r;
                         
    float gradX = -TL + TR - 2.0 * ML + 2.0 * MR - BL + BR;
    float gradY =  TL + 2.0 * TM + TR - BL - 2.0 * BM - BR;
    float diff = gradX * gradX + gradY * gradY;

    uint bSelected = unpackToSceneNodeSelected(texture(usampler2D(inSceneNodeIdTexture, pointClampEdgeSampler), uvDt * (workPos + 0.5) - cancelJitter).r);

    float mixDensity = saturate(smoothstep(0.0, 16.0, diff)) * (bSelected != 0 ? 1.0 : 0.3);
    vec3 mixColor = bSelected != 0 ? vec3(1.0, 0.5, 0.0) : vec3(0.6, 0.2, 0.0);

    resultColor.xyz = mix(resultColor.xyz, mixColor, mixDensity);

    imageStore(inoutLdrColor, workPos, resultColor);
}

