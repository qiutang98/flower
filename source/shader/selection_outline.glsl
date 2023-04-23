#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable

#include "common/shared_functions.glsl"

layout (set = 0, binding = 0, rgba8) uniform image2D inoutLdrColor;
layout (set = 0, binding = 1) uniform texture2D inMask;
layout (set = 0, binding = 2) uniform utexture2D inSceneNodeIdTexture;

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

    vec4 resultColor = imageLoad(inoutLdrColor, workPos);

    const int distanceDiff = 3;

    float TL = texelFetch(inMask, workPos + distanceDiff * ivec2(-1,  1), 0).r;
    float TM = texelFetch(inMask, workPos + distanceDiff * ivec2( 0,  1), 0).r;
    float TR = texelFetch(inMask, workPos + distanceDiff * ivec2( 1,  1), 0).r;
    float ML = texelFetch(inMask, workPos + distanceDiff * ivec2(-1,  0), 0).r;
    float MR = texelFetch(inMask, workPos + distanceDiff * ivec2( 1,  0), 0).r;
    float BL = texelFetch(inMask, workPos + distanceDiff * ivec2(-1, -1), 0).r;
    float BM = texelFetch(inMask, workPos + distanceDiff * ivec2( 0, -1), 0).r;
    float BR = texelFetch(inMask, workPos + distanceDiff * ivec2( 1, -1), 0).r;
                         
    float gradX = -TL + TR - 2.0 * ML + 2.0 * MR - BL + BR;
    float gradY =  TL + 2.0 * TM + TR - BL - 2.0 * BM - BR;
    float diff = gradX * gradX + gradY * gradY;

    uint bSelected = unpackToSceneNodeSelected(texelFetch(inSceneNodeIdTexture, workPos, 0).r);

    float mixDensity = saturate(smoothstep(0.0, 16.0, diff)) * (bSelected != 0 ? 1.0 : 0.3);
    vec3 mixColor = bSelected != 0 ? vec3(1.0, 0.5, 0.0) : vec3(0.6, 0.2, 0.0);

    resultColor.xyz = mix(resultColor.xyz, mixColor, mixDensity);

    imageStore(inoutLdrColor, workPos, resultColor);
}

