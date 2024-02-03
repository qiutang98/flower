#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable

#define SHARED_SAMPLER_SET 1
#include "common_shader.glsl"

layout (set = 0, binding = 0, rgba16f) uniform image2D outHDRSceneColor;
layout (set = 0, binding = 1) uniform texture2D inGBufferId;
layout (set = 0, binding = 2) uniform UniformFrameData { PerFrameData frameData; };
layout (set = 0, binding = 3) uniform texture2D inAdaptedLumTex;
layout (set = 0, binding = 4) readonly buffer SSBOPerObject { PerObjectInfo objectDatas[]; };

float selectWeight(vec2 uv)
{
    float v = texture(sampler2D(inGBufferId, pointClampEdgeSampler), uv).r;

    uint objectId = unpackFrom16bitObjectId(v);
    bool bSelected = false;
    if(objectId <= kMaxObjectId)
    {
        PerObjectInfo object = objectDatas[objectId];
        bSelected = (object.bSelected != 0);
    }
    else if(objectId == kSkyObjectId)
    {
        bSelected = (frameData.bSkyComponentSelected != 0);
    }

    return bSelected ? 1.0f : 0.0f;
}

layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    uvec2 groupThreadId = remap8x8(gl_LocalInvocationIndex);
    uvec2 dispatchId = groupThreadId + gl_WorkGroupID.xy * 8;
    ivec2 workPos = ivec2(dispatchId);
    ivec2 workSize = imageSize(outHDRSceneColor);

    // border check.
    if(workPos.x >= workSize.x || workPos.y >= workSize.y)
    {
        return;
    }


    vec2 uvDt = 1.0f / vec2(frameData.renderWidth, frameData.renderHeight);
    vec4 resultColor = imageLoad(outHDRSceneColor, workPos);


    const float postScaleToRender = frameData.renderWidth / frameData.postWidth;
    const float distanceDiff = 4.0 * postScaleToRender;

    float TL = selectWeight(uvDt * (workPos + 0.5 + distanceDiff * ivec2(-1,  1)));
    float TM = selectWeight(uvDt * (workPos + 0.5 + distanceDiff * ivec2( 0,  1)));
    float TR = selectWeight(uvDt * (workPos + 0.5 + distanceDiff * ivec2( 1,  1)));
    float ML = selectWeight(uvDt * (workPos + 0.5 + distanceDiff * ivec2(-1,  0)));
    float MR = selectWeight(uvDt * (workPos + 0.5 + distanceDiff * ivec2( 1,  0)));
    float BL = selectWeight(uvDt * (workPos + 0.5 + distanceDiff * ivec2(-1, -1)));
    float BM = selectWeight(uvDt * (workPos + 0.5 + distanceDiff * ivec2( 0, -1)));
    float BR = selectWeight(uvDt * (workPos + 0.5 + distanceDiff * ivec2( 1, -1)));
                         
    float gradX = -TL + TR - 2.0 * ML + 2.0 * MR - BL + BR;
    float gradY =  TL + 2.0 * TM + TR - BL - 2.0 * BM - BR;
    float diff = gradX * gradX + gradY * gradY;

    // Center weight.
    bool bSelected = selectWeight(uvDt * (workPos + 0.5)) > 0.5f;

    // Get mix factor.
    float mixDensity = saturate(smoothstep(0.0, 16.0, diff)) * (bSelected ? 1.0 : 0.3);
    if(mixDensity > 0.0)
    {
        vec3 mixColor = vec3(0.8, 0.15, 0.0);
        const float exposure = max(1e-5f, getExposure(frameData, inAdaptedLumTex).r);

        // Simple inverse reinhard, can't fully inverse film or gt tonemmaper.
        mixColor = mixColor / (exposure * max(vec3(1.0) - mixColor / exposure, 1e-3f));

        // Mix with source color.
        resultColor.xyz = mix(resultColor.xyz, mixColor, mixDensity);
        imageStore(outHDRSceneColor, workPos, resultColor);
    }
}

