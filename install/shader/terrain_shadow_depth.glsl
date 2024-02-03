#version 460

#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable

#define SHARED_SAMPLER_SET 1
#include "common_shader.glsl"

layout(set = 0, binding = 0) uniform UniformFrameData { PerFrameData frameData;                   };
layout(set = 0, binding = 1) buffer SSBOPatchBuffer { TerrainPatch patches[]; } ssboPatchBuffer;
layout(set = 0, binding = 2) uniform texture2D heightmapTexture;
layout(set = 0, binding = 3) buffer SSBOCascadeInfoBuffer { CascadeInfo cascadeInfos[]; }; 

layout(location = 0) in  vec2  vsIn;

layout (push_constant) uniform PushConsts 
{  
    uint inCascadeId;
};

void main()
{
    TerrainPatch patchInfo = ssboPatchBuffer.patches[gl_InstanceIndex];

    vec3 localPos = vec3(patchInfo.position.x, 0.0, patchInfo.position.y);

    uint topDiff   = (patchInfo.patchCrossLOD & (0xff <<  0)) >>  0;
    uint downDiff  = (patchInfo.patchCrossLOD & (0xff <<  8)) >>  8;
    uint leftDiff  = (patchInfo.patchCrossLOD & (0xff << 16)) >> 16;
    uint rightDiff = (patchInfo.patchCrossLOD & (0xff << 24)) >> 24;

    const float kQuadSize = 1.0 / 16.0;
    const float kEpsilonQuad = kQuadSize * 0.5;

    vec2 snapPos = vsIn;

    // Top fix.
    if((vsIn.y < kEpsilonQuad) && (topDiff > 0))
    {
        float modSize = exp2(topDiff) * kQuadSize;
        float lessValue = mod(vsIn.x, modSize);
        if(lessValue > kEpsilonQuad)
        {
            snapPos.x = vsIn.x + (modSize - lessValue);
        }
    }

    // Down fix
    if((vsIn.y > 1.0 - kEpsilonQuad) && (downDiff > 0))
    {
        float modSize = exp2(downDiff) * kQuadSize;
        float lessValue = mod(vsIn.x, modSize);
        if(lessValue > kEpsilonQuad)
        {
            snapPos.x = vsIn.x - lessValue;
        }
    }

    // left fix
    if((vsIn.x < kEpsilonQuad) && (leftDiff > 0))
    {
        float modSize = exp2(leftDiff) * kQuadSize;
        float lessValue = mod(vsIn.y, modSize);
        if(lessValue > kEpsilonQuad)
        {
            snapPos.y = vsIn.y + (modSize - lessValue);
        }
    }

    // right fix
    if((vsIn.x > 1.0 - kEpsilonQuad) && (rightDiff > 0))
    {
        float modSize = exp2(rightDiff) * kQuadSize;
        float lessValue = mod(vsIn.y, modSize);
        if(lessValue > kEpsilonQuad)
        {
            snapPos.y = vsIn.y - lessValue;
        }
    }

    float tileDim = getTerrainLODSizeFromLOD(patchInfo.lod);
    float patchDim = tileDim / 8.0; // Meter.
    localPos += vec3(snapPos.x, 0.0, snapPos.y) * patchDim;

    vec2 localUv = vec2(localPos.x - frameData.landscape.offsetX, localPos.z - frameData.landscape.offsetY) / vec2(frameData.landscape.terrainDimension);
    float heightmapValue = texture(sampler2D(heightmapTexture, linearClampEdgeSampler), localUv).x;

    localPos.y = mix(frameData.landscape.minHeight, frameData.landscape.maxHeight, heightmapValue);

    vec4 worldPos = vec4(localPos, 1.0);

    gl_Position = cascadeInfos[inCascadeId].viewProj * worldPos;
}