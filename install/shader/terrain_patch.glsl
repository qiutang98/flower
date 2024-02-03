#version 460

#include "common_shader.glsl"

layout (set = 0, binding = 0)  uniform UniformFrameData { PerFrameData frameData; };
layout (set = 0, binding = 1) buffer SSBOLODReadyNodeList { uvec3 data[]; } ssboReadyLodNodeList;
layout (set = 0, binding = 2) buffer SSBOLODReadyNodeListCount { uint counter; } ssboReadyLodNodeListCounter;

layout (set = 0, binding = 3) buffer SSBOPatchCount { uint counter; } ssboPatchCounter;

layout (set = 0, binding = 4) buffer SSBOPatchBuffer { TerrainPatch patches[]; } ssboPatchBuffer;

layout (set = 0, binding = 5) buffer SSBOIndirectDraws { uint drawCommand[]; };
layout (set = 0, binding = 6) uniform texture2D lodTexture;

#ifdef DRAW_COMMAND_PASS

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main()
{
    uint patchCount = ssboPatchCounter.counter;

    drawCommand[0] = 16 * 16 * 3 * 2;
    drawCommand[1] = patchCount;
    drawCommand[2] = 0;
    drawCommand[3] = 0;
}

#endif 

#ifdef PATCH_CULL_PASS

layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    uint wavefrontIndex = gl_WorkGroupID.x;
    if (wavefrontIndex >= ssboReadyLodNodeListCounter.counter)
    {
        return;
    }
    uvec3 lodNode = ssboReadyLodNodeList.data[wavefrontIndex];

    ivec2 lodTexSize = textureSize(lodTexture, 0).xy;

    // Generate patch per lane.

    ivec2 patchInnerPos = ivec2(gl_LocalInvocationID.xy);

    // Get patch position.
    float tileDim = getTerrainLODSizeFromLOD(lodNode.z);
    float patchDim = tileDim / 8.0;

    vec2 pos2D = vec2(lodNode.xy) * tileDim;

    // Patch offset.
    pos2D += vec2(patchInnerPos) * patchDim;

    // Global offset in 2d.
    pos2D += vec2(frameData.landscape.offsetX, frameData.landscape.offsetY);

    TerrainPatch patchResult;
    patchResult.position = pos2D;
    patchResult.lod = lodNode.z;
    patchResult.patchCrossLOD = 0;

    const bool bPatchInEdge = (patchInnerPos.x == 0) || (patchInnerPos.y == 0) || (patchInnerPos.x == 7) || (patchInnerPos.y == 7);

    // Check cross lod case.
    if(bPatchInEdge)
    {
        const bool bUpCheck    = (patchInnerPos.y == 0);
        const bool bDownCheck  = (patchInnerPos.y == 7);
        const bool bLeftCheck  = (patchInnerPos.x == 0);
        const bool bRightCheck = (patchInnerPos.x == 7);

        uint level = frameData.landscape.lodCount - lodNode.z - 1;

        vec2 currentUv = vec2(lodNode.xy) / float(4 * exp2(level));

        ivec2 currentPos = ivec2(currentUv * lodTexSize + 0.5);

        int texelSize = int(1.0f / float(4 * exp2(level)) * lodTexSize);
        int currentLod = int(texelFetch(lodTexture, currentPos, 0).x * 255.0f + 0.5f);

        // NOTE: currentLod - sampleLod > 7 case? I hope it will never happen.
        if(bUpCheck)
        {
            ivec2 samplePos = currentPos + ivec2(0, -1) * texelSize;
            int sampleLod = int(texelFetch(lodTexture, samplePos, 0).x * 255.0f + 0.5f);

            if(sampleLod < currentLod)
            {
                patchResult.patchCrossLOD |= uint(currentLod - sampleLod) << 0;
            }
        }

        if(bDownCheck)
        {
            ivec2 samplePos = currentPos + ivec2(0,  1) * texelSize;
            int sampleLod = int(texelFetch(lodTexture, samplePos, 0).x * 255.0f + 0.5f);

            if(sampleLod < currentLod)
            {
                patchResult.patchCrossLOD |= uint(currentLod - sampleLod) << 8;
            }
        }

        if(bLeftCheck)
        {
            ivec2 samplePos = currentPos + ivec2(-1, 0) * texelSize;
            int sampleLod = int(texelFetch(lodTexture, samplePos, 0).x * 255.0f + 0.5f);

            if(sampleLod < currentLod)
            {
                patchResult.patchCrossLOD |= uint(currentLod - sampleLod) << 16;
            }
        }

        if(bRightCheck)
        {
            ivec2 samplePos = currentPos + ivec2(1, 0) * texelSize;
            int sampleLod = int(texelFetch(lodTexture, samplePos, 0).x * 255.0f + 0.5f);

            if(sampleLod < currentLod)
            {
                patchResult.patchCrossLOD |= uint(currentLod - sampleLod) << 24;
            }
        }
    }

    uint indexId = atomicAdd(ssboPatchCounter.counter, 1);
    ssboPatchBuffer.patches[indexId] = patchResult;
}

#endif