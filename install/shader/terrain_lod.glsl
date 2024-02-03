#version 460

#include "common_shader.glsl"

layout (set = 0, binding = 0)  uniform UniformFrameData { PerFrameData frameData; };

// Ready lod node for terrain, .xy is position of 2d, .z is lod level.
layout (set = 0, binding = 1) buffer SSBOLODReadyNodeList { uvec3 data[]; } ssboReadyLodNodeList;
layout (set = 0, binding = 2) buffer SSBOLODReadyNodeListCount { uint counter; } ssboReadyLodNodeListCounter;

// Counter for current lod level which need continue sperate. (size = frameData.landscape.lodCount - 2)
layout (set = 0, binding = 3) buffer SSBOLODRenderCount { uint counter[]; } ssboLODContinueCounters;

// Store temporal buffer of current lod.
layout (set = 0, binding = 4) buffer SSBOLODContinueBufferRead { uvec2 data[]; } ssboLODContinueBufferRead;
layout (set = 0, binding = 5) buffer SSBOLODContinueBufferWrite { uvec2 data[]; } ssboLODContinueBufferWrite;


struct DispatchIndirectCommand 
{
    uint x;
    uint y;
    uint z;
    uint pad;
};

layout (set = 0, binding = 6) buffer lodDispatchCmdSSBO
{ 
    DispatchIndirectCommand args; 
} lodDispatchCmd;

layout (set = 0, binding = 7) buffer patchDispatchCmdSSBO
{ 
    DispatchIndirectCommand args; 
} patchDispatchCmd;

layout (set = 0, binding = 8) uniform texture2D imageHeightFieldMipmap;

layout (set = 0, binding = 9) buffer lodNodeContinueSSBO
{
    uint data[];
} lodNodeContinue;

layout(push_constant) uniform PushConsts
{   
    uint lodIndex; // from (frameData.landscape.lodCount - 1) to 1.
    float coefficientLodContinue;
    uint maxLodMipmap;
} terrainPush;

#ifdef LOD_PATH_PASS
    layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
    void main()
    {
        uint nodeCount = ssboReadyLodNodeListCounter.counter;

        patchDispatchCmd.args.x = nodeCount;
        patchDispatchCmd.args.y = 1;
        patchDispatchCmd.args.z = 1;
    }
#endif 

#ifdef LOD_ARGS_PASS
    layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
    void main()
    {
        uint level = frameData.landscape.lodCount - terrainPush.lodIndex - 1;
        uint nodeCount = ssboLODContinueCounters.counter[level - 1];

        lodDispatchCmd.args.x = (nodeCount + 63) / 64;
        lodDispatchCmd.args.y = 1;
        lodDispatchCmd.args.z = 1;
    }
#endif


#ifdef LOD_PREPARE_PASS

bool shouldContinueCurrentLOD(uvec2 lodNodePos, uint level)
{
    float tileDim = getTerrainLODSizeFromLOD(terrainPush.lodIndex);
    vec2 pos2D = vec2(lodNodePos + 0.5) * tileDim + vec2(frameData.landscape.offsetX, frameData.landscape.offsetY);

    vec3 worldPos = vec3(pos2D.x, 0.0f, pos2D.y);

    vec2 localSampleUv = vec2(worldPos.x - frameData.landscape.offsetX, worldPos.z - frameData.landscape.offsetY) / frameData.landscape.terrainDimension;
    localSampleUv = saturate(localSampleUv);

    // 
    int workLevel = max(0, int(terrainPush.maxLodMipmap) - 2 - int(level));
    ivec2 levelSize = textureSize(imageHeightFieldMipmap, workLevel);
    ivec2 localPos = ivec2(levelSize * localSampleUv + 0.5);

    vec2 minMax = texelFetch(imageHeightFieldMipmap, localPos, workLevel).xy;


    // Evaluate 3 mode to get best lod quality.
    float cH = mix(frameData.landscape.minHeight, frameData.landscape.maxHeight, (minMax.x + minMax.y) * 0.5);
    float lH = mix(frameData.landscape.minHeight, frameData.landscape.maxHeight, minMax.x);
    float tH = mix(frameData.landscape.minHeight, frameData.landscape.maxHeight, minMax.y);

    worldPos.y = cH;
    float dist = distance(frameData.camWorldPos.xyz, worldPos);

    worldPos.y = lH;
    dist = min(dist, distance(frameData.camWorldPos.xyz, worldPos));

    worldPos.y = tH;
    dist = min(dist, distance(frameData.camWorldPos.xyz, worldPos));


    float factor = dist / (tileDim * terrainPush.coefficientLodContinue);
    return factor < 1.0f;
}

layout (local_size_x = 64) in;
void main()
{
    uint idx = gl_GlobalInvocationID.x;
    uint level = frameData.landscape.lodCount - terrainPush.lodIndex - 1;

    const bool bLoopPass = (level > 0);

    // Default from 0, which need evaluate all node.
    uint nodeCount = kTerrainCoarseNodeDim * kTerrainCoarseNodeDim;
    if (bLoopPass)
    {
        // Evaluate from counter.
        nodeCount = ssboLODContinueCounters.counter[level - 1];
    }

    // Pre-return.
    if (idx >= nodeCount)
    {
        return;
    }

    uvec2 lodNodePos = uvec2(idx % kTerrainCoarseNodeDim, idx / kTerrainCoarseNodeDim);
    if (bLoopPass)
    {
        lodNodePos = ssboLODContinueBufferRead.data[idx];
    }

    

    int lodNodeId = int((16 * (1 - pow(4, int(level)))) / -3) + int(lodNodePos.y * 4 * exp2(level) + lodNodePos.x);

    if(shouldContinueCurrentLOD(lodNodePos, level))
    {
        if (terrainPush.lodIndex == 1)
        {
            // lod #1, so just add to final
            uint indexId = atomicAdd(ssboReadyLodNodeListCounter.counter, 4);

            ssboReadyLodNodeList.data[indexId + 0] = uvec3(lodNodePos * 2 + uvec2(0, 0), 0);
            ssboReadyLodNodeList.data[indexId + 1] = uvec3(lodNodePos * 2 + uvec2(0, 1), 0);
            ssboReadyLodNodeList.data[indexId + 2] = uvec3(lodNodePos * 2 + uvec2(1, 0), 0);
            ssboReadyLodNodeList.data[indexId + 3] = uvec3(lodNodePos * 2 + uvec2(1, 1), 0);
        }
        else
        {
            uint indexId = atomicAdd(ssboLODContinueCounters.counter[level], 4);

            ssboLODContinueBufferWrite.data[indexId + 0] = lodNodePos * 2 + uvec2(0, 0);
            ssboLODContinueBufferWrite.data[indexId + 1] = lodNodePos * 2 + uvec2(0, 1);
            ssboLODContinueBufferWrite.data[indexId + 2] = lodNodePos * 2 + uvec2(1, 0);
            ssboLODContinueBufferWrite.data[indexId + 3] = lodNodePos * 2 + uvec2(1, 1);
        }

        lodNodeContinue.data[lodNodeId] = 1;
    }
    else
    {
        // Don't need continue, just use current lod.
        uint indexId = atomicAdd(ssboReadyLodNodeListCounter.counter, 1);
        ssboReadyLodNodeList.data[indexId] = uvec3(lodNodePos, terrainPush.lodIndex);


        lodNodeContinue.data[lodNodeId] = 0;
    }


}
#endif