#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_shader_subgroup_basic : enable

#include "KinoBokehDof_Common.glsl"

// We always want to focus miku chan, so evaluate focus depth from scene shading model id.

const uint kDimBlock = 3; // x3 is load-balance.
layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    const uvec2 groupThreadId = remap8x8(gl_LocalInvocationIndex);
    const uvec2 dispatchId = groupThreadId + gl_WorkGroupID.xy * 8;
    const uvec2 workPosBasic = dispatchId * kDimBlock;

    float minDepthInBlock =  99999999999999.0f;
    float maxDepthInBlock = -1.0f;

    float sumDepthInBlock = 0.0f;
    uint pmxPixelCountInBlock = 0;

    const uvec2 shadingModelIdSize = uvec2(textureSize(inGbufferA, 0) - 1);
    for(uint i = 0; i < kDimBlock; i ++)
    {
        for(uint j = 0; j < kDimBlock; j ++)
        {
            // OpenGL core 4.3
            // Unlike filtered texel accesses, texel fetches do not support LOD clamping or any texture wrap mode.
            ivec2 workPosLoop = ivec2(clamp(workPosBasic + uvec2(i, j), uvec2(0), shadingModelIdSize));

            const bool bPMXMesh = isPMXMeshShadingModelCharacter(texelFetch(inGbufferA, workPosLoop, 0).a);
            const float depthSample = linearizeDepth(texelFetch(inDepth, workPosLoop, 0).x, viewData);

            if(bPMXMesh)
            {
                minDepthInBlock = min(minDepthInBlock, depthSample);
                maxDepthInBlock = max(maxDepthInBlock, depthSample);

                sumDepthInBlock += depthSample;
                pmxPixelCountInBlock ++;
            }
        }
    }

    uint subGroupMinDepth = subgroupMin(floatBitsToUint(minDepthInBlock));
    uint subGroupMaxDepth = subgroupMax(floatBitsToUint(maxDepthInBlock));    

    uint subGroupSumPMXDepth = subgroupAdd(floatBitsToUint(sumDepthInBlock));    
    uint subGroupSumPMXPixelNum = subgroupAdd(pmxPixelCountInBlock);    

    if(subgroupElect())
    {
        atomicMin(depthRange.minDepth, uint(subGroupMinDepth));
        atomicMax(depthRange.maxDepth, uint(subGroupMaxDepth));

        atomicAdd(depthRange.sumPmxDepth, uint(subGroupSumPMXDepth));
        atomicAdd(depthRange.pmxPixelCount, uint(subGroupSumPMXPixelNum));
    }
}

