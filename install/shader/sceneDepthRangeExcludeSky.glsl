#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_shader_subgroup_basic : enable

#include "common_shader.glsl"

layout(set = 0, binding = 0) uniform texture2D inDepth; // Depth z.
layout(set = 0, binding = 1) buffer SSBODepthRangeBuffer { uint depthMinMax[]; }; // Depth range min max buffer

// x3 is load-balance.
const uint kDimBlock = 3; 

layout(local_size_x = 8, local_size_y = 8) in;
void main()
{
    const uvec2 depthRangeSize = uvec2(textureSize(inDepth, 0) - 1);
    const uvec2 groupThreadId = remap8x8(gl_LocalInvocationIndex);
    const uvec2 dispatchId = groupThreadId + gl_WorkGroupID.xy * 8;
    const uvec2 workPosBasic = dispatchId * kDimBlock;

    // Depth is 32-bit float, gather no performance better to 4 fetch.
    uint minDepthInBlock = ~0;
    uint maxDepthInBlock = 0;
    for(uint i = 0; i < kDimBlock; i ++)
    {
        for(uint j = 0; j < kDimBlock; j ++)
        {
            // OpenGL core 4.3
            // Unlike filtered texel accesses, texel fetches do not support LOD clamping or any texture wrap mode.
            ivec2 workPosLoop = ivec2(clamp(workPosBasic + uvec2(i, j), uvec2(0), depthRangeSize));
            const float deviceZ = texelFetch(inDepth, workPosLoop, 0).x;
            const bool bValidDraw = (deviceZ > 0.0f);
            const uint packDepth = depthPackUnit(deviceZ);

            uint packDepthForMin = bValidDraw ? packDepth : ~0;
            uint packDepthForMax = bValidDraw ? packDepth :  0;

            minDepthInBlock = min(minDepthInBlock, packDepthForMin);
            maxDepthInBlock = max(maxDepthInBlock, packDepthForMax);
        }
    }

    uint subGroupMinDepth = subgroupMin(minDepthInBlock);
    uint subGroupMaxDepth = subgroupMax(maxDepthInBlock);

    if(subgroupElect())
    {
        atomicMin(depthMinMax[0], uint(subGroupMinDepth));
        atomicMax(depthMinMax[1], uint(subGroupMaxDepth));
    }
}