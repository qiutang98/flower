#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_KHR_shader_subgroup_quad : enable

#include "sssr_common.glsl"

bool IsBaseRay(uvec2 dispatchThreadId, uint samplesPerQuad) 
{
    switch (samplesPerQuad) 
    {
    case 1: // Deactivates 3 out of 4 rays
        return ((dispatchThreadId.x & 1) | (dispatchThreadId.y & 1)) == 0; 
    case 2: // Deactivates 2 out of 4 rays. Keeps diagonal.
        return (dispatchThreadId.x & 1) == (dispatchThreadId.y & 1); 
    default: // Other case 4. all is base ray.
        return true;
    }
}

shared uint sharedTileCount;

void classifyTiles(uvec2 dispatchThreadId, uvec2 groupThreadId, float roughness)
{
    // Shared tile count clear.
    sharedTileCount = 0;
    const uvec2 workSize = textureSize(inGbufferS, 0);

    const uint samplesPerQuad = SSRPush.samplesPerQuad;

    const bool bAllInScreen = (dispatchThreadId.x < workSize.x) && (dispatchThreadId.y < workSize.y);
    const bool bCanReflective = (texelFetch(inDepth, ivec2(dispatchThreadId), 0).r != 0) && isGlossyReflection(roughness);
    const bool bMirrorPlane = isMirrorReflection(roughness);
    const bool bBaseRay = IsBaseRay(dispatchThreadId, samplesPerQuad);

    // Only need ray in these case.
    bool bNeedRay = bAllInScreen && bCanReflective;

    // Only run denoise for no mirror plane.
    bool bNeedDenoise = bNeedRay && (!bMirrorPlane); 

    // If is mirror plane, always full ray.
    // If not mirror plane, add deactivate check.
    bNeedRay = bNeedRay && (!bNeedDenoise || bBaseRay);

    // Create ray if temporal variance guided enable.
    const bool bTemporalVarianceGuidedEnable = SSRPush.temporalVarianceGuidedTracingEnabled > 0;
    if(bTemporalVarianceGuidedEnable && bNeedDenoise && (!bNeedRay))
    {
        const bool bTemporalVarianceRequire = texelFetch(inSSRVarianceHistory, ivec2(dispatchThreadId), 0).r > SSRPush.temporalVarianceThreshold;
        bNeedRay = bNeedRay || bTemporalVarianceRequire;
    }

    // Sync for shared tile count clear done.
    groupMemoryBarrier();
    barrier();

    // Add tile count flag when plane is reflective.
    if(bCanReflective)
    {
        atomicAdd(sharedTileCount, 1);
    }

    // Our pixel only requires a copy if we want to run a denoiser on it but don't want to shoot a ray for it.
    const bool bRequireCopy = (!bNeedRay) && bNeedDenoise;

    // Issue: https://github.com/google/shaderc/issues/1277
    // Read quad data 
    // Exist one bad fact here after compile:
    //
    //  uint* _485 = &samplesPerQuadd;
    //  uint _486 = *_485;
    //  bool _488 = _486 != 4;
    //  bool _489 = *bBaseRay;
    //  bool _490 = _488 && _489;
    //  
    //  if(_490) 
    //  {
    //      bool _498 = *bRequireCopy;
    //      bool _499 = GroupNonUniformQuadSwap(Subgroup, _498, 0);
    //  }
    //  bool _494 = Phi(_490, _493);
    //  *bCopyHorizontal@3 = _494;

    // bool bCopyHorizontal = ((samplesPerQuad != 4) && bBaseRay) && subgroupQuadSwapHorizontal(bRequireCopy);
    // bool bCopyVertical   = ((samplesPerQuad == 1) && bBaseRay) &&   subgroupQuadSwapVertical(bRequireCopy);
    // bool bCopyDiagonal   = ((samplesPerQuad == 1) && bBaseRay) &&   subgroupQuadSwapDiagonal(bRequireCopy);

    bool bCopyH = subgroupQuadSwapHorizontal(bRequireCopy);
    bool bCopyV = subgroupQuadSwapVertical(bRequireCopy);
    bool bCopyD = subgroupQuadSwapDiagonal(bRequireCopy);

    bool bCopyHorizontal = bCopyH && ((samplesPerQuad != 4) && bBaseRay);
    bool bCopyVertical   = bCopyV && ((samplesPerQuad == 1) && bBaseRay);
    bool bCopyDiagonal   = bCopyD && ((samplesPerQuad== 1) && bBaseRay);

    // Compact ray and append to ray list.
    uvec4 ballotNeedRay = subgroupBallot(bNeedRay);
    uint waveRayCount = subgroupBallotBitCount(ballotNeedRay); 
    uint localRayIndexInWave = subgroupBallotExclusiveBitCount(ballotNeedRay);

    uint baseRayIndex;
    if(subgroupElect())
    {
        baseRayIndex = addRayCount(waveRayCount);
    }
    baseRayIndex = subgroupBroadcastFirst(baseRayIndex);

    if(bNeedRay)
    {
        uint rayIndex = baseRayIndex + localRayIndexInWave;
        addRay(rayIndex, dispatchThreadId, bCopyHorizontal, bCopyVertical, bCopyDiagonal);
    }

    vec4 intersectionOutput = vec4(0);
    imageStore(SSRIntersection, ivec2(dispatchThreadId), intersectionOutput);

    // Sync for sharedTileCount add done.
    groupMemoryBarrier();
    barrier();

    // Add denoise tile info.
    if ((groupThreadId.x == 0) && (groupThreadId.y == 0) && (sharedTileCount > 0)) 
    {
        uint tileOffset = addDenoiseTileCount();
        addDenoiserTile(tileOffset, dispatchThreadId);
    }
}

layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    uvec2 groupThreadId = remap8x8(gl_LocalInvocationIndex);
    uvec2 dispatchId = groupThreadId + gl_WorkGroupID.xy * 8;
    ivec2 workPos = ivec2(dispatchId);

    float perceptualRoughness = texelFetch(inGbufferS, workPos, 0).g;
    float roughness = perceptualRoughness * perceptualRoughness;
    classifyTiles(dispatchId, groupThreadId, roughness);

    // Also store roughness.
    imageStore(SSRExtractRoughness, workPos, vec4(roughness, 0.0f, 0.0f, 0.0f));
}