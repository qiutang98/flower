#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_ballot : enable

#include "sssr_common.glsl"

vec2 getHizMipResolution(int mipLevel) 
{
    // https://community.khronos.org/t/cost-of-texturesize/65968
    return vec2(textureSize(inHiz, mipLevel));
}

float loadDepth(ivec2 coord, int mip)
{
    return texelFetch(inHiz, coord, mip).r; // use cloest depth.
}

vec3 getReflectionDir(const vec3 viewDir, const vec3 viewNormal, float roughness, ivec2 sampleCoord, uvec2 screenSize)
{
    uvec2 offset = uvec2(vec2(0.754877669, 0.569840296) * (frameData.frameIndex.x % frameData.jitterPeriod) * uvec2(screenSize));
    uvec2 offsetId = uvec2(sampleCoord) + offset;
    offsetId.x = offsetId.x % screenSize.x;
    offsetId.y = offsetId.y % screenSize.y;

    vec2 u;
    u.x = samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d(offsetId.x, offsetId.y, 0, 0u);
    u.y = samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d(offsetId.x, offsetId.y, 0, 1u);

    mat3 tbnTransform = createTBN(viewNormal);
    vec3 viewDirTbn = tbnTransform * (-viewDir);

    vec3 sampledNormalTbn = importanceSampleGGXVNDF(viewDirTbn, roughness, roughness, u.x, u.y);

    vec3 reflectedDirTbn = reflect(-viewDirTbn, sampledNormalTbn);
    return transpose(tbnTransform) * reflectedDirTbn;
}


// NOTE: Hiz ray intersection is accurate, but need more step to get good result.
//       Maybe we just need some fast intersect like linear search with only 16 tap.
//       Eg, unreal engine 4's SSR use this tech, full screen SSR just cost 0.5ms in 2K.
bool advanceRay(
    vec3 origin, 
    vec3 direction, 
    vec3 invDirection, 
    vec2 currentMipPosition, 
    vec2 currentMipResolutionInv, 
    vec2 floorOffset, 
    vec2 uvOffset, 
    float surfaceZ, 
    inout vec3 position, 
    inout float currentT) 
{
    vec2 xyPlane = floor(currentMipPosition) + floorOffset;
    xyPlane = xyPlane * currentMipResolutionInv + uvOffset;
    vec3 boundaryPlanes = vec3(xyPlane, surfaceZ);

    // Intersect ray with the half box that is pointing away from the ray origin.
    // o + d * t = p' => t = (p' - o) / d
    vec3 t = boundaryPlanes * invDirection - origin * invDirection;

    // Prevent using z plane when shooting out of the depth buffer.
    t.z = direction.z < 0.0 ? t.z : 3.402823466e+38; // reverse z.
    // t.z = direction.z > 0.0 ? t.z : 3.402823466e+38; // No reverse z.

    // Choose nearest intersection with a boundary.
    float tMin = min(min(t.x, t.y), t.z);

    bool bAboveSurface = surfaceZ < position.z; // reverse z.
    // bool bAboveSurface = surfaceZ > position.z; // No reverse z.

    // Decide whether we are able to advance the ray until we hit the xy boundaries or if we had to clamp it at the surface.
    // We use the asuint comparison to avoid NaN / Inf logic, also we actually care about bitwise equality here to see if t_min is the t.z we fed into the min3 above.
    bool bSkipTile = floatBitsToUint(tMin) != floatBitsToUint(t.z) && bAboveSurface;

    // Make sure to only advance the ray if we're still above the surface.
    currentT = bAboveSurface ? tMin : currentT;

    // Advance ray.
    position = origin + currentT * direction;

    return bSkipTile;
}

vec3 hizMarching(
    vec3 origin, 
    vec3 dir, 
    bool bMirror, 
    vec2 screenSize,
    int mostDetailedMip,
    uint minTraversalOccupancy,
    uint maxTraversalIntersections,
    out bool bValidHit)
{
    vec3 invDir;
    invDir.x = dir.x != 0.0 ? 1.0 / dir.x : 3.402823466e+38;
    invDir.y = dir.y != 0.0 ? 1.0 / dir.y : 3.402823466e+38;
    invDir.z = dir.z != 0.0 ? 1.0 / dir.z : 3.402823466e+38;

    int currentMip = mostDetailedMip;

    vec2 currentMipRes = getHizMipResolution(currentMip);
    vec2 currentMipResInv = 1.0 / currentMipRes;

    // Slightly offset ensure ray step into pixel cell.
    vec2 uvOffset = 0.005f * exp2(mostDetailedMip) / screenSize * sign(dir.xy);

    // Offset applied depending on current mip resolution to move the boundary to the left/right upper/lower border depending on ray direction.
    vec2 floorOffset;
    floorOffset.x = dir.x < 0.0 ? 0.0 : 1.0;
    floorOffset.y = dir.y < 0.0 ? 0.0 : 1.0;

    float currentT;
    vec3 position;
    // Init advance ray avoid self hit.
    {
        vec2 currentMipPosition = currentMipRes * origin.xy;

        vec2 xyPlane = floor(currentMipPosition) + floorOffset;
        xyPlane = xyPlane * currentMipResInv + uvOffset;

        // o + d * t = p' => t = (p' - o) / d
        vec2 t = xyPlane * invDir.xy - origin.xy * invDir.xy;
        currentT = min(t.x, t.y);
        position = origin + currentT * dir;
    }

    bool bExitDueToLowOccupancy = false;
    int i = 0;
    while(i < maxTraversalIntersections && !bExitDueToLowOccupancy && currentMip >= mostDetailedMip)
    {
        vec2 currentMipPosition = currentMipRes * position.xy;

        float surfaceZ = loadDepth(ivec2(currentMipPosition), currentMip);
        bExitDueToLowOccupancy = !bMirror && subgroupBallotBitCount(subgroupBallot(true)) <= minTraversalOccupancy;

        bool bSkipTile = advanceRay(
            origin, 
            dir, 
            invDir, 
            currentMipPosition, 
            currentMipResInv, 
            floorOffset, 
            uvOffset, 
            surfaceZ, 
            position, 
            currentT);

        currentMip += bSkipTile ? 1 : -1; 
        currentMipRes = getHizMipResolution(currentMip);
        currentMipResInv = 1.0 / currentMipRes;

        ++i;
    }

    bValidHit = (i <= maxTraversalIntersections);

    return position;
}

float validateHit(
    vec3 hit, 
    vec2 uv, 
    vec3 worldSpaceRayDirection, 
    vec2 screenSize, 
    float depthBufferThickness) 
{
    // Reject the hit if we didnt advance the ray significantly to avoid immediate self reflection
    vec2 manhattanDist = abs(hit.xy - uv);
    vec2 manhattanDistEdge = 2.0f / screenSize;
    if((manhattanDist.x < manhattanDistEdge.x) && (manhattanDist.y < manhattanDistEdge.y)) 
    {
        return 0;
    }

    ivec2 texelCoords = ivec2(screenSize * hit.xy);

    // Don't lookup radiance from the background.
    float surfaceZ = texelFetch(inDepth, texelCoords, 0).r;
    if(surfaceZ <= 0.0)
    {
        return 0;
    }

    // We check if we hit the surface from the back, these should be rejected.
    vec3 hitNormal = unpackWorldNormal(texelFetch(inGbufferB, texelCoords, 0).rgb);
    if (dot(hitNormal, worldSpaceRayDirection) > 0) 
    {
        return 0;
    }

    vec3 viewSpaceSurface = getViewPos(hit.xy, surfaceZ, frameData);
    vec3 viewSpaceHit = getViewPos(hit.xy, hit.z, frameData);
    float distance = length(viewSpaceSurface - viewSpaceHit);

    // Fade out hits near the screen borders
    vec2 fov = 0.05 * vec2(screenSize.y / screenSize.x, 1);
    vec2 border = smoothstep(vec2(0), fov, hit.xy) * (1 - smoothstep(vec2(1 - fov), vec2(1), hit.xy));
    float vignette = border.x * border.y;

    // We accept all hits that are within a reasonable minimum distance below the surface.
    // Add constant in linear space to avoid growing of the reflections toward the reflected objects.
    float confidence = 1 - smoothstep(0, depthBufferThickness, distance);
    confidence *= confidence;

    return vignette * confidence;
}

layout (local_size_x = 64) in;
void main()
{
    uint rayIndex = gl_GlobalInvocationID.x;

    if(rayIndex >= ssboRayCounter.rayCount)
    {
        return;
    }

    uint packedCoords = ssboRayList.data[rayIndex];

    uvec2 rayCoord;
    bool bCopyHorizontal;
    bool bCopyVertical;
    bool bCopyDiagonal;

    unpackRayCoords(packedCoords, rayCoord, bCopyHorizontal, bCopyVertical, bCopyDiagonal);
    const uvec2 screenSize = imageSize(SSRIntersection);
    const vec2 screenSizeInv = 1.0 / vec2(screenSize);
    const vec2 uv = (rayCoord + 0.5) * screenSizeInv;

    const vec3 worldNormal = unpackWorldNormal(texelFetch(inGbufferB, ivec2(rayCoord), 0).xyz); 
    const float roughness = texelFetch(inSSRExtractRoughness, ivec2(rayCoord), 0).r;


    const bool bMirrorPlane = isMirrorReflection(roughness);
    const int mostDetailedMip = bMirrorPlane ? 0 : int(SSRPush.mostDetailedMip); 

    const float z = loadDepth(ivec2(rayCoord), mostDetailedMip);
    const vec3 screenSpaceUVzStart = vec3(uv, z);

    const vec3 viewPos = getViewPos(uv, z, frameData);
    const vec3 viewDir = normalize(viewPos);
    const vec3 viewNormal = normalize((frameData.camView * vec4(worldNormal, 0.0)).rgb);
    const vec3 viewReflectedDir = getReflectionDir(viewDir, viewNormal, roughness, ivec2(rayCoord), screenSize);
    const vec3 viewEnd = viewPos + viewReflectedDir;
    const vec3 screenSpaceUVzEnd = projectPos(viewEnd, frameData.camProj);

    // Now get the screen space step dir.
    const vec3 screenSpaceUVz = screenSpaceUVzEnd - screenSpaceUVzStart;

    const bool bGlossy = isGlossyReflection(roughness);

    bool bValidHit = false;
    vec3 hit;
    if(bGlossy && roughness < 0.2f) // Skip out ray hit.
    {
        hit = hizMarching(
            screenSpaceUVzStart, 
            screenSpaceUVz, 
            bMirrorPlane, 
            vec2(screenSize),
            mostDetailedMip,
            kMinTraversalOccupancy,
            kMaxTraversalIterations,
            bValidHit
        );
    }
    else
    {
        // Same with src pos, so ray length will be zero.
        hit = vec3(uv, z);
    }

    vec3 worldOrigin = getWorldPos(uv, z, frameData);
    vec3 worldHit = getWorldPos(hit.xy, hit.z, frameData);
    vec3 worldRay = worldHit - worldOrigin;

    float confidence = bValidHit ? validateHit(hit, uv, worldRay, vec2(screenSize), kDepthBufferThickness) : 0;
    float worldRayLength = max(0, length(worldRay));

    vec3 reflectionRadiance = vec3(0);
    if (confidence > 0) 
    {
        vec2 historyUv = hit.xy + texelFetch(inGbufferV, ivec2(screenSize * hit.xy), 0).rg;

        if(historyUv.x >= 0 && historyUv.y >= 0 && historyUv.x <= 1 && historyUv.y <= 1)
        {
            // Found an intersection with the depth buffer -> We can lookup the color from lit scene.
            reflectionRadiance = texelFetch(inHDRSceneColor, ivec2(screenSize * historyUv.xy), 0).rgb;
            // reflectionRadiance *= kPI; // Scale ssr hit result to keep energy full.
        }
        else
        {
            confidence = 0.0;
        }
    }

    vec3 worldSpaceReflectedDir = (frameData.camInvertView * vec4(viewReflectedDir, 0.0)).xyz;
    vec3 envFallback = getIBLContribution(roughness, worldSpaceReflectedDir, normalize(frameData.camWorldPos.xyz - worldOrigin), worldOrigin);

    reflectionRadiance = mix(envFallback, reflectionRadiance, confidence);

    vec4 newSample = vec4(reflectionRadiance, worldRayLength);

    imageStore(SSRIntersection, ivec2(rayCoord), newSample);

    uvec2 copyTarget = rayCoord ^ 0x1; // Flip last bit to find the mirrored coords along the x and y axis within a quad.
    if (bCopyHorizontal) 
    {
        uvec2 copyCoords = uvec2(copyTarget.x, rayCoord.y);
        imageStore(SSRIntersection, ivec2(copyCoords), newSample);
    }
    if (bCopyVertical) 
    {
        uvec2 copyCoords = uvec2(rayCoord.x, copyTarget.y);
        imageStore(SSRIntersection, ivec2(copyCoords), newSample);
    }
    if (bCopyDiagonal)
    {
        uvec2 copyCoords = copyTarget;
        imageStore(SSRIntersection, ivec2(copyCoords), newSample);
    }
}