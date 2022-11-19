#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_ballot : enable

#include "SSGI_Common.glsl"
#include "IBL_Common.glsl"

vec2 getHizMipResolution(int mipLevel) 
{
    // https://community.khronos.org/t/cost-of-texturesize/65968
    return vec2(textureSize(inHiz, mipLevel));
}

float loadDepth(ivec2 coord, int mip)
{
    return texelFetch(inHiz, coord, mip).r; // use cloest depth.
}

vec3 SphericalToCartesian(float phi, float cosTheta)
{
    float sinPhi = sin(phi);
    float cosPhi = cos(phi);

    float sinTheta = sqrt(saturate(1.0 - cosTheta * cosTheta));

    return vec3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
}

vec3 SampleSphereUniform(float u1, float u2)
{
    float phi = 6.28318530718f * u2;
    float cosTheta = 1.0 - 2.0 * u1;

    return SphericalToCartesian(phi, cosTheta);
}

vec3 SampleHemisphereCosine(float u1, float u2, vec3 normal)
{
    vec3 pointOnSphere = SampleSphereUniform(u1, u2);
    return normalize(normal + pointOnSphere);
}

vec3 getSampleDir(const vec3 viewNormal,ivec2 sampleCoord)
{
    vec2 u = texelFetch(spp_1_blueNoise, sampleCoord % 128, 0).rg;

    return SampleHemisphereCosine(u.x, u.y, viewNormal);
}

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
        bExitDueToLowOccupancy = subgroupBallotBitCount(subgroupBallot(true)) <= minTraversalOccupancy;

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
    // Fade out hits near the screen borders
    vec2 fov = 0.05 * vec2(screenSize.y / screenSize.x, 1);
    vec2 border = smoothstep(vec2(0), fov, hit.xy) * (1 - smoothstep(vec2(1 - fov), vec2(1), hit.xy));
    float vignette = border.x * border.y;


    // Reject the hit if we didnt advance the ray significantly to avoid immediate self reflection
    vec2 manhattanDist = abs(hit.xy - uv);
    vec2 manhattanDistEdge = 2.0f / screenSize;

    ivec2 texelCoords = ivec2(screenSize * hit.xy);

    // Don't lookup radiance from the background.
    float surfaceZ = texelFetch(inHiz, texelCoords, 0).r;
    if(!isShadingModelValid(texelFetch(inGbufferA, texelCoords, 0).a) || surfaceZ == 0.0)
    {
        return vignette;
    }

    // We check if we hit the surface from the back, these should be rejected.
    vec3 hitNormal = normalize(texelFetch(inGbufferB, texelCoords, 0).rgb);

    vec3 viewSpaceSurface = getViewPos(hit.xy, surfaceZ, viewData);
    vec3 viewSpaceHit = getViewPos(hit.xy, hit.z, viewData);
    float distance = length(viewSpaceSurface - viewSpaceHit);


    // We accept all hits that are within a reasonable minimum distance below the surface.
    // Add constant in linear space to avoid growing of the reflections toward the reflected objects.
    float confidence = 1 - smoothstep(0, depthBufferThickness, distance);
    confidence *= confidence;

    return vignette * confidence;
}

layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    uvec2 groupThreadId = remap8x8(gl_LocalInvocationIndex);
    uvec2 rayCoord = groupThreadId + gl_WorkGroupID.xy * 8;

    const uvec2 screenSize = imageSize(SSRIntersection);
    const vec2 screenSizeInv = 1.0 / vec2(screenSize);
    const vec2 uv = (rayCoord + 0.5) * screenSizeInv;

    const vec3 worldNormal = normalize(texelFetch(inGbufferB, ivec2(rayCoord), 0).xyz); 
    const vec3 sampelDirWorld = getSampleDir(worldNormal,ivec2(rayCoord));

    const uvec2 hizSize = textureSize(inHiz, 0);

    const int mostDetailedMip = int(kMostDetailedMip); 
    const vec2 mipResolution = getHizMipResolution(mostDetailedMip);

    const float z = loadDepth(ivec2(rayCoord), mostDetailedMip);
    const vec3 screenSpaceUVzStart = vec3(uv, z);

    const vec3 viewPos = getViewPos(uv, z, viewData);
    const vec3 viewDir = normalize(viewPos);
    const vec3 viewNormal = normalize((viewData.camView * vec4(worldNormal, 0.0)).rgb);
    const vec3 viewSampleDir = normalize((viewData.camView * vec4(sampelDirWorld, 0.0)).rgb);


    const vec3 viewEnd = viewPos + viewSampleDir;
    const vec3 screenSpaceUVzEnd = projectPos(viewEnd, viewData.camProj);

    // Now get the screen space step dir.
    const vec3 screenSpaceUVz = screenSpaceUVzEnd - screenSpaceUVzStart;

    bool bValidHit = false;
    vec3 hit = hizMarching(
        screenSpaceUVzStart, 
        screenSpaceUVz, 
        vec2(screenSize),
        mostDetailedMip,
        kMinTraversalOccupancy,
        kMaxTraversalIterations,
        bValidHit
    );

    vec3 worldOrigin = getWorldPos(uv, z, viewData);
    vec3 worldHit = getWorldPos(hit.xy, hit.z, viewData);
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

            vec3 sampleN = normalize(texelFetch(inGbufferB, ivec2(screenSize * historyUv.xy), 0).rgb);

            float NoL = saturate(dot(sampleN, sampelDirWorld));

            reflectionRadiance = reflectionRadiance * (1.0 - NoL);
        }
        else
        {
            confidence = 0.0;
        }
    }

    vec3 worldSpaceReflectedDir = (viewData.camInvertView * vec4(viewSampleDir, 0.0)).xyz;
    reflectionRadiance = mix(vec3(0.0), reflectionRadiance, confidence);

    vec4 newSample = vec4(reflectionRadiance, worldRayLength);
    imageStore(SSRIntersection, ivec2(rayCoord), newSample);
}