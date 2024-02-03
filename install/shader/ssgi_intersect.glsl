#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_EXT_samplerless_texture_functions : enable

#define SHARED_SAMPLER_SET    1
#define BLUE_NOISE_BUFFER_SET 2

#include "common_shader.glsl"
#include "common_lighting.glsl"

layout (set = 0, binding = 0) uniform texture2D inHiz;
layout (set = 0, binding = 1) uniform texture2D inDepth;
layout (set = 0, binding = 2) uniform texture2D inGbufferB;
layout (set = 0, binding = 3) uniform texture2D inGbufferV;
layout (set = 0, binding = 4) uniform texture2D inHDRSceneColor;
layout (set = 0, binding = 5) uniform UniformFrameData { PerFrameData frameData; };
layout (set = 0, binding = 6, rgba16f) uniform image2D imageSSGIResult;
layout (set = 0, binding = 7)  uniform textureCube inProbe0;
layout (set = 0, binding = 8)  uniform textureCube inProbe1;
layout (set = 0, binding = 9) uniform textureCube inSkyIrradiance; // SSR fallback env ibl.

layout(push_constant) uniform PushConsts
{   
    vec3  probe0Pos;
    float probe0ValidFactor;
    vec3  probe1Pos;
    float probe1ValidFactor;

    vec4  boxExtentData0;
    vec4  boxExtentData1;
    vec4  boxExtentData2;
} SSGIPush;

const uint kMaxTraversalIterations   = 64; 
const uint kMinTraversalOccupancy    = 4;
const uint kMostDetailedMip          = 0; // Half resolution
#define kDepthBufferThickness        1.0

vec2 getHizMipResolution(int mipLevel) 
{
    // https://community.khronos.org/t/cost-of-texturesize/65968
    return vec2(textureSize(inHiz, mipLevel));
}

float loadDepth(ivec2 coord, int mip)
{
    return texelFetch(inHiz, coord, mip).r; // use cloest depth.
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
    // Reject the hit if we didnt advance the ray significantly to avoid immediate self reflection
    vec2 manhattanDist = abs(hit.xy - uv);
    vec2 manhattanDistEdge = 2.0f / screenSize;
    if((manhattanDist.x < manhattanDistEdge.x) && (manhattanDist.y < manhattanDistEdge.y)) 
    {
        return 0;
    }

    // Don't lookup radiance from the background.
    float surfaceZ = textureLod(sampler2D(inHiz, pointClampEdgeSampler), hit.xy, kMostDetailedMip).r;
    if(surfaceZ <= 0.0)
    {
        return 0;
    }

    // We check if we hit the surface from the back, these should be rejected.
    vec3 hitNormal = unpackWorldNormal(texture(sampler2D(inGbufferB, pointClampEdgeSampler), hit.xy).rgb);
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

vec3 getReflectionDir(const vec3 viewDir, const vec3 viewNormal, float u, float v)
{
    mat3 tbnTransform = createTBN(viewNormal);
    vec3 viewDirTbn = tbnTransform * (-viewDir);

    const float lobeRoughness = 0.3f;
    vec3 sampledNormalTbn = importanceSampleGGXVNDF(viewDirTbn, lobeRoughness, lobeRoughness, u, v);
    return transpose(tbnTransform) * sampledNormalTbn;
}

layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    ivec2 workSize = imageSize(imageSSGIResult);

    uvec2 groupThreadId = remap8x8(gl_LocalInvocationIndex);
    uvec2 dispatchId = groupThreadId + gl_WorkGroupID.xy * 8;
    ivec2 workPos = ivec2(dispatchId);

    if(workPos.x >= workSize.x || workPos.y >= workSize.y)
    {
        return;
    }

    const vec2 texelSize = 1.0f / vec2(workSize);
    const vec2 uv = (vec2(workPos) + vec2(0.5f)) * texelSize;

    // Skip sky background pixels.
    const float depth = textureLod(sampler2D(inHiz, pointClampEdgeSampler), uv, kMostDetailedMip).r;
    if(depth <= 0.0)
    {
        imageStore(imageSSGIResult, ivec2(workPos), vec4(0.0));
        return;
    }


    // UVz start, we step in screen space in uv unit.
    const vec3 screenSpaceUVzStart = vec3(uv, depth);

    const vec4 inGbufferBValue = texture(sampler2D(inGbufferB, pointClampEdgeSampler), uv);
    const vec3 worldNormal = unpackWorldNormal(inGbufferBValue.rgb);

    const vec3 viewPos = getViewPos(uv, depth, frameData);
    const vec3 viewDir = -normalize(viewPos);
    const vec3 viewNormal = normalize((frameData.camView * vec4(worldNormal, 0.0)).rgb);

    uvec2 offset = uvec2(vec2(0.754877669, 0.569840296) * (frameData.frameIndex.x) * uvec2(workSize));
    uvec2 offsetId = uvec2(workPos) + offset;
    offsetId.x = offsetId.x % workSize.x;
    offsetId.y = offsetId.y % workSize.y;

    float u0 = samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d(offsetId.x, offsetId.y, 0, 0u);
    float u1 = samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d(offsetId.x, offsetId.y, 0, 1u);

    const vec3 viewReflectedDir = // getReflectionDir(viewDir, viewNormal, u0, u1);
        importanceSampleCosine(vec2(u0, 1.0 - u0), viewNormal);

    const vec3 viewEnd = viewPos + viewReflectedDir;
    const vec3 screenSpaceUVzEnd = projectPos(viewEnd, frameData.camProj);

    const vec3 screenSpaceUVz = screenSpaceUVzEnd - screenSpaceUVzStart;

    bool bValidHit = false;
    vec3 hit;
    hit = hizMarching(
        screenSpaceUVzStart, 
        screenSpaceUVz, 
        vec2(workSize),
        int(kMostDetailedMip),
        kMinTraversalOccupancy,
        kMaxTraversalIterations,
        bValidHit
    );

    vec3 worldOrigin = getWorldPos(uv, depth, frameData);
    vec3 worldHit = getWorldPos(hit.xy, hit.z, frameData);
    vec3 worldRay = worldHit - worldOrigin;

    float confidence = bValidHit ? validateHit(hit, uv, worldRay, vec2(workSize), kDepthBufferThickness) : 0;

    vec3 hitResult = vec3(0.0);

    if(confidence > 0.0 && onRange(hit.xy, vec2(0.0), vec2(1.0)))
    {
        vec2 historyUv = hit.xy + texture(sampler2D(inGbufferV, pointClampEdgeSampler), hit.xy).rg;
        if(onRange(historyUv, vec2(0.0), vec2(1.0)))
        {
            hitResult = texture(sampler2D(inHDRSceneColor, pointClampEdgeSampler), historyUv).xyz;
        }
        else
        {
            confidence = 0.0f;
        }
    }
    else
    {
        confidence = 0.0f;
    }

    vec3 envFallback = texture(samplerCube(inSkyIrradiance, linearClampEdgeSampler), worldNormal).rgb;// ;
    hitResult = mix(envFallback, hitResult, confidence);

    // Don't need ray length, all ray is valid hit.
    imageStore(imageSSGIResult, ivec2(workPos), vec4(hitResult, 1.0f));
}