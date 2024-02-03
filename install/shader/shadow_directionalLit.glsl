#version 460

#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable

#define SHARED_SAMPLER_SET    1
#define BLUE_NOISE_BUFFER_SET 2
#include "common_shader.glsl"

// Scene depth z.
layout(set = 0, binding = 0) uniform texture2D inDepth;

// Cascade infos.
layout(set = 0, binding = 1) buffer SSBOCascadeInfoBuffer { CascadeInfo cascadeInfos[]; }; 

layout(set = 0, binding = 2) uniform texture2D inGbufferB;
layout(set = 0, binding = 3) uniform texture2D inCloudShadowDepth;

layout(set = 0, binding = 4) uniform texture2D inTerrainShadowDepth;

// Shadow mask runtime evaluate every frame.
layout(set = 0, binding = 5, rg8) uniform image2D imageShadowMask;

// Per cascade draw info.
layout(set = 0, binding = 6) uniform UniformFrameData { PerFrameData frameData;                   };
layout(set = 0, binding = 7) buffer  SSBOPerObject    { PerObjectInfo objectDatas[];              };

layout(set = 0, binding = 8) buffer SSBODepthRangeBuffer { uint depthMinMax[]; }; // Depth range min max buffer

// Drawcall.
layout(set = 0, binding = 9) buffer SSBOIndirectDraws { StaticMeshDrawCommand indirectCommands[]; };
layout(set = 0, binding =10) buffer SSBODrawCount     { uint drawCount;                           };



// Bindless texture array.
layout(set = 3, binding = 0) buffer BindlessSSBOVertices{ float data[]; } verticesArray[];
layout(set = 4, binding = 0) buffer BindlessSSBOIndices{ uint data[]; } indicesArray[];
layout(set = 5, binding = 0) uniform sampler bindlessSampler[];
layout(set = 6, binding = 0) uniform  texture2D texture2DBindlessArray[];

layout (push_constant) uniform PushConsts 
{  
    uint sdsmShadowDepthIndices[kMaxCascadeNum];

    // For culling.
    uint cullCountPercascade;
    uint cascadeCount;
    uint inCascadeId;
    uint bSDSM;

    vec3 lightDirection;
    float maxDrawDepthDistance;

    uint percascadeDimXY;
    float cascadeSplitLambda;
    float filterSize;
    float cascadeMixBorder;

    float contactShadowLength;
    uint contactShadowSampleNum;
    uint bContactShadow;
    uint bCloudShadow;
};

const int kShadowFilterSampleCount = 8;

// RH look at function for compute shadow camera eye matrix.
mat4 lookAtRH(vec3 eye,vec3 center,vec3 up)
{
    const vec3 f = normalize(center - eye);
    const vec3 s = normalize(cross(f, up));
    const vec3 u = cross(s, f);

    mat4 ret =  
    {
        {1.0f,0.0f,0.0f,0.0f},
        {0.0f,1.0f,0.0f,0.0f},
        {0.0f,0.0f,1.0f,0.0f},
        {1.0f,0.0f,0.0f,1.0f}
    };

    ret[0][0] = s.x; ret[0][1] = u.x; ret[0][2] =-f.x; ret[3][0] =-dot(s, eye);
    ret[1][0] = s.y; ret[1][1] = u.y; ret[1][2] =-f.y; ret[3][1] =-dot(u, eye);
    ret[2][0] = s.z; ret[2][1] = u.z; ret[2][2] =-f.z; ret[3][2] = dot(f, eye);

    return ret;
}

// RH ortho projection function for light matrix.
mat4 orthoRHZeroOne(float left, float right, float bottom, float top, float zNear, float zFar)
{
    mat4 ret =  
    {
        {1.0f,0.0f,0.0f,0.0f},
        {0.0f,1.0f,0.0f,0.0f},
        {0.0f,0.0f,1.0f,0.0f},
        {1.0f,0.0f,0.0f,1.0f}
    };

    ret[0][0] =   2.0f / (right - left);
    ret[1][1] =   2.0f / (top - bottom);
    ret[2][2] =  -1.0f / (zFar - zNear);
    ret[3][0] = -(right + left) / (right - left);
    ret[3][1] = -(top + bottom) / (top - bottom);
    ret[3][2] = -zNear / (zFar - zNear);

	return ret;
}

#ifdef CASCADE_PREPARE_PASS

#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_shader_subgroup_basic : enable

float logCascadeSplit(
    in const float nearZ, 
    in const float farDepthPlane, 
    in const float nearDepthPlane, 
    in const float clipRange, 
    in const uint  cascadeId)
{
    float range = farDepthPlane - nearDepthPlane;
    float ratio = farDepthPlane / nearDepthPlane;

    // get current part factor.
    float p = float(cascadeId + 1) / float(cascadeCount);

    // get log scale factor and uniform scale factor.
    float logScale = nearDepthPlane * pow(abs(ratio), p);
    float uniformScale = nearDepthPlane + range * p;

    // final get split distance.
    float d = cascadeSplitLambda * (logScale - uniformScale) + uniformScale;
    return (d - nearZ) / clipRange;
}

// This pass build sdsm cascade info.
layout(local_size_x = 32) in; 
void main()
{
    const uint idx = gl_GlobalInvocationID.x;
    const uint cascadeId  = idx;

    if(cascadeId >= cascadeCount)
    {
        return;
    }

    // camera info get.
    const float nearZ = frameData.camInfo.z;
    const float farZ  = frameData.camInfo.w;

    // Get depth start and end pos which in range [0, 1].
    const float clipRange = farZ - nearZ;
    
    const float minDepth = uintDepthUnpack(depthMinMax[0]);
    const float maxDepth = uintDepthUnpack(depthMinMax[1]);

    // We reverse z, so min dpeth value is far plane, max depth value is near plane.
    float nearPlaneLinear =  linearizeDepth(maxDepth, nearZ, farZ);
    float farPlaneLinear  =  linearizeDepth(minDepth, nearZ, farZ);

    const bool bFitToScene = false;
    const bool bSDSMEnabled = (bSDSM != 0);

    if(bSDSMEnabled)
    {

        farPlaneLinear = min(farPlaneLinear, nearPlaneLinear + maxDrawDepthDistance);
//      farPlaneLinear = max(farPlaneLinear, nearPlaneLinear + 100.0); // At least draw 100 meter to save some performance when cascade near.
    }
    else
    {
        nearPlaneLinear = nearZ;
        farPlaneLinear  = nearZ + maxDrawDepthDistance;
    }

    // Now setup each cascade frustum corners.
    vec3 frustumCornersWS[8];
    frustumCornersWS[0] = vec3(-1.0f,  1.0f, 1.0f);
    frustumCornersWS[1] = vec3( 1.0f,  1.0f, 1.0f);
    frustumCornersWS[2] = vec3( 1.0f, -1.0f, 1.0f);
    frustumCornersWS[3] = vec3(-1.0f, -1.0f, 1.0f);
    frustumCornersWS[4] = vec3(-1.0f,  1.0f, 0.0f);
    frustumCornersWS[5] = vec3( 1.0f,  1.0f, 0.0f);
    frustumCornersWS[6] = vec3( 1.0f, -1.0f, 0.0f);
    frustumCornersWS[7] = vec3(-1.0f, -1.0f, 0.0f);
    for(uint i = 0; i < 8; i ++)
    {
        vec4 invCorner = frameData.camInvertViewProj * vec4(frustumCornersWS[i], 1.0f);
        frustumCornersWS[i] = invCorner.xyz / invCorner.w;
    }

    vec3 upDir = vec3(0.f, 1.f, 0.f);
    const vec3 lightDir = normalize(lightDirection);

    // Prev split.
    float prevSplitDist = (cascadeId == 0) ? 
        (nearPlaneLinear - nearZ) / clipRange:
        logCascadeSplit(nearZ, farPlaneLinear, nearPlaneLinear, clipRange, cascadeId - 1);

    // Current split.
    float splitDist = logCascadeSplit(nearZ, farPlaneLinear, nearPlaneLinear, clipRange, cascadeId);

    // Calculate 4 corner world pos of cascade view frustum.
    for(uint i = 0; i < 4; i ++)
    {
        vec3 cornerRay = frustumCornersWS[i + 4] - frustumCornersWS[i]; // distance ray.

        vec3 nearCornerRay = cornerRay * prevSplitDist;
        vec3 farCornerRay  = cornerRay * splitDist;

        frustumCornersWS[i + 4] = frustumCornersWS[i] + farCornerRay;
        frustumCornersWS[i + 0] = frustumCornersWS[i] + nearCornerRay;
    }

    // Calculate center pos of view frustum.
    vec3 frustumCenter = vec3(0.0f);
    for(uint i = 0; i < 8; i ++)
    {
        frustumCenter += frustumCornersWS[i];
    }
    frustumCenter /= 8.0f;

    // Get view sphere bounds radius.
    float sphereRadius = 0.0f;
    for(uint i = 0; i < 8; ++i)
    {
        float dist = length(frustumCornersWS[i] - frustumCenter);
        sphereRadius = max(sphereRadius, dist);
    }

    // Round 16.
    sphereRadius = ceil(sphereRadius * 16.0f) / 16.0f;
    vec3 maxExtents = vec3(sphereRadius);
    vec3 minExtents = -maxExtents;
    vec3 cascadeExtents = maxExtents - minExtents;

    // create temporary view project matrix for cascade.
    vec3 shadowCameraPos = frustumCenter - normalize(lightDir) * cascadeExtents.z * 0.5f;

    float nearZProj = 1e-2f; 
    float farZProj  = cascadeExtents.z;     

    mat4 shadowView = lookAtRH(shadowCameraPos,frustumCenter,upDir);
    mat4 shadowProj = orthoRHZeroOne(
        minExtents.x, 
        maxExtents.x, 
        minExtents.y, 
        maxExtents.y,
        farZProj, // Also reverse z for shadow depth.
        nearZProj
    );

    // Texel align.
    const float sMapSize = float(percascadeDimXY);
    mat4 shadowViewProjMatrix = shadowProj * shadowView;
    vec4 shadowOrigin = vec4(0.0f, 0.0f, 0.0f, 1.0f);
    shadowOrigin = shadowViewProjMatrix * shadowOrigin;
    shadowOrigin *= (sMapSize / 2.0f);  
    
    // Move to center uv pos
    vec3 roundedOrigin = round(shadowOrigin.xyz);
    vec3 roundOffset = roundedOrigin - shadowOrigin.xyz;
    roundOffset   = roundOffset * (2.0f / sMapSize);
    roundOffset.z = 0.0f;

    // Push back round offset data to project matrix.
    shadowProj[3][0] += roundOffset.x;
    shadowProj[3][1] += roundOffset.y;

    // Final proj view matrix
    mat4 shadowFinalViewProj = shadowProj * shadowView;

    // push to buffer.
    cascadeInfos[cascadeId].viewProj = shadowFinalViewProj;
    mat4 reverseToWorld = inverse(shadowFinalViewProj);

    // Build frustum plane.
    vec3 p[8];
    {
        p[0] = vec3(-1.0f,  1.0f, 1.0f);
        p[1] = vec3( 1.0f,  1.0f, 1.0f);
        p[2] = vec3( 1.0f, -1.0f, 1.0f);
        p[3] = vec3(-1.0f, -1.0f, 1.0f);
        p[4] = vec3(-1.0f,  1.0f, 0.0f);
        p[5] = vec3( 1.0f,  1.0f, 0.0f);
        p[6] = vec3( 1.0f, -1.0f, 0.0f);
        p[7] = vec3(-1.0f, -1.0f, 0.0f);

        for(uint i = 0; i < 8; i++)
        {
            vec4 invCorner = reverseToWorld * vec4(p[i], 1.0f);
            p[i] = invCorner.xyz / invCorner.w;
        }

        // left
        vec3 leftN = normalize(cross((p[4] - p[7]), (p[3] - p[7])));
        cascadeInfos[cascadeId].frustumPlanes[0] = vec4(leftN, -dot(leftN, p[7]));

        // down
        vec3 downN = normalize(cross((p[6] - p[2]), (p[3] - p[2])));
        cascadeInfos[cascadeId].frustumPlanes[1] = vec4(downN, -dot(downN, p[2]));
        
        // right
        vec3 rightN = normalize(cross((p[6] - p[5]), (p[1] - p[5])));
        cascadeInfos[cascadeId].frustumPlanes[2] = vec4(rightN, -dot(rightN, p[5]));
        
        // top
        vec3 topN = normalize(cross((p[5] - p[4]), (p[0] - p[4])));
        cascadeInfos[cascadeId].frustumPlanes[3] = vec4(topN, -dot(topN, p[4]));

        // front
        vec3 frontN = normalize(cross((p[1] - p[0]), (p[3] - p[0])));
        cascadeInfos[cascadeId].frustumPlanes[4] = vec4(frontN, -dot(frontN, p[0]));

        // back
        vec3 backN  = normalize(cross((p[5] - p[6]), (p[7] - p[6])));
        cascadeInfos[cascadeId].frustumPlanes[5] = vec4(backN, -dot(frontN, p[6]));
    }

    groupMemoryBarrier();
    barrier();

    mat4 coarseShadowMapVP = cascadeInfos[cascadeCount - 1].viewProj;
    {
        // Construct cascade shadow map corner position and reproject to world space.
        vec3 worlPosition00 = constructPos(vec2(0.0, 0.0), 1.0, reverseToWorld); // reverse z.
        vec3 worlPosition11 = constructPos(vec2(1.0, 1.0), 0.0, reverseToWorld); // reverse z.

        // Project to coarse shadow map uv space.
        vec4 v00 = coarseShadowMapVP * vec4(worlPosition00, 1.0f);
        v00.xyz /= v00.w;
        v00.xy = v00.xy * 0.5f + 0.5f;
        v00.y = 1.0f - v00.y;

        // Project to coarse shadow map uv space.
        vec4 v11 = coarseShadowMapVP * vec4(worlPosition11, 1.0f);
        v11.xyz /= v11.w;
        v11.xy = v11.xy * 0.5f + 0.5f;
        v11.y = 1.0f - v11.y;

        // Scale filter size on accurate cascade.
        cascadeInfos[cascadeId].cascadeScale = vec4(1.0f / abs(v11.xy - v00.xy), splitDist, prevSplitDist); 
    }
}

#endif // CASCADE_PREPARE_PASS

#ifdef CASCADE_CULL_PASS

layout (local_size_x = 64) in;
void main()
{
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= cullCountPercascade)
    {
        return;
    }

    PerObjectInfo objectData = objectDatas[idx];
    const MeshInfo meshInfo = objectData.meshInfoData;

    if(meshInfo.meshType != EMeshType_StaticMesh)
    {
        return;
    }

    vec3 localPos = meshInfo.sphereBounds.xyz;
	vec4 worldPos = objectData.modelMatrix * vec4(localPos, 1.0f);

    // local to world normal matrix.
	mat3 normalMatrix = transpose(inverse(mat3(objectData.modelMatrix)));
	mat3 world2Local = inverse(normalMatrix);

    // frustum test.
    for (int i = 0; i < 4; i++) // frustum 4, 5 is back and front face, don't test.
    {
        vec3 worldSpaceN = cascadeInfos[inCascadeId].frustumPlanes[i].xyz;
        float castDistance = dot(worldPos.xyz, worldSpaceN);

        // transfer to local matrix and use abs get first dimensions project value,
        // use that for test.
        vec3 localNormal = world2Local * worldSpaceN;
        float absDiff = dot(abs(localNormal), meshInfo.extents.xyz);
        if (castDistance + absDiff + cascadeInfos[inCascadeId].frustumPlanes[i].w < 0.0)
        {
            // no visibile
            return;
        }
    }
    
    // Build draw command if visible.
    uint drawId = atomicAdd(drawCount, 1);
    indirectCommands[drawId].objectId = idx;

    // We fetech vertex by index, so vertex count is index count.
    indirectCommands[drawId].vertexCount = meshInfo.indicesCount;
    indirectCommands[drawId].firstVertex = meshInfo.indexStartPosition;

    // We fetch vertex in vertex shader, so instancing is unused when rendering.
    indirectCommands[drawId].instanceCount = 1;
}

#endif // CASCADE_CULL_PASS

#ifdef CASCADE_DEPTH_PASS

struct VS2PS
{
    vec2  uv0;
};

vec4 texlod(uint texId, uint samplerId, vec2 uv, float lod)
{
    return textureLod(
        sampler2D(texture2DBindlessArray[nonuniformEXT(texId)], 
        bindlessSampler[nonuniformEXT(samplerId)]), uv, lod);
}

vec4 tex(uint texId,uint samplerId,vec2 uv)
{
    return texture(
        sampler2D(texture2DBindlessArray[nonuniformEXT(texId)], 
        bindlessSampler[nonuniformEXT(samplerId)]), uv);
}

#ifdef VERTEX_SHADER ///////////// vertex shader start 

layout(location = 0) out flat uint outObjectId;
layout(location = 1) out VS2PS vsOut;

void main()
{
    // Load object data.
    outObjectId = indirectCommands[gl_DrawID].objectId;
    const PerObjectInfo objectData = objectDatas[outObjectId];

    // We get bindless array id first.
    const uint indicesId  = objectData.meshInfoData.indicesArrayId;
    const uint positionId = objectData.meshInfoData.positionsArrayId;
    const uint uv0Id = objectData.meshInfoData.uv0sArrayId;

    // Vertex count same with index count, so vertex index same with index index.
    const uint indexId = gl_VertexIndex;

    // Then fetech vertex index from indices array.
    const uint vertexId = indicesArray[nonuniformEXT(indicesId)].data[indexId];

    vec3 position;
    vec2 uv0;

    position.x = verticesArray[nonuniformEXT(positionId)].data[vertexId * kPositionStrip + 0];
    position.y = verticesArray[nonuniformEXT(positionId)].data[vertexId * kPositionStrip + 1];
    position.z = verticesArray[nonuniformEXT(positionId)].data[vertexId * kPositionStrip + 2];
    uv0.x = verticesArray[nonuniformEXT(uv0Id)].data[vertexId * kUv0Strip + 0];
    uv0.y = verticesArray[nonuniformEXT(uv0Id)].data[vertexId * kUv0Strip + 1];

    // Uv0 ready.
    vsOut.uv0 = uv0;

    // All ready, start to do vertex space-transform.
    const mat4 modelMatrix = objectData.modelMatrix;

    // Local vertex position.
    const vec4 localPosition = vec4(position, 1.0f);
    const vec4 worldPosition = modelMatrix * localPosition;

    // Convert to clip space.
    gl_Position = cascadeInfos[inCascadeId].viewProj * worldPosition;

    // NOTE: Depth clamp disabled, all depth will output.
}

#endif /////////////////////////// vertex shader end

#ifdef PIXEL_SHADER ////////////// pixel shader start 

layout(location = 0) in flat uint inObjectId;
layout(location = 1) in VS2PS vsIn;

void main()
{
    // Load object data.
    const PerObjectInfo objectData = objectDatas[inObjectId];
    const BSDFMaterialInfo material = objectData.materialInfoData;

    // Load base color and cut off alpha.
    vec4 baseColor = tex(material.baseColorId, material.baseColorSampler, vsIn.uv0);
    baseColor = baseColor * material.baseColorMul + material.baseColorAdd;
    if(baseColor.a < material.cutoff)
    {
        discard;
    }
}

#endif //////////////////////////// pixel shader end

#endif // CASCADE_DEPTH_PASS

#ifdef SHADOW_MASK_EVALUATE_PASS

vec4 texDepth(uint cascadeId, vec2 uv)
{
    return texture(
        sampler2D(texture2DBindlessArray[nonuniformEXT(sdsmShadowDepthIndices[cascadeId])], pointClampEdgeSampler), uv);
}

// Depth Aware Contact harden pcf. See GDC2021: "Shadows of Cold War" for tech detail.
// Use cache occluder dist to fit one curve similar to tonemapper, to get some effect like pcss.
// can reduce tiny acne natively.
float contactHardenPCFKernal(
    const float occluders, 
    const float occluderDistSum, 
    const float compareDepth,
    const uint shadowSampleCount)
{
    // Normalize occluder dist.
    float occluderAvgDist = occluderDistSum / occluders;
    float w = 1.0f / float(shadowSampleCount); 
    
    // 0 -> contact harden.
    // 1 -> soft, full pcf.
    float pcfWeight =  clamp(occluderAvgDist / compareDepth, 0.0, 1.0);
    
    // Normalize occluders.
    float percentageOccluded = clamp(occluders * w, 0.0, 1.0);

    // S curve fit.
    percentageOccluded = 2.0f * percentageOccluded - 1.0f;
    float occludedSign = sign(percentageOccluded);
    percentageOccluded = 1.0f - (occludedSign * percentageOccluded);
    percentageOccluded = mix(percentageOccluded * percentageOccluded * percentageOccluded, percentageOccluded, pcfWeight);
    percentageOccluded = 1.0f - percentageOccluded;

    percentageOccluded *= occludedSign;
    percentageOccluded = 0.5f * percentageOccluded + 0.5f;

    return 1.0f - percentageOccluded;
}

// Auto bias by cacsade and NoL, some magic number here.
float autoBias(float NoL, float biasMul)
{
    const float baseFactor = filterSize + 1.0;
    return baseFactor * 1e-5f + (1.0f - NoL) * biasMul * 1e-4f * baseFactor;
}

// Surface normal based bias, see https://learn.microsoft.com/en-us/windows/win32/dxtecharts/cascaded-shadow-maps for more details.
vec3 biasNormalOffset(vec3 N, float NoL, float texelSize)
{
    return N * clamp(1.0f - NoL, 0.0f, 1.0f) * texelSize * 10.0f;
}

float shadowPcf(vec3 shadowCoord, float safeNoL, uint activeCascadeId, uvec2 offsetId, float kFilterTexelOffset)
{
    float occluders = 0.0f;
    float occluderDistSum = 0.0f;

    // vec2 cascadeScale = cascadeInfos[cascadeCount - 1 - activeCascadeId].cascadeScale.xy;

    const float compareDepth = shadowCoord.z;
    for (uint i = 0; i < kShadowFilterSampleCount; i++)
    {
        float randRadius  = samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d(offsetId.x, offsetId.y, i, 0u) * kFilterTexelOffset;
        float randomAngle = samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d(offsetId.x, offsetId.y, i, 1u) * 2.0f * kPI;

        // Random offset uv.
        vec2 offsetUv = vec2(cos(randomAngle), sin(randomAngle)) * randRadius; // / max(vec2(1.0), 0.01 * cascadeScale);

        vec2 sampleUv = shadowCoord.xy + offsetUv;
        float depthShadow = texDepth(activeCascadeId, sampleUv).r;

        float dist = depthShadow - compareDepth;
        float occluder = step(0.0, dist); // reverse z.

        // Collect occluders.
        occluders += occluder;
        occluderDistSum += dist * occluder;
    }

    return contactHardenPCFKernal(occluders, occluderDistSum, compareDepth, kShadowFilterSampleCount);
}

float sdsmShadow(float linearZ01, vec3 offsetPos, float safeNoL, vec3 worldPos, uvec2 offsetId, float kFilterTexelOffset)
{
    vec3 shadowCoord;

#if 0 // Flick on sdsm frequency.
    int activeCascadeId = -1;
    for(int i = int(cascadeCount) - 1; i >= 0; i --)
    {
        if(linearZ01 < cascadeInfos[i].cascadeScale.z)
        {
            activeCascadeId = i;
        }
    }

    // Out of shadow area return lit.
    if(activeCascadeId < 0)
    {
        return 1.0f;
    }
#else
    // First find active cascade.
    uint activeCascadeId = 0;
    // Loop to find suitable cascade.
    for(uint cascadeId = 0; cascadeId < cascadeCount; cascadeId ++)
    {
        // Perspective divide to get ndc position.
        shadowCoord = projectPos(worldPos + offsetPos, cascadeInfos[cascadeId].viewProj);
        shadowCoord.z += autoBias(safeNoL, cascadeId + 1.0);

        // Check current cascade is valid in range.
        if(onRange(shadowCoord.xyz, vec3(0.0), vec3(1.0)))
        {
            break;
        }
        activeCascadeId ++;
    }
    // Out of shadow area return lit.
    if(activeCascadeId == cascadeCount)
    {
        return 1.0f;
    }
#endif

    // Main cascade evaluate.
    float sdsmShadowResult = shadowPcf(shadowCoord, safeNoL, activeCascadeId, offsetId, kFilterTexelOffset);

    // Try second cascade evaluate and mix.
    float fadeFactor = 
        (cascadeInfos[activeCascadeId].cascadeScale.z - linearZ01) / 
        (cascadeInfos[activeCascadeId].cascadeScale.z - cascadeInfos[activeCascadeId].cascadeScale.w);

    float distToEdge = 1.0 - max(max(shadowCoord.x, shadowCoord.y), shadowCoord.z);
    distToEdge = min(distToEdge, min(min(shadowCoord.x, shadowCoord.y), shadowCoord.z));

    fadeFactor = max(fadeFactor, distToEdge);
    if(fadeFactor < cascadeMixBorder && activeCascadeId < int(cascadeCount) - 1)
    {
        activeCascadeId ++;
        shadowCoord = projectPos(worldPos + offsetPos, cascadeInfos[activeCascadeId].viewProj);
        shadowCoord.z += autoBias(safeNoL, activeCascadeId + 1.0);

        sdsmShadowResult = mix(
            shadowPcf(shadowCoord, safeNoL, activeCascadeId, offsetId, kFilterTexelOffset), 
            sdsmShadowResult, 
            smoothstep(0.0, cascadeMixBorder, fadeFactor));
    }

    return sdsmShadowResult;
}

layout(local_size_x = 8, local_size_y = 8) in;
void main()
{
    ivec2 depthSize = textureSize(inDepth, 0);

    uvec2 groupThreadId = remap8x8(gl_LocalInvocationIndex);
    uvec2 dispatchId = groupThreadId + gl_WorkGroupID.xy * 8;
    ivec2 workPos = ivec2(dispatchId);

    if(workPos.x >= depthSize.x || workPos.y >= depthSize.y)
    {
        // Skip out of bounds area.
        return;
    }

    const float deviceZ = texelFetch(inDepth, workPos, 0).r;
    if(deviceZ <= 0.0f)
    {
        // Skip sky area.
        return;
    }

    const vec2 uv = (vec2(workPos) + vec2(0.5f)) / vec2(depthSize);
    vec3 worldPos = getWorldPos(uv, deviceZ, frameData);
    float linearZ01 = linearizeDepth01(deviceZ, frameData); 

    // Offset retarget for new seeds each frame
    uvec2 offset = uvec2(vec2(0.754877669, 0.569840296) * (frameData.frameIndex.x) * uvec2(depthSize));
    uvec2 offsetId = uvec2(workPos) + offset;
    offsetId.x = offsetId.x % depthSize.x;
    offsetId.y = offsetId.y % depthSize.y;

    // Final shadow result.
    float shadowResult = 1.0f;
    float cloudShadow = 1.0f;

    const vec4 inGbufferBValue = texelFetch(inGbufferB, workPos, 0);
    vec3   normal = unpackWorldNormal(inGbufferBValue.rgb);
    float safeNoL = clamp(dot(normal, -lightDirection), 0.0, 1.0);

    if(bCloudShadow != 0)
    {
        AtmosphereParameters atmosphere = getAtmosphereParameters(frameData);
        vec3 skyPos = convertToAtmosphereUnit(worldPos, frameData) + vec3(0.0, atmosphere.bottomRadius, 0.0);

        // Now convert cloud coordinate.
        vec3 cloudUvz = projectPos(skyPos, atmosphere.cloudShadowViewProj);
        vec2 texSize = textureSize(inCloudShadowDepth, 0).xy;

        // Offset one pixel avoid always sample one pixel which cause banding.
        cloudUvz.x += (samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d(offsetId.x, offsetId.y, 0, 0u) * 2.0f - 1.0f) / texSize.x;
        cloudUvz.y += (samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d(offsetId.x, offsetId.y, 0, 1u) * 2.0f - 1.0f) / texSize.y;

        // ESM shadowmap.
        if(onRange(cloudUvz.xy, vec2(0), vec2(1)))
        {
            float cloudExpZ = texture(sampler2D(inCloudShadowDepth, linearClampEdgeSampler), cloudUvz.xy).x;
            float cloudComputeExpZ = cloudExpZ * exp(kCloudShadowExp * cloudUvz.z);

            cloudShadow = min(cloudShadow, saturate(cloudComputeExpZ));
        }
    }

#if 0
    // Terrain raymarching shadow.
    if (frameData.landscape.bLandscapeValid != 0 && 
       (-lightDirection.y >  0.0) && // No sun set.
       (worldPos.y < frameData.landscape.maxHeight)) // No over max height. 
    {
        vec3 rayDir    = normalize(-lightDirection);

        // AABB box intersection.

        // Get valid ray start position.
        vec3 rayStart = worldPos;
        if (rayStart.y < frameData.landscape.minHeight)
        {
            float offset = (frameData.landscape.minHeight - rayStart.y) / rayDir.y;
            rayStart += offset * rayDir;
        }

        vec3 rayEnd = rayStart;
        {
            float offset = (frameData.landscape.maxHeight - rayEnd.y) / rayDir.y;
            rayEnd += offset * rayDir;
        }

        const float altitude = rayEnd.y - rayStart.y;

        vec3 startUvz;
        vec3 endUvz;



        startUvz.xy = vec2(rayStart.x - frameData.landscape.offsetX, rayStart.z - frameData.landscape.offsetY) / vec2(frameData.landscape.terrainDimension);
        endUvz.xy   = vec2(rayEnd.x   - frameData.landscape.offsetX, rayEnd.z   - frameData.landscape.offsetY) / vec2(frameData.landscape.terrainDimension);

        startUvz.z = (rayStart.y - frameData.landscape.minHeight) / frameData.landscape.maxHeight;
        endUvz.z   = (rayEnd.y   - frameData.landscape.minHeight) / frameData.landscape.maxHeight;

        const float kNumSample = 8.0;
        const float kMixFactor = 5.0;
        const float kBias      = -0.0001;
        
        vec3 dt = (endUvz - startUvz) / (kNumSample + 1.0);
        float jitter = interleavedGradientNoise(vec2(workPos), frameData.frameIndex.x % frameData.jitterPeriod);


        float terrainShadow = 1.0;
        for(float i = 0; i < kNumSample; i ++)
        {
            vec3 traceUvz = startUvz + dt * (i + jitter);

            // .x is max height.
            float heightmapValue = textureLod(
                sampler2D(texture2DBindlessArray[nonuniformEXT(frameData.landscape.hzbUUID)], pointClampEdgeSampler), traceUvz.xy, 1).y;
            
            terrainShadow *= pow(saturate(traceUvz.z / heightmapValue), 8.0);
        }

        shadowResult = min(shadowResult, terrainShadow);
    }
#endif

    bool bShouldRayTraceShadow;

    const float cascadeMapTexelSize = 1.0f / float(percascadeDimXY);
    const float kFilterTexelOffset = filterSize * cascadeMapTexelSize;


    const vec3 offsetPos = biasNormalOffset(normal, safeNoL, cascadeMapTexelSize); // Offset position align normal direction.

    if(shadowResult > 0.0)
    {
        shadowResult = min(shadowResult, sdsmShadow(linearZ01, offsetPos, safeNoL, worldPos, offsetId, kFilterTexelOffset));
    }


    
    float lightAngleNormal = dot(normal, -lightDirection);
    bShouldRayTraceShadow = 
        shadowResult     >  0.0 && // Skip shadow area.
        bContactShadow   != 0   && // Skip when all close.
        lightAngleNormal >  0.0 && // Avoid self shadow.
       -lightDirection.y >  0.0;   // Skip sun set.

    // Scale by light normal angle avoid self shadow.
    float contactShadowLenWS = (150 + saturate(1.0 - lightAngleNormal * 4.0) * 800.0) * linearZ01 * contactShadowLength;
    contactShadowLenWS = clamp(contactShadowLenWS, 0.1, 500.0);

    // Contact shadow.
    if(bShouldRayTraceShadow)
    {
        float rayTraceShadow = 1.0f - screenSpaceContactShadow(
            inDepth,
            pointClampEdgeSampler,
            frameData,
            interleavedGradientNoise(vec2(workPos), frameData.frameIndex.x % frameData.jitterPeriod)
            , contactShadowSampleNum
            , worldPos
            , -lightDirection
            , contactShadowLenWS
        );

        shadowResult = min(rayTraceShadow, shadowResult);
    }




    imageStore(imageShadowMask, workPos, vec4(shadowResult, cloudShadow, 1.0f, 1.0f));
}

#endif // SHADOW_MASK_EVALUATE_PASS