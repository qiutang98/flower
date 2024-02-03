#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_ray_query : enable

#define SHARED_SAMPLER_SET    1
#define BLUE_NOISE_BUFFER_SET 2

#include "common_shader.glsl"

layout (set = 0, binding = 0) uniform writeonly image2D rayShadowMask;
layout (set = 0, binding = 1) uniform UniformFrameData { PerFrameData frameData; };
layout (set = 0, binding = 2) uniform accelerationStructureEXT topLevelAS;
layout (set = 0, binding = 3) uniform texture2D inDepth;
layout (set = 0, binding = 4) buffer  SSBOPerObject    { PerObjectInfo objectDatas[];              };

layout(set = 3, binding = 0) buffer BindlessSSBOVertices{ float data[]; } verticesArray[];
layout(set = 4, binding = 0) buffer BindlessSSBOIndices{ uint data[]; } indicesArray[];
layout(set = 5, binding = 0) uniform sampler bindlessSampler[];
layout(set = 6, binding = 0) uniform  texture2D texture2DBindlessArray[];

layout (push_constant) uniform PushConsts 
{  
    vec3 lightDirection;
    float lightRadius;

    float rayMinRange;
    float rayMaxRange;
};

bool hitTest(in rayQueryEXT rayQuery)
{
    // Get hit object info.
    int instanceCustomIndexEXT = rayQueryGetIntersectionInstanceIdEXT(rayQuery, false);
    const PerObjectInfo objectData = objectDatas[instanceCustomIndexEXT];

    // Get material info and mesh info.
    const MeshInfo meshInfo = objectData.meshInfoData;
    const BSDFMaterialInfo material = objectData.materialInfoData;

    // 
    int primitiveID = int(meshInfo.indexStartPosition) + rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, false) * 3;

    const uint indicesId  = meshInfo.indicesArrayId;
    const uint uv0Id      = meshInfo.uv0sArrayId;

    // Hit triangle id.
    const uint vertexId_0 = indicesArray[nonuniformEXT(indicesId)].data[primitiveID + 0];
    const uint vertexId_1 = indicesArray[nonuniformEXT(indicesId)].data[primitiveID + 1];
    const uint vertexId_2 = indicesArray[nonuniformEXT(indicesId)].data[primitiveID + 2];

    vec2 v0, v1, v2;
    v0.x = verticesArray[nonuniformEXT(uv0Id)].data[vertexId_0 * kUv0Strip + 0];
    v0.y = verticesArray[nonuniformEXT(uv0Id)].data[vertexId_0 * kUv0Strip + 1];

    v1.x = verticesArray[nonuniformEXT(uv0Id)].data[vertexId_1 * kUv0Strip + 0];
    v1.y = verticesArray[nonuniformEXT(uv0Id)].data[vertexId_1 * kUv0Strip + 1];

    v2.x = verticesArray[nonuniformEXT(uv0Id)].data[vertexId_2 * kUv0Strip + 0];
    v2.y = verticesArray[nonuniformEXT(uv0Id)].data[vertexId_2 * kUv0Strip + 1];

    vec2  bary = rayQueryGetIntersectionBarycentricsEXT(rayQuery, false);
    const vec3 barycentrics = vec3(1.0 - bary.x - bary.y, bary.x, bary.y);

    vec4 baseColor = texture(
        sampler2D(
            texture2DBindlessArray[nonuniformEXT(material.baseColorId)], 
            bindlessSampler[nonuniformEXT(material.baseColorSampler)]), 
        v0 * barycentrics.x + v1 * barycentrics.y + v2 * barycentrics.z);

    // Mask cutoff.
    if(baseColor.a < material.cutoff)
    {
        return false;
    }

    return true;
}

// Accurate rt hard shadow need sample mask.
layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    ivec2 colorSize = imageSize(rayShadowMask);

    uvec2 groupThreadId = remap8x8(gl_LocalInvocationIndex);
    uvec2 dispatchId = groupThreadId + gl_WorkGroupID.xy * 8;
    ivec2 workPos = ivec2(dispatchId);
    if(workPos.x >= colorSize.x || workPos.y >= colorSize.y)
    {
        return;
    }
    const vec2 uv = (vec2(workPos) + vec2(0.5f)) / vec2(colorSize);

    float shadow = 1.0f;
    const float deviceZ = texelFetch(inDepth, workPos, 0).r;

    if(deviceZ <= 0.0f)
    {
        imageStore(rayShadowMask, workPos, vec4(shadow));
        return;
    }

    // Offset retarget for new seeds each frame
    uvec2 offset = uvec2(vec2(0.754877669, 0.569840296) * (frameData.frameIndex.x) * uvec2(colorSize));
    uvec2 offsetId = uvec2(workPos) + offset;
    offsetId.x = offsetId.x % colorSize.x;
    offsetId.y = offsetId.y % colorSize.y;

    vec3 rayDir = -lightDirection;
	{
		vec2 e = vec2(
            samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d(offsetId.x, offsetId.y, 0, 0u),
            samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d(offsetId.x, offsetId.y, 0, 1u)
        );

        float lightRadiusRandom = lightRadius * e.x;
        float randomAngle = e.y * 2.0f * kPI;
        vec2 diskUv = vec2(cos(randomAngle), sin(randomAngle)) * lightRadiusRandom;

		vec3 N = rayDir;
		vec3 dPdu = cross(N, (abs(N.x) > 1e-6f) ? vec3(1, 0, 0) : vec3(0, 1, 0));
		vec3 dPdv = cross(dPdu, N);

		rayDir += dPdu * diskUv.x + dPdv * diskUv.y;
        rayDir = normalize(rayDir);
	}

    float dtRand = 1.0f + samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d(offsetId.x, offsetId.y, 0, 2u);

    {
        vec3 worldPos = getWorldPos(uv, deviceZ, frameData);
        uint rayFlags = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT | gl_RayFlagsCullBackFacingTrianglesEXT;

        rayQueryEXT rayQuery;
        rayQueryInitializeEXT(rayQuery, topLevelAS, rayFlags, 0xFF, worldPos, rayMinRange * dtRand, rayDir, rayMaxRange);
        while(rayQueryProceedEXT(rayQuery))
        {
            if(rayQueryGetIntersectionTypeEXT(rayQuery, false) == gl_RayQueryCandidateIntersectionTriangleEXT)
            {
                if(hitTest(rayQuery))
                {
                    rayQueryConfirmIntersectionEXT(rayQuery);
                }
            }
        }

        if (rayQueryGetIntersectionTypeEXT(rayQuery, true) != gl_RayQueryCommittedIntersectionNoneEXT)
        {
            shadow *= 0.0;
        }
    }

    imageStore(rayShadowMask, workPos, vec4(shadow));
}