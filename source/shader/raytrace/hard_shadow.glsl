#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_ray_query : enable

#include "../common/shared_functions.glsl"

layout (set = 0, binding = 0) uniform writeonly image2D rayShadowMask;
layout (set = 0, binding = 1) uniform UniformFrameData { PerFrameData frameData; };
layout (set = 0, binding = 2) uniform accelerationStructureEXT topLevelAS;
layout (set = 0, binding = 3)  uniform texture2D inDepth;
layout (set = 0, binding = 4) readonly buffer SSBOPerObject { StaticMeshPerObjectData objectDatas[]; };

layout (set = 1, binding = 0) readonly buffer BindlessSSBOVertices { float data[]; } verticesArray[];
layout (set = 2, binding = 0) readonly buffer BindlessSSBOIndices { uint data[]; } indicesArray[];
layout (set = 3, binding = 0) uniform  texture2D texture2DBindlessArray[];
layout (set = 4, binding = 0) uniform  sampler samplerArray[];

#define SHARED_SAMPLER_SET 5
#include "../common/shared_sampler.glsl"

#define BLUE_NOISE_BUFFER_SET 6
#include "../common/shared_bluenoise.glsl"

bool hitTest(in rayQueryEXT rayQuery)
{
    int instanceCustomIndexEXT = rayQueryGetIntersectionInstanceIdEXT(rayQuery, false);
    const StaticMeshPerObjectData objectData = objectDatas[instanceCustomIndexEXT];
    const MaterialStandardPBR material = objectData.material;


    int primitiveID = int(objectData.indexStartPosition) + rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, false) * 3;



    const uint indicesId  = objectData.indicesArrayId;
    const uint uv0Id = objectData.uv0sArrayId;

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
        sampler2D(texture2DBindlessArray[nonuniformEXT(material.baseColorId)], samplerArray[nonuniformEXT(material.baseColorSampler)]), 
        v0 * barycentrics.x + v1 * barycentrics.y + v2 * barycentrics.z);

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
    if(deviceZ > 0.0f)
    {
        vec3 worldPos = getWorldPos(uv, deviceZ, frameData);


        uint rayFlags = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT | gl_RayFlagsCullBackFacingTrianglesEXT;

        const vec3 rayDirection = -normalize(frameData.sky.direction);
        const float rayMinRange = 0.01f;
        const float rayMaxRange = 1000.0f;

        rayQueryEXT rayQuery;
        rayQueryInitializeEXT(rayQuery, topLevelAS, rayFlags, 0xFF, worldPos, rayMinRange, rayDirection, rayMaxRange);
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

    // Final store.
    imageStore(rayShadowMask, workPos, vec4(shadow));
}