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
#include "common_lighting.glsl"

layout (set = 0, binding = 0) uniform writeonly image2D imageSSGIResult;
layout (set = 0, binding = 1) uniform UniformFrameData { PerFrameData frameData; };
layout (set = 0, binding = 2) uniform accelerationStructureEXT topLevelAS;
layout (set = 0, binding = 3) uniform texture2D inDepth;
layout (set = 0, binding = 4) buffer  SSBOPerObject    { PerObjectInfo objectDatas[];              };
layout (set = 0, binding = 5) uniform textureCube inSkyIrradiance;
layout (set = 0, binding = 6) uniform texture2D inGbufferB;

layout(set = 3, binding = 0) buffer BindlessSSBOVertices{ float data[]; } verticesArray[];
layout(set = 4, binding = 0) buffer BindlessSSBOIndices{ uint data[]; } indicesArray[];
layout(set = 5, binding = 0) uniform sampler bindlessSampler[];
layout(set = 6, binding = 0) uniform  texture2D texture2DBindlessArray[];

layout (push_constant) uniform PushConsts 
{  
    float rayMinRange;
    float rayMaxRange;
};

struct HitPayload
{
    vec3 color;
    vec3 position;
    vec3 normal;
};

HitPayload hitTest(in rayQueryEXT rayQuery)
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
    const uint normalId   = meshInfo.normalsArrayId;
    const uint positionId = meshInfo.positionsArrayId;

    // Hit triangle id.
    const uint vertexId_0 = indicesArray[nonuniformEXT(indicesId)].data[primitiveID + 0];
    const uint vertexId_1 = indicesArray[nonuniformEXT(indicesId)].data[primitiveID + 1];
    const uint vertexId_2 = indicesArray[nonuniformEXT(indicesId)].data[primitiveID + 2];

    vec2  bary = rayQueryGetIntersectionBarycentricsEXT(rayQuery, false);
    const vec3 barycentrics = vec3(1.0 - bary.x - bary.y, bary.x, bary.y);

    vec2 uv;
    {
        vec2 v0, v1, v2;
        v0.x = verticesArray[nonuniformEXT(uv0Id)].data[vertexId_0 * kUv0Strip + 0];
        v0.y = verticesArray[nonuniformEXT(uv0Id)].data[vertexId_0 * kUv0Strip + 1];

        v1.x = verticesArray[nonuniformEXT(uv0Id)].data[vertexId_1 * kUv0Strip + 0];
        v1.y = verticesArray[nonuniformEXT(uv0Id)].data[vertexId_1 * kUv0Strip + 1];

        v2.x = verticesArray[nonuniformEXT(uv0Id)].data[vertexId_2 * kUv0Strip + 0];
        v2.y = verticesArray[nonuniformEXT(uv0Id)].data[vertexId_2 * kUv0Strip + 1];

        uv =  v0 * barycentrics.x + v1 * barycentrics.y + v2 * barycentrics.z;
    }

    vec4 baseColor = texture(sampler2D(texture2DBindlessArray[nonuniformEXT(material.baseColorId)], pointClampEdgeSampler), uv);

    vec3 normal;
    {
        vec3 n0, n1, n2;
        n0.x = verticesArray[nonuniformEXT(normalId)].data[vertexId_0 * kNormalStrip + 0];
        n0.y = verticesArray[nonuniformEXT(normalId)].data[vertexId_0 * kNormalStrip + 1];
        n0.z = verticesArray[nonuniformEXT(normalId)].data[vertexId_0 * kNormalStrip + 2];

        n1.x = verticesArray[nonuniformEXT(normalId)].data[vertexId_1 * kNormalStrip + 0];
        n1.y = verticesArray[nonuniformEXT(normalId)].data[vertexId_1 * kNormalStrip + 1];
        n1.z = verticesArray[nonuniformEXT(normalId)].data[vertexId_1 * kNormalStrip + 2];

        n2.x = verticesArray[nonuniformEXT(normalId)].data[vertexId_2 * kNormalStrip + 0];
        n2.y = verticesArray[nonuniformEXT(normalId)].data[vertexId_2 * kNormalStrip + 1];
        n2.z = verticesArray[nonuniformEXT(normalId)].data[vertexId_2 * kNormalStrip + 2];

        normal = n0 * barycentrics.x + n1 * barycentrics.y + n2 * barycentrics.z;
    }

    vec3 position;
    {
        vec3 p0, p1, p2;

        p0.x = verticesArray[nonuniformEXT(positionId)].data[vertexId_0 * kPositionStrip + 0];
        p0.y = verticesArray[nonuniformEXT(positionId)].data[vertexId_0 * kPositionStrip + 1];
        p0.z = verticesArray[nonuniformEXT(positionId)].data[vertexId_0 * kPositionStrip + 2];

        p1.x = verticesArray[nonuniformEXT(positionId)].data[vertexId_1 * kPositionStrip + 0];
        p1.y = verticesArray[nonuniformEXT(positionId)].data[vertexId_1 * kPositionStrip + 1];
        p1.z = verticesArray[nonuniformEXT(positionId)].data[vertexId_1 * kPositionStrip + 2];

        p2.x = verticesArray[nonuniformEXT(positionId)].data[vertexId_2 * kPositionStrip + 0];
        p2.y = verticesArray[nonuniformEXT(positionId)].data[vertexId_2 * kPositionStrip + 1];
        p2.z = verticesArray[nonuniformEXT(positionId)].data[vertexId_2 * kPositionStrip + 2];

        position = p0 * barycentrics.x + p1 * barycentrics.y + p2 * barycentrics.z;
    }


    HitPayload result;
 
    const mat4 modelMatrix = objectData.modelMatrix;
    const vec4 localPosition = vec4(position, 1.0f);
    const vec4 worldPosition = modelMatrix * localPosition;

    const mat3 normalMatrix = transpose(inverse(mat3(modelMatrix)));


    result.normal  = normalize(normalMatrix * normalize(normal));
    result.position = worldPosition.xyz / worldPosition.w;
    result.color = baseColor.xyz;

    return result;
}

// Accurate rt hard shadow need sample mask.
layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    ivec2 colorSize = imageSize(imageSSGIResult);

    uvec2 groupThreadId = remap8x8(gl_LocalInvocationIndex);
    uvec2 dispatchId = groupThreadId + gl_WorkGroupID.xy * 8;
    ivec2 workPos = ivec2(dispatchId);
    if(workPos.x >= colorSize.x || workPos.y >= colorSize.y)
    {
        return;
    }
    const vec2 uv = (vec2(workPos) + vec2(0.5f)) / vec2(colorSize);
    const float deviceZ = texture(sampler2D(inDepth, pointClampEdgeSampler), uv).r;

    if(deviceZ <= 0.0f)
    {
        imageStore(imageSSGIResult, workPos, vec4(0.0));
        return;
    }

    // Offset retarget for new seeds each frame
    uvec2 offset = uvec2(vec2(0.754877669, 0.569840296) * (frameData.frameIndex.x) * uvec2(colorSize));
    uvec2 offsetId = uvec2(workPos) + offset;
    offsetId.x = offsetId.x % colorSize.x;
    offsetId.y = offsetId.y % colorSize.y;

    float u0 = samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d(offsetId.x, offsetId.y, 0, 0u);
    float u1 = samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d(offsetId.x, offsetId.y, 0, 1u);


    const vec4 inGbufferBValue = texture(sampler2D(inGbufferB, pointClampEdgeSampler), uv);
    const vec3 worldNormal = unpackWorldNormal(inGbufferBValue.rgb);

    const vec3 rayDir = // getReflectionDir(viewDir, viewNormal, u0, u1);
        importanceSampleCosine(vec2(u0, 1.0 - u0), worldNormal);

    float dtRand = 1.0f + u1;

    vec3 ssgiResult = vec3(0.0);

    if(frameData.bSkyComponentValid != 0)
    {
        vec3 worldPos = getWorldPos(uv, deviceZ, frameData);
        uint rayFlags = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT | gl_RayFlagsCullBackFacingTrianglesEXT;

        rayQueryEXT rayQuery;
        rayQueryInitializeEXT(rayQuery, topLevelAS, rayFlags, 0xFF, worldPos, rayMinRange * dtRand, rayDir, rayMaxRange);
        while(rayQueryProceedEXT(rayQuery))
        {
            if(rayQueryGetIntersectionTypeEXT(rayQuery, false) == gl_RayQueryCandidateIntersectionTriangleEXT)
            {
                HitPayload hitResult = hitTest(rayQuery);

                float NoL = max(0.0, dot(-normalize(frameData.sunLightInfo.direction), hitResult.normal));

                ssgiResult = hitResult.color * NoL;



                rayQueryConfirmIntersectionEXT(rayQuery);
            }
        }
    }

    imageStore(imageSSGIResult, workPos, vec4(ssgiResult, 0.0));
}