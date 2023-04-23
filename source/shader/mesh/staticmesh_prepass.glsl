#version 460

#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable

#include "../common/shared_struct.glsl"
#include "../common/shared_shading_model.glsl"

// Attributes need lerp.
struct VS2PS
{
    vec2 uv0;
    vec4 unjitterPos;
};

layout (set = 0, binding = 0) uniform UniformFrameData { PerFrameData frameData; };
layout (set = 0, binding = 1) readonly buffer SSBOPerObject { StaticMeshPerObjectData objectDatas[]; };
layout (set = 0, binding = 2) readonly buffer SSBOIndirectDraws { StaticMeshDrawCommand drawCommands[]; };
layout (set = 0, binding = 3, r8) uniform image2D outSelectionMask;

layout (set = 1, binding = 0) readonly buffer BindlessSSBOVertices { float data[]; } verticesArray[];
layout (set = 2, binding = 0) readonly buffer BindlessSSBOIndices { uint data[]; } indicesArray[];
layout (set = 3, binding = 0) uniform  texture2D texture2DBindlessArray[];
layout (set = 4, binding = 0) uniform  sampler samplerArray[];

#ifdef VERTEX_SHADER ///////////// vertex shader start 

layout(location = 0) out flat uint outObjectId;
layout(location = 2) out VS2PS vsOut;

void main()
{
    // Load object data.
    outObjectId = drawCommands[gl_DrawID].objectId;
    const StaticMeshPerObjectData objectData = objectDatas[outObjectId];

    // We get bindless array id first.
    const uint indicesId  = objectData.indicesArrayId;
    const uint positionId = objectData.positionsArrayId;
    const uint uv0Id = objectData.uv0sArrayId;

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
    gl_Position = frameData.camViewProj * worldPosition;

    vsOut.unjitterPos = frameData.camViewProjNoJitter * worldPosition;
}

#endif /////////////////////////// vertex shader end

#ifdef PIXEL_SHADER ////////////// pixel shader start 

vec4 tex(uint texId,uint samplerId,vec2 uv)
{
    return texture(sampler2D(texture2DBindlessArray[nonuniformEXT(texId)], samplerArray[nonuniformEXT(samplerId)]), uv, frameData.basicTextureLODBias);
}

layout(location = 0) in flat uint inObjectId;
layout(location = 2) in VS2PS vsIn;

void main()
{
    // Load object data.
    const StaticMeshPerObjectData objectData = objectDatas[inObjectId];
    const MaterialStandardPBR material = objectData.material;

    // Load base color and cut off alpha.
    vec4 baseColor = tex(material.baseColorId, material.baseColorSampler, vsIn.uv0);
    baseColor = baseColor * material.baseColorMul + material.baseColorAdd;
    if(baseColor.a < material.cutoff)
    {
        discard;
    }

    // Select mask, don't need z test.
    if(objectData.bSelected != 0)
    {
        vec3 projPosUnjitter = vsIn.unjitterPos.xyz / vsIn.unjitterPos.w;

        projPosUnjitter.xy = 0.5 * projPosUnjitter.xy + 0.5;
        projPosUnjitter.y  = 1.0 - projPosUnjitter.y;

        ivec2 storeMaskPos = ivec2(projPosUnjitter.xy * imageSize(outSelectionMask));
        imageStore(outSelectionMask, storeMaskPos, vec4(1.0));
    }
}

#endif //////////////////////////// pixel shader end