#ifndef STATIC_MESH_COMMON_GLSL
#define STATIC_MESH_COMMON_GLSL

/*
** Physical based render code, develop by engineer: qiutanguu.
*/

#include "Common.glsl"

struct StaticMeshVertexRaw
{
    vec4 x0;
    vec4 x1;
    vec4 x2;
};

struct StaticMeshVertex
{
    vec3 position;
    vec3 normal;
    vec4 tangent;
    vec2 uv0;
};

StaticMeshVertex buildVertex(in StaticMeshVertexRaw raw)
{
    StaticMeshVertex result;
    result.position = raw.x0.xyz;

    result.normal.x  = raw.x0.w;
    result.normal.yz = raw.x1.xy;

    result.tangent.xy = raw.x1.zw;
    result.tangent.zw = raw.x2.xy;
    
    result.uv0 = raw.x2.zw;

    return result;
}

struct StaticMeshStandardPBR
{
    uint baseColorId;
    uint baseColorSampler;
    uint normalTexId;   
    uint normalSampler;

    uint specTexId; 
    uint specSampler;
    uint occlusionTexId; 
    uint occlusionSampler;

    uint emissiveTexId; 
    uint emissiveSampler;
    float cutoff;
    float faceCut; // > 1.0f is backface cut, < -1.0f is frontface cut, [-1.0f, 1.0f] is no face cut.

    vec4 baseColorMul;
	vec4 baseColorAdd;// x4

	float metalMul;
	float metalAdd;
	float roughnessMul;
	float roughnessAdd;// x4

	vec4 emissiveMul;
	vec4 emissiveAdd;
};

struct PerObjectData
{
    mat4 modelMatrix;
    mat4 modelMatrixPrev; // Prev-frame model matrix.
    // x4
   
    uint verticesArrayId; // Vertices buffer in bindless buffer id.    
    uint indicesArrayId; // Indices buffer in bindless buffer id.
    uint indexStartPosition; // Index start offset position.
    uint indexCount; // Mesh object info, used to build draw calls.
    // x4

    // .xyz is localspace center pos
    // .w   sphere radius
    vec4 sphereBounds;  
                       
    // .xyz extent XYZ
    vec3 extents;    
    uint bObjectMove; // object move state, = 1 when modelMatrix != modelMatrixPrev;

    StaticMeshStandardPBR material;
};

struct DrawIndirectCount
{
    uint count;
};

/**
*   typedef struct VkDrawIndexedIndirectCommand {
*       uint32_t    indexCount;
*       uint32_t    instanceCount;
*       uint32_t    firstIndex;
*       int32_t     vertexOffset;
*       uint32_t    firstInstance;
*   } VkDrawIndexedIndirectCommand;
*
*   typedef struct VkDrawIndirectCommand {
*       uint32_t    vertexCount;
*       uint32_t    instanceCount;
*       uint32_t    firstVertex;
*       uint32_t    firstInstance;
*   } VkDrawIndirectCommand;
**/
struct DrawIndirectCommand
{
    // VkDrawIndirectCommand

    // Build draw call data.
    uint vertexCount;
    uint instanceCount;
    uint firstVertex;
    uint firstInstance;

    // Object id for PerObjectData array indexing.
    uint objectId;
};

DrawIndirectCommand buildDefaultCommand()
{
    DrawIndirectCommand result;

    result.vertexCount = 0;
    result.instanceCount = 0;
    result.firstVertex = 0;
    result.firstInstance = 0;
    result.objectId = 0;

    return result;
}

// We use vkDrawIndirect to draw all vertex.
// Instead of binding index buffer, we load vertex data from index buffer.
// So the index count same with vertex count.

#endif