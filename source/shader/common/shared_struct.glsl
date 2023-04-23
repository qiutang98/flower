#ifndef SHARED_STRUCT_GLSL
#define SHARED_STRUCT_GLSL

#define kPI 3.141592653589793

const float kLog2 = log(2.0);

#define kPositionStrip 3
#define kNormalStrip   3
#define kUv0Strip      2
#define kTangentStrip  4

const float kMaxHalfFloat   = 65504.0f;
const float kMax11BitsFloat = 65024.0f;
const float kMax10BitsFloat = 64512.0f;
const vec3  kMax111110BitsFloat3 = vec3(kMax11BitsFloat, kMax11BitsFloat, kMax10BitsFloat);

struct CascadeShadowConfig
{
    int cascadeCount;
    int percascadeDimXY;
    float cascadeSplitLambda;
    float maxDrawDepthDistance;

    float shadowBiasConst; 
    float shadowBiasSlope; 
    float shadowFilterSize;
    float maxFilterSize;

    float cascadeBorderAdopt;
    float cascadeEdgeLerpThreshold;
    float pad0;
    float pad1;
};

// All units in kilometers
struct AtmosphereConfig
{
    float atmospherePreExposure;
    float pad0;
    float pad1;
    float pad2;

    vec3 absorptionColor;
    float absorptionLength;

	vec3 rayleighScatteringColor;
    float rayleighScatterLength;

    float multipleScatteringFactor;  
	float miePhaseFunctionG; 
    float bottomRadius; 
	float topRadius;

    vec3 mieScatteringColor;
    float mieScatteringLength;

    vec3 mieAbsColor;
    float mieAbsLength;

	vec3 mieAbsorption;  
	int viewRayMarchMinSPP;

	vec3 groundAlbedo;  
	int viewRayMarchMaxSPP;

	vec4 rayleighDensity[3];
	vec4 mieDensity[3]; 
	vec4 absorptionDensity[3];
};

struct SkyInfo
{
    vec3  color;
    float intensity;

    vec3  direction;
    int  shadowType; // Shadow type of this sky light.

    CascadeShadowConfig cacsadeConfig;
    AtmosphereConfig atmosphereConfig;
};

struct PerFrameData
{
    // .x is app runtime, .y is sin(.x), .z is cos(.x), .w is pad
    vec4 appTime;

    // .x is frame count, .y is frame count % 8, .z is frame count % 16, .w is frame count % 32
    uvec4 frameIndex;

    // Camera world space position.
    vec4 camWorldPos;

    // .x fovy, .y aspectRatio, .z nearZ, .w farZ
    vec4 camInfo;
    
    // prev-frame's cam info.
    vec4 camInfoPrev;

    // Camera matrixs.
    mat4 camView;
    mat4 camProj;
    mat4 camViewProj;

    // Camera inverse matrixs.
    mat4 camInvertView;
    mat4 camInvertProj;
    mat4 camInvertViewProj;

    // Camera matrix remove jitter effects.
    mat4 camProjNoJitter;
    mat4 camViewProjNoJitter;

    // Camera invert matrixs no jitter effects.
    mat4 camInvertProjNoJitter;
    mat4 camInvertViewProjNoJitter;

    // Prev-frame camera infos.
    mat4 camViewProjPrev;
    mat4 camViewProjPrevNoJitter;

    // Camera frustum planes for culling.
    vec4 frustumPlanes[6];

    // Halton sequence jitter data, .xy is current frame jitter data, .zw is prev frame jitter data.
    vec4 jitterData;
    
    uint  jitterPeriod;        // jitter period for jitter data.
    uint  bEnableJitter;       // Is main camera enable jitter in this frame.
    float basicTextureLODBias; // Lod basic texture bias when render mesh, used when upscale need.
    uint  bCameraCut;          // Camera cut in this frame or not.

    uint skyValid; // sky is valid.
    uint skySDSMValid;
    float fixExposure;
    uint bAutoExposure;

    SkyInfo sky;
};

// See GPUStaticMeshStandardPBR in shader_struct.h
struct MaterialStandardPBR
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
    // > 1.0f is backface cut, < -1.0f is frontface cut, [-1.0f, 1.0f] is no face cut.
    float faceCut; 

    // Base color operator.
    vec4 baseColorMul;
    vec4 baseColorAdd;

    // Metal operator.
    float metalMul;
    float metalAdd;
    // Roughness operator.
    float roughnessMul;
    float roughnessAdd;

    // Emissive operator.
    vec4 emissiveMul;
    vec4 emissiveAdd;
};

// Memory layout same with mesh_misc.h
// Raw data of one static mesh vertex.
struct StaticMeshVertexRaw
{
    vec4 x0;
    vec4 x1;
    vec4 x2;
};
// Static mesh vertex.
struct StaticMeshVertex
{
    vec3 position;
    vec3 normal;
    vec4 tangent;
    vec2 uv0;
};
StaticMeshVertex buildVertex(in const StaticMeshVertexRaw raw)
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

struct StaticMeshPerObjectData
{
    // Material for static mesh.
    MaterialStandardPBR material;

    // Current-frame model matrix.
    mat4 modelMatrix;

    // Prev-frame model matrix.
    mat4 modelMatrixPrev; 
   
    uint uv0sArrayId;    // Vertices buffer in bindless buffer id.    
    uint positionsArrayId;   // Positions buffer in bindless buffer id.
    uint indicesArrayId;     // Indices buffer in bindless buffer id.
    uint indexStartPosition; // Index start offset position.

    vec4 sphereBounds;

    vec3 extents;
    uint indexCount;         // Mesh object info, used to build draw calls.

    uint sceneNodeId; // Object id of scene node.
    uint bSelected; // Can pack with sceneNodeId like shader.
    uint tangentsArrayId;
    uint normalsArrayId;
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
struct StaticMeshDrawCommand
{
    // VkDrawIndirectCommand

    // Build draw call data.
    uint vertexCount;
    uint instanceCount;
    uint firstVertex;
    uint firstInstance;

    // Object id for StaticMeshPerObjectData array indexing.
    uint objectId;
    uint pad0;
    uint pad1;
    uint pad2;
};
StaticMeshDrawCommand defaultStaticMeshDrawCommand()
{
    StaticMeshDrawCommand result;

    result.vertexCount   = 0;
    result.instanceCount = 0;
    result.firstVertex   = 0;
    result.firstInstance = 0;
    result.objectId      = 0;

    return result;
}
// We use vkDrawIndirect to draw all vertex.
// Instead of binding index buffer, we load vertex data from index buffer.
// So the index count same with vertex count.

struct CascadeInfo
{
    mat4 viewProj;
    vec4 frustumPlanes[6];
    vec4 cascadeScale;
};

struct Ray
{
	vec3 o;
	vec3 d;
};

Ray createRay(in vec3 p, in vec3 d)
{
	Ray r;
	r.o = p;
	r.d = d;
	return r;
}

struct ScreenSpaceRay 
{
    vec3 ssRayStart;
    vec3 ssRayEnd;
    vec3 ssViewRayEnd;
    vec3 uvRayStart;
    vec3 uvRay;
};

struct DispatchIndirectCommand 
{
    uint x;
    uint y;
    uint z;
    uint pad;
};

#endif