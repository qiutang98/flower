#version 460

#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable

#include "../common/shared_struct.glsl"
#include "../common/shared_functions.glsl"
#include "../common/shared_shading_model.glsl"

// Attributes need lerp.
struct VS2PS
{
    vec2 uv0;
    vec3 normal;
    vec3 tangent;
    vec3 bitangent;
    vec3 worldPos;
    vec4 posNDCPrevNoJitter;
    vec4 posNDCCurNoJitter;
};

layout (set = 0, binding = 0) uniform UniformFrameData { PerFrameData frameData; };
layout (set = 0, binding = 1) readonly buffer SSBOPerObject { StaticMeshPerObjectData objectDatas[]; };
layout (set = 0, binding = 2) readonly buffer SSBOIndirectDraws { StaticMeshDrawCommand drawCommands[]; };

layout (set = 1, binding = 0) readonly buffer BindlessSSBOVertices { float data[]; } verticesArray[];
layout (set = 2, binding = 0) readonly buffer BindlessSSBOIndices { uint data[]; } indicesArray[];
layout (set = 3, binding = 0) uniform  texture2D texture2DBindlessArray[];
layout (set = 4, binding = 0) uniform  sampler samplerArray[];

#ifdef VERTEX_SHADER ///////////// vertex shader start 

layout(location = 0) out flat uint outObjectId;
layout(location = 1) out flat uint outTriangleId;
layout(location = 2) out VS2PS vsOut;

void main()
{
    // Load object data.
    outObjectId = drawCommands[gl_DrawID].objectId;
    const StaticMeshPerObjectData objectData = objectDatas[outObjectId];

    // We get bindless array id first.
    const uint indicesId  = objectData.indicesArrayId;
    const uint positionId = objectData.positionsArrayId;
    const uint tangentId = objectData.tangentsArrayId;
    const uint normalId = objectData.normalsArrayId;
    const uint uv0Id = objectData.uv0sArrayId;

    // Vertex count same with index count, so vertex index same with index index.
    const uint indexId = gl_VertexIndex;

    // Then fetech vertex index from indices array.
    const uint vertexId = indicesArray[nonuniformEXT(indicesId)].data[indexId];

    // Now we can get triangle id easily.
    const uint triangleId = vertexId / 3;
    outTriangleId = triangleId;

    // Finally we get vertex info.
    vec3 position;
    vec4 tangent;
    vec2 uv0;
    vec3 normal;

    position.x = verticesArray[nonuniformEXT(positionId)].data[vertexId * kPositionStrip + 0];
    position.y = verticesArray[nonuniformEXT(positionId)].data[vertexId * kPositionStrip + 1];
    position.z = verticesArray[nonuniformEXT(positionId)].data[vertexId * kPositionStrip + 2];

    tangent.x = verticesArray[nonuniformEXT(tangentId)].data[vertexId * kTangentStrip + 0];
    tangent.y = verticesArray[nonuniformEXT(tangentId)].data[vertexId * kTangentStrip + 1];
    tangent.z = verticesArray[nonuniformEXT(tangentId)].data[vertexId * kTangentStrip + 2];
    tangent.w = verticesArray[nonuniformEXT(tangentId)].data[vertexId * kTangentStrip + 3];

    normal.x = verticesArray[nonuniformEXT(normalId)].data[vertexId * kNormalStrip + 0];
    normal.y = verticesArray[nonuniformEXT(normalId)].data[vertexId * kNormalStrip + 1];
    normal.z = verticesArray[nonuniformEXT(normalId)].data[vertexId * kNormalStrip + 2];

    uv0.x = verticesArray[nonuniformEXT(uv0Id)].data[vertexId * kUv0Strip + 0];
    uv0.y = verticesArray[nonuniformEXT(uv0Id)].data[vertexId * kUv0Strip + 1];

    // Uv0 ready.
    vsOut.uv0 = uv0;

    // All ready, start to do vertex space-transform.
    const mat4 modelMatrix = objectData.modelMatrix;

    // Local vertex position.
    const vec4 localPosition = vec4(position, 1.0f);
    const vec4 worldPosition = modelMatrix * localPosition;
    vsOut.worldPos = worldPosition.xyz / worldPosition.w;

    // Convert to clip space.
    gl_Position = frameData.camViewProj * worldPosition;

    // Non-uniform scale need normal matrix convert.
    // see http://www.lighthouse3d.com/tutorials/glsl-12-tutorial/the-normal-matrix/.
    const mat3 normalMatrix = transpose(inverse(mat3(modelMatrix)));
    vsOut.normal  = normalize(normalMatrix * normalize(normal));

    // Tangent direction don't care about non-uniform scale.
    // see http://www.lighthouse3d.com/tutorials/glsl-12-tutorial/the-normal-matrix/.
    vsOut.tangent = normalize(vec3(modelMatrix * vec4(tangent.xyz, 0.0)));

    // Gram-Schmidt re-orthogonalize. https://learnopengl.com/Advanced-Lighting/Normal-Mapping
    vsOut.tangent = normalize(vsOut.tangent - dot(vsOut.tangent, vsOut.normal) * vsOut.normal);

    // Then it's easy to compute bitangent now.
    // bitangent is assimp compute direction.
    // tangent.w = sign(dot(normalize(bitangent), normalize(cross(normal, tangent))));
    vsOut.bitangent = cross(vsOut.normal, vsOut.tangent) * tangent.w;

    // Compute velocity for static mesh. https://github.com/GPUOpen-Effects/FidelityFX-FSR2
    // FSR2 will perform better quality upscaling when more objects provide their motion vectors. 
    // It is therefore advised that all opaque, alpha-tested and alpha-blended objects should write their motion vectors for all covered pixels.
    vsOut.posNDCPrevNoJitter = frameData.camViewProjPrevNoJitter * objectData.modelMatrixPrev * localPosition;
    vsOut.posNDCCurNoJitter = frameData.camViewProjNoJitter * worldPosition;
}

#endif /////////////////////////// vertex shader end

#ifdef PIXEL_SHADER ////////////// pixel shader start 

vec4 tex(uint texId,uint samplerId,vec2 uv)
{
    return texture(sampler2D(texture2DBindlessArray[nonuniformEXT(texId)], samplerArray[nonuniformEXT(samplerId)]), uv, frameData.basicTextureLODBias);
}

layout(location = 0) in flat uint inObjectId;
layout(location = 1) in flat uint inTriangleId;
layout(location = 2) in VS2PS vsIn;

layout(location = 0) out vec4 outHDRSceneColor; // Scene hdr color: r16g16b16a16. .rgb store emissive color.
layout(location = 1) out vec4 outGBufferA; // GBuffer A: r8g8b8a8 unorm, .rgb store base color, .a is shading model id.
layout(location = 2) out vec4 outGBufferB; // GBuffer B: r16g16b16a16. store worldspace normal.
layout(location = 3) out vec4 outGBufferS; // GBuffer S: r8g8b8a8 unorm, .r is metal, .g is roughness, .b is mesh ao.
layout(location = 4) out vec2 outGBufferV; // GBuffer V: r16g16 sfloat, store velocity.
layout(location = 5) out uint outId; // Id texture, [0 : 14] is sceneNode id, [15 : 15] is selection bit. 
void main()
{
    // Load object data.
    const StaticMeshPerObjectData objectData = objectDatas[inObjectId];
    const MaterialStandardPBR material = objectData.material;

    outId = packToIdBuffer(objectData.sceneNodeId, objectData.bSelected);

    // Load base color and cut off alpha.
    vec4 baseColor = tex(material.baseColorId, material.baseColorSampler, vsIn.uv0);
    baseColor = baseColor * material.baseColorMul + material.baseColorAdd;
    if(baseColor.a < material.cutoff)
    {
        discard;
    }

    // Emissive color.
    vec4 emissiveColor = tex(material.emissiveTexId, material.emissiveSampler, vsIn.uv0);
    emissiveColor = emissiveColor * material.emissiveMul + material.emissiveAdd;

    // World normal build.
    vec4 normalTex = tex(material.normalTexId, material.normalSampler, vsIn.uv0);
    vec3 worldNormal;
    {
        const mat3 tbn = mat3(normalize(vsIn.tangent), normalize(vsIn.bitangent), normalize(vsIn.normal));

        // Remap to [-1, 1].
        vec2 xy = 2.0 * normalTex.rg - 1.0;

        // Construct z.
        float z = sqrt(1.0 - dot(xy, xy));

        worldNormal = normalize(tbn * vec3(xy, z));
    }

    // Specular texture.
    vec4 specularTex = tex(material.specTexId, material.specSampler, vsIn.uv0);
    float roughness = clamp(specularTex.g * material.roughnessMul + material.roughnessAdd, 0.0, 1.0);
    float metallic  = clamp(specularTex.b * material.metalMul     + material.metalAdd,     0.0, 1.0);

    // Occlusion texture.
    vec4 occlusionTex = tex(material.occlusionTexId, material.occlusionSampler, vsIn.uv0);
    float meshAo = clamp(occlusionTex.r, 0.0, 1.0);

    // Scene hdr color. r16g16b16a16 sfloat.
    outHDRSceneColor.rgb = emissiveColor.rgb; // Store emissive color in RGB channel.
    outHDRSceneColor.a = 0.0f;

    // GBufferA: r8g8b8a8 unorm.
    outGBufferA.rgb = baseColor.rgb; // Output base color in GBuffer A rgb channel.
    outGBufferA.a = material.shadeingModel; // Shading model id.

    // GBuffer B: r16g16b16a16.
    outGBufferB.rgb = worldNormal; 
    outGBufferB.w = float(inObjectId);

    // GBuffer S: r8g8b8a8 unorm.
    outGBufferS.r = metallic;  // Metalic
    outGBufferS.g = roughness; // Actually it is perceptualRoughness.
    outGBufferS.b = meshAo;    // Mesh Ao
    outGBufferS.a = length(fwidth(normalize(vsIn.normal))) / length(fwidth(vsIn.worldPos));

    // Velocity output.
    outGBufferV = (vsIn.posNDCPrevNoJitter.xy / vsIn.posNDCPrevNoJitter.w) - (vsIn.posNDCCurNoJitter.xy / vsIn.posNDCCurNoJitter.w);

    // Also can do this if jitter:
    // const vec2 cancelJitter = frameData.jitterData.zw - frameData.jitterData.xy;
    // outGBufferV -= cancelJitter;

    // Transform motion vector from NDC space to UV space (+Y is top-down).
    outGBufferV *= vec2(0.5f, -0.5f);
}

#endif //////////////////////////// pixel shader end