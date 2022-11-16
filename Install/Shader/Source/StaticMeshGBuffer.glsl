#version 460
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable

#include "StaticMeshCommon.glsl"
#include "ColorSpace.glsl"

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

layout (set = 0, binding = 0) uniform UniformView{  ViewData viewData; };
layout (set = 1, binding = 0) uniform UniformFrame{ FrameData frameData; };
layout (set = 2, binding = 0) buffer BindlessSSBOVertices{ StaticMeshVertexRaw data[]; } verticesArray[];
layout (set = 3, binding = 0) buffer BindlessSSBOIndices{ uint data[]; } indicesArray[];
layout (set = 4, binding = 0) uniform texture2D bindlessTexture2D[];
layout (set = 5, binding = 0) uniform sampler bindlessSampler[];
layout (set = 6, binding = 0) readonly buffer SSBOPerObject{PerObjectData objectDatas[];};
layout (set = 7, binding = 0) readonly buffer SSBOIndirectDraws{DrawIndirectCommand indirectCommands[]; };

#ifdef VERTEX_SHADER ///////////// vertex shader start 

layout(location = 0) out flat uint outObjectId;
layout(location = 1) out flat uint outTriangleId;
layout(location = 2) out VS2PS vsOut;

void main()
{
    // Load object data.
    outObjectId = indirectCommands[gl_DrawID].objectId;
    const PerObjectData objectData = objectDatas[outObjectId];

    // We get bindless array id first.
    const uint indicesId = objectData.indicesArrayId;
    const uint verticesId = objectData.verticesArrayId;

    // Vertex count same with index count, so vertex index same with index index.
    const uint indexId = gl_VertexIndex;

    // Then fetech vertex index from indices array.
    const uint vertexId = indicesArray[nonuniformEXT(indicesId)].data[indexId];

    const uint triangleId = vertexId / 3;
    outTriangleId = triangleId;

    // Finally we get vertex info.
    const StaticMeshVertexRaw rawVertex = verticesArray[nonuniformEXT(verticesId)].data[vertexId];
    const StaticMeshVertex vertex = buildVertex(rawVertex);

    vsOut.uv0 = vertex.uv0;

    // All ready, start to do vertex space-transform.
    const mat4 modelMatrix = objectData.modelMatrix;

    // Local vertex position.
    const vec4 localPosition = vec4(vertex.position, 1.0f);
    const vec4 worldPosition = modelMatrix * localPosition;
    vsOut.worldPos = worldPosition.xyz / worldPosition.w;

    // Convert to clip space.
    gl_Position = viewData.camViewProj * worldPosition;

    // Non-uniform scale need normal matrix convert.
    // see http://www.lighthouse3d.com/tutorials/glsl-12-tutorial/the-normal-matrix/.
    const mat3 normalMatrix = transpose(inverse(mat3(modelMatrix)));
    vsOut.normal  = normalize(normalMatrix * normalize(vertex.normal));

    // Tangent direction don't care about non-uniform scale.
    // see http://www.lighthouse3d.com/tutorials/glsl-12-tutorial/the-normal-matrix/.
    vsOut.tangent = normalize(vec3(modelMatrix * vec4(vertex.tangent.xyz, 0.0)));

    // Gram-Schmidt re-orthogonalize. https://learnopengl.com/Advanced-Lighting/Normal-Mapping
    vsOut.tangent = normalize(vsOut.tangent - dot(vsOut.tangent, vsOut.normal) * vsOut.normal);

    // Then it's easy to compute bitangent now.
    // bitangent is assimp compute direction.
    // tangent.w = sign(dot(normalize(bitangent), normalize(cross(normal, tangent))));
    vsOut.bitangent = cross(vsOut.normal, vsOut.tangent) * vertex.tangent.w;

    // Compute velocity for static mesh. https://github.com/GPUOpen-Effects/FidelityFX-FSR2
    // FSR2 will perform better quality upscaling when more objects provide their motion vectors. 
    // It is therefore advised that all opaque, alpha-tested and alpha-blended objects should write their motion vectors for all covered pixels.
    vsOut.posNDCPrevNoJitter = viewData.camViewProjPrevNoJitter * objectData.modelMatrixPrev * localPosition;
    vsOut.posNDCCurNoJitter = viewData.camViewProjNoJitter * worldPosition;
}

#endif /////////////////////////// vertex shader end

#ifdef PIXEL_SHADER ////////////// pixel shader start 

vec4 tex(uint texId,uint samplerId,vec2 uv)
{
    return texture(sampler2D(bindlessTexture2D[nonuniformEXT(texId)], bindlessSampler[nonuniformEXT(samplerId)]), uv, frameData.basicTextureLODBias);
}

layout(location = 0) in flat uint inObjectId;
layout(location = 1) in flat uint inTriangleId;
layout(location = 2) in VS2PS vsIn;

// Scene hdr color. .rgb store emissive color.
layout(location = 0) out vec4 outHDRSceneColor;

// GBuffer A: r8g8b8a8 unorm, .rgb store base color, .a is shading model id.
layout(location = 1) out vec4 outGBufferA;

// GBuffer B: r16g16b16a16 sfloat, .rgb store worldspace normal, .a is mesh id.
layout(location = 2) out vec4 outGBufferB;

// GBuffer S: r8g8b8a8 unorm, .r is metal, .g is roughness, .b is mesh ao.
layout(location = 3) out vec4 outGBufferS;

// GBuffer V: r16g16 sfloat, store velocity.
layout(location = 4) out vec2 outGBufferV;

void main()
{
    const PerObjectData objectData = objectDatas[inObjectId];
    const StaticMeshStandardPBR mat = objectData.material;

    const vec4 baseColor = tex(mat.baseColorId, mat.baseColorSampler, vsIn.uv0);
    if(baseColor.a < mat.cutoff)
    {
        discard;
    }
    // Output base color in GBuffer A rgb channel.
    outGBufferA.rgb = inputColorPrepare(baseColor.rgb);
    // outGBufferA.rgb = simpleHashColor(inTriangleId);

    // Shading model id.
    outGBufferA.a = kShadingModelStandardPBR;

    // Emissive color.
    vec4 emissiveTex = tex(mat.emissiveTexId, mat.emissiveSampler, vsIn.uv0);
    outHDRSceneColor.rgb = inputColorPrepare(emissiveTex.rgb);

    // World normal build.
    vec4 normalTex = tex(mat.normalTexId, mat.normalSampler, vsIn.uv0);
    vec3 worldNormal;
    {
        const mat3 tbn = mat3(normalize(vsIn.tangent), normalize(vsIn.bitangent), normalize(vsIn.normal));

        // Remap to [-1, 1].
        vec2 xy = 2.0 * normalTex.rg - 1.0;

        // Construct z.
        float z = sqrt(1.0 - dot(xy, xy));

        worldNormal = normalize(tbn * vec3(xy, z));
    }
    outGBufferB.rgb = worldNormal; // Output world normal in GBuffer B rgb channel.
    outGBufferB.a = float(inTriangleId); // float 16 is enough? need some pack here?

    // Specular texture.
    vec4 specularTex = tex(mat.specTexId, mat.specSampler, vsIn.uv0);
    outGBufferS.r = specularTex.b; // metal

    // Actually it is perceptualRoughness.
    outGBufferS.g = clamp(specularTex.g, 0.0, 1.0); // roughness
    outGBufferS.b = tex(mat.occlusionTexId, mat.occlusionSampler, vsIn.uv0).r; // mesh ao

    // Velocity output.
    outGBufferV = (vsIn.posNDCPrevNoJitter.xy / vsIn.posNDCPrevNoJitter.w) - (vsIn.posNDCCurNoJitter.xy / vsIn.posNDCCurNoJitter.w);

    // Also can do this if jitter:
    // const vec2 cancelJitter = frameData.jitterData.zw - frameData.jitterData.xy;
    // outGBufferV -= cancelJitter;

    // Transform motion vector from NDC space to UV space (+Y is top-down).
    outGBufferV *= vec2(0.5f, -0.5f);
}

#endif //////////////////////////// pixel shader end