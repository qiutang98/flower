#version 460

#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable

#include "common_shader.glsl"

#ifndef BOUNDS_DRAW_DEBUG_LINE
#define BOUNDS_DRAW_DEBUG_LINE (DEBUG_LINE_ENABLE && 0)
#endif


#ifdef STATIC_MESH_PREPASS_CULL_PASS

layout (set = 0, binding = 0) uniform UniformFrameData{ PerFrameData frameData; };
layout (set = 0, binding = 1) readonly buffer SSBOPerObject { PerObjectInfo objectDatas[]; };
layout (set = 0, binding = 2) buffer SSBOIndirectDraws { StaticMeshDrawCommand drawCommands[]; };
layout (set = 0, binding = 3) buffer SSBODrawCount{ uint drawCount; };

layout (push_constant) uniform PushConsts 
{
    // Total static mesh count need to cull.  
    uint cullCount; 
};

layout(local_size_x = 64) in;
void main()
{
    // get working id.
    uint idx = gl_GlobalInvocationID.x;
    if(idx >= cullCount)
    {
        return;
    }

    const PerObjectInfo objectData = objectDatas[idx];
    const MeshInfo meshInfo = objectData.meshInfoData;

    if(frameData.renderType == ERendererType_ReflectionCapture)
    {
        if(meshInfo.meshType != EMeshType_StaticMesh)
        {
            return;
        }
    }

    vec3 localPos = meshInfo.sphereBounds.xyz;
	vec4 worldPos = objectData.modelMatrix * vec4(localPos, 1.0f);

    // local to world normal matrix.
	mat3 normalMatrix = transpose(inverse(mat3(objectData.modelMatrix)));
	mat3 world2Local = inverse(normalMatrix);

	// frustum culling test.
	for (int i = 0; i < 6; i++) 
	{
        vec3 worldSpaceN = frameData.frustumPlanes[i].xyz;
        float castDistance = dot(worldPos.xyz, worldSpaceN);

		// transfer to local matrix and use abs get first dimensions project value,
		// use that for test.
		vec3 localNormal = world2Local * worldSpaceN;
		float absDiff = dot(abs(localNormal), meshInfo.extents.xyz);
		if (castDistance + absDiff + frameData.frustumPlanes[i].w < 0.0)
		{
            // no visibile
            return; 
		}
	}

    // Build draw command if visible.
    {
        uint drawId = atomicAdd(drawCount, 1);
        drawCommands[drawId].objectId = idx;

        // We fetech vertex by index, so vertex count is index count.
        drawCommands[drawId].vertexCount = meshInfo.indicesCount;
        drawCommands[drawId].firstVertex = meshInfo.indexStartPosition;

        // We fetch vertex in vertex shader, so instancing is unused when rendering.
        drawCommands[drawId].instanceCount = 1;
    }
}

#endif // STATIC_MESH_PREPASS_CULL_PASS

#ifdef STATIC_MESH_PREPASS

// Attributes need lerp.
struct VS2PS
{
    vec2 uv0;
};

layout (set = 0, binding = 0) uniform UniformFrameData { PerFrameData frameData; };
layout (set = 0, binding = 1) readonly buffer SSBOPerObject { PerObjectInfo objectDatas[]; };
layout (set = 0, binding = 2) readonly buffer SSBOIndirectDraws { StaticMeshDrawCommand drawCommands[]; };

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
    gl_Position = frameData.camViewProj * worldPosition;
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

#endif // PIXEL_SHADER

#endif // STATIC_MESH_PREPASS

#ifdef STATIC_MESH_GBUFFER_CULL_PASS

layout (set = 0, binding = 0) uniform UniformFrameData{ PerFrameData frameData; };
layout (set = 0, binding = 1) readonly buffer SSBOPerObject { PerObjectInfo objectDatas[]; };
layout (set = 0, binding = 2) buffer SSBOIndirectDraws { StaticMeshDrawCommand drawCommands[]; };
layout (set = 0, binding = 3) buffer SSBODrawCount{ uint drawCount; };
layout (set = 0, binding = 4) uniform texture2D inHzbFurthest;
layout (set = 0, binding = 5) buffer  SSBOLineVertexBuffers  { LineDrawVertex lineVertices[]; };
layout (set = 0, binding = 6) buffer  SSBODrawCmdCountBuffer  { uint lineCount; };

layout (push_constant) uniform PushConsts 
{
    // Total static mesh count need to cull.  
    uint cullCount; 
    uint hzbMipCount;
    vec2 hzbSrcSize;
};

layout(local_size_x = 64) in;
void main()
{
    // get working id.
    uint idx = gl_GlobalInvocationID.x;
    if(idx >= cullCount)
    {
        return;
    }

    const PerObjectInfo objectData = objectDatas[idx];
    const MeshInfo meshInfo = objectData.meshInfoData;

    if(frameData.renderType == ERendererType_ReflectionCapture)
    {
        if(meshInfo.meshType != EMeshType_StaticMesh)
        {
            return;
        }
    }

    const mat4 mvp = frameData.camViewProj * objectData.modelMatrix;
    const vec3 extent = meshInfo.extents;

    vec3 localPos = meshInfo.sphereBounds.xyz;
	vec4 worldPos = objectData.modelMatrix * vec4(localPos, 1.0f);

    // local to world normal matrix.
	mat3 normalMatrix = transpose(inverse(mat3(objectData.modelMatrix)));
	mat3 world2Local = inverse(normalMatrix);

	// frustum culling test.
	for (int i = 0; i < 6; i++) 
	{
        vec3 worldSpaceN = frameData.frustumPlanes[i].xyz;
        float castDistance = dot(worldPos.xyz, worldSpaceN);

		// transfer to local matrix and use abs get first dimensions project value,
		// use that for test.
		vec3 localNormal = world2Local * worldSpaceN;
		float absDiff = dot(abs(localNormal), meshInfo.extents.xyz);
		if (castDistance + absDiff + frameData.frustumPlanes[i].w < 0.0)
		{
            return; // no visibile
		}
	}

    // Hzb culling test.
    {
        // Cast eight vertex to screen space, then compute texel size, then sample hzb, then compare depth occlusion state.


        const vec3 uvZ0 = projectPos(localPos + extent * vec3( 1.0,  1.0,  1.0), mvp);
        const vec3 uvZ1 = projectPos(localPos + extent * vec3(-1.0,  1.0,  1.0), mvp);
        const vec3 uvZ2 = projectPos(localPos + extent * vec3( 1.0, -1.0,  1.0), mvp);
        const vec3 uvZ3 = projectPos(localPos + extent * vec3( 1.0,  1.0, -1.0), mvp);
        const vec3 uvZ4 = projectPos(localPos + extent * vec3(-1.0, -1.0,  1.0), mvp);
        const vec3 uvZ5 = projectPos(localPos + extent * vec3( 1.0, -1.0, -1.0), mvp);
        const vec3 uvZ6 = projectPos(localPos + extent * vec3(-1.0,  1.0, -1.0), mvp);
        const vec3 uvZ7 = projectPos(localPos + extent * vec3(-1.0, -1.0, -1.0), mvp);

        vec3 maxUvz = max(max(max(max(max(max(max(uvZ0, uvZ1), uvZ2), uvZ3), uvZ4), uvZ5), uvZ6), uvZ7);
        vec3 minUvz = min(min(min(min(min(min(min(uvZ0, uvZ1), uvZ2), uvZ3), uvZ4), uvZ5), uvZ6), uvZ7);

        if(maxUvz.z < 1.0f && minUvz.z > 0.0f)
        {
            const vec2 bounds = maxUvz.xy - minUvz.xy;

            const float edge = max(1.0, max(bounds.x, bounds.y) * max(hzbSrcSize.x, hzbSrcSize.y));
            int mipLevel = int(min(ceil(log2(edge)), hzbMipCount - 1));

            const vec2 mipSize = vec2(textureSize(inHzbFurthest, mipLevel));
            const ivec2 samplePosMax = ivec2(saturate(maxUvz.xy) * mipSize);
            const ivec2 samplePosMin = ivec2(saturate(minUvz.xy) * mipSize);

            vec4 occ = vec4(
                texelFetch(inHzbFurthest, samplePosMax.xy, mipLevel).x, 
                texelFetch(inHzbFurthest, samplePosMin.xy, mipLevel).x, 
                texelFetch(inHzbFurthest, ivec2(samplePosMax.x, samplePosMin.y), mipLevel).x, 
                texelFetch(inHzbFurthest, ivec2(samplePosMin.x, samplePosMax.y), mipLevel).x);

            float occDepth = min(occ.w, min(occ.z, min(occ.x, occ.y)));

            // Occlusion, pre-return.
            if(occDepth > maxUvz.z)
            {
                return;
            }
        }
    }

#if BOUNDS_DRAW_DEBUG_LINE
    {
        const vec3 p0 = posTransform(localPos + extent * vec3( 1.0,  1.0,  1.0), objectData.modelMatrix);
        const vec3 p1 = posTransform(localPos + extent * vec3(-1.0,  1.0,  1.0), objectData.modelMatrix);
        const vec3 p2 = posTransform(localPos + extent * vec3( 1.0, -1.0,  1.0), objectData.modelMatrix);
        const vec3 p3 = posTransform(localPos + extent * vec3( 1.0,  1.0, -1.0), objectData.modelMatrix);
        const vec3 p4 = posTransform(localPos + extent * vec3(-1.0, -1.0,  1.0), objectData.modelMatrix);
        const vec3 p5 = posTransform(localPos + extent * vec3( 1.0, -1.0, -1.0), objectData.modelMatrix);
        const vec3 p6 = posTransform(localPos + extent * vec3(-1.0,  1.0, -1.0), objectData.modelMatrix);
        const vec3 p7 = posTransform(localPos + extent * vec3(-1.0, -1.0, -1.0), objectData.modelMatrix);

        uint drawId = atomicAdd(lineCount, 24);

        lineVertices[drawId + 0].worldPos = p0;
        lineVertices[drawId + 1].worldPos = p1;

        lineVertices[drawId + 2].worldPos = p0;
        lineVertices[drawId + 3].worldPos = p2;

        lineVertices[drawId + 4].worldPos = p0;
        lineVertices[drawId + 5].worldPos = p3;

        lineVertices[drawId + 6].worldPos = p6;
        lineVertices[drawId + 7].worldPos = p7;

        lineVertices[drawId + 8].worldPos = p5;
        lineVertices[drawId + 9].worldPos = p7;

        lineVertices[drawId + 10].worldPos = p4;
        lineVertices[drawId + 11].worldPos = p7;

        lineVertices[drawId + 12].worldPos = p1;
        lineVertices[drawId + 13].worldPos = p6;

        lineVertices[drawId + 14].worldPos = p2;
        lineVertices[drawId + 15].worldPos = p5;

        lineVertices[drawId + 16].worldPos = p1;
        lineVertices[drawId + 17].worldPos = p4;

        lineVertices[drawId + 18].worldPos = p2;
        lineVertices[drawId + 19].worldPos = p4;

        lineVertices[drawId + 20].worldPos = p3;
        lineVertices[drawId + 21].worldPos = p6;

        lineVertices[drawId + 22].worldPos = p3;
        lineVertices[drawId + 23].worldPos = p5;
    }
#endif

    // Build draw command if visible.
    {
        uint drawId = atomicAdd(drawCount, 1);
        drawCommands[drawId].objectId = idx;

        // We fetech vertex by index, so vertex count is index count.
        drawCommands[drawId].vertexCount = meshInfo.indicesCount;
        drawCommands[drawId].firstVertex = meshInfo.indexStartPosition;

        // We fetch vertex in vertex shader, so instancing is unused when rendering.
        drawCommands[drawId].instanceCount = 1;
    }
}

#endif // STATIC_MESH_GBUFFER_CULL_PASS

#ifdef STATIC_MESH_GBUFFER_PASS

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
layout (set = 0, binding = 1) readonly buffer SSBOPerObject { PerObjectInfo objectDatas[]; };
layout (set = 0, binding = 2) readonly buffer SSBOIndirectDraws { StaticMeshDrawCommand drawCommands[]; };

layout (set = 1, binding = 0) readonly buffer BindlessSSBOVertices { float data[]; } verticesArray[];
layout (set = 2, binding = 0) readonly buffer BindlessSSBOIndices { uint data[]; } indicesArray[];
layout (set = 3, binding = 0) uniform  texture2D texture2DBindlessArray[];
layout (set = 4, binding = 0) uniform  sampler samplerArray[];

#ifdef VERTEX_SHADER ///////////// vertex shader start 

layout(location = 0) out flat uint outObjectId;
layout(location = 1) out VS2PS vsOut;

void main()
{
    // Load object data.
    outObjectId = drawCommands[gl_DrawID].objectId;
    const PerObjectInfo objectData = objectDatas[outObjectId];

    // We get bindless array id first.
    const uint indicesId  = objectData.meshInfoData.indicesArrayId;
    const uint positionId = objectData.meshInfoData.positionsArrayId;
    const uint tangentId = objectData.meshInfoData.tangentsArrayId;
    const uint normalId = objectData.meshInfoData.normalsArrayId;
    const uint uv0Id = objectData.meshInfoData.uv0sArrayId;

    // Vertex count same with index count, so vertex index same with index index.
    const uint indexId = gl_VertexIndex;

    // Then fetech vertex index from indices array.
    const uint vertexId = indicesArray[nonuniformEXT(indicesId)].data[indexId];

    // Now we can get triangle id easily.
    const uint triangleId = vertexId / 3;

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
layout(location = 1) in VS2PS vsIn;

layout(location = 0) out vec4 outHDRSceneColor; // Scene hdr color: r16g16b16a16. .rgb store emissive color.
layout(location = 1) out vec4 outGBufferA; // GBuffer A: r8g8b8a8 unorm, .rgb store base color, .a is shading model id.
layout(location = 2) out vec4 outGBufferB; // GBuffer B: r10g10b10a2. rgb store worldspace normal.
layout(location = 3) out vec4 outGBufferS; // GBuffer S: r8g8b8a8 unorm, .r is metal, .g is roughness, .b is mesh ao.
layout(location = 4) out vec2 outGBufferV; // GBuffer V: r16g16 sfloat, store velocity.
layout(location = 5) out float outGBufferId;

float geometricAA(vec3 N, float r)
{
    //reference: "Improved Geometric Specular Antialiasing"
    float kappa = 0.18f; // threshold

    float pixelVariance = 0.5f; // mix(0.5f, 0.75f, 1.0 - r);
    float pxVar2 = pixelVariance * pixelVariance;

    vec3 N_U = dFdxFine(N);
    vec3 N_V = dFdyFine(N);

    // Squared lengths
    float lengthN_U2 = dot(N_U, N_U);
    float lengthN_V2 = dot(N_V, N_V);

    float variance = pxVar2 * (lengthN_V2 + lengthN_U2); // max((lengthN_V2 + lengthN_U2), pow((lengthN_V2 + lengthN_U2), 2.0));    
    float kernelRoughness2 = min(2.f * variance, kappa);
    float rFiltered = clamp(sqrt(r * r + kernelRoughness2), 0.f, 1.f);

    return rFiltered;
}
 
void main()
{
    // Load object data.
    const PerObjectInfo objectData = objectDatas[inObjectId];
    const BSDFMaterialInfo material = objectData.materialInfoData;

    // Load base color and cut off alpha.
    vec4 baseColor = tex(material.baseColorId, material.baseColorSampler, vsIn.uv0);
    baseColor = baseColor * material.baseColorMul + material.baseColorAdd;

    baseColor.xyz = convertColorSpace(baseColor.xyz);
    if(baseColor.a < material.cutoff)
    {
        discard;
    }

    // Emissive color.
    vec4 emissiveColor = tex(material.emissiveTexId, material.emissiveSampler, vsIn.uv0);
    emissiveColor = emissiveColor * material.emissiveMul + material.emissiveAdd;
    emissiveColor.xyz = convertColorSpace(emissiveColor.xyz);

    // World normal build.
    vec4 normalTex = tex(material.normalTexId, material.normalSampler, vsIn.uv0);

    vec3 vertexWorldPos = vsIn.worldPos;
    vec3 vertexWorldNormal = normalize(vsIn.normal);
    // gl_FrontFacing simulated to support two side face normal.
    {
        // gl_FrontFacing no work here, we custom recompute normal face orient by surface normal.
        vec3 faceNormal = normalize(cross(dFdx(vertexWorldPos), dFdy(vertexWorldPos)));
        if(frameData.renderType == ERendererType_ReflectionCapture)
        {
            // Open GL style.
            if (dot(faceNormal, vertexWorldNormal) < 0.0) vertexWorldNormal *= -1;
        }
        else
        {
            // Vulkan style.
            if (dot(faceNormal, vertexWorldNormal) > 0.0) vertexWorldNormal *= -1;
        }
    }

    vec3 worldNormal;
    {
        const mat3 tbn = mat3(normalize(vsIn.tangent), normalize(vsIn.bitangent), vertexWorldNormal);

        // Remap to [-1, 1].
        vec2 xy = 2.0 * normalTex.rg - 1.0;

        // Construct z.
        float z = sqrt(1.0 - dot(xy, xy));
        worldNormal = normalize(tbn * vec3(xy, z));
    }

    // Specular texture.
    vec4 metalRoughnessTex = tex(material.metalRoughnessTexId, material.metalRoughnessSampler, vsIn.uv0);
    float perceptualRoughness = clamp(metalRoughnessTex.g * material.roughnessMul + material.roughnessAdd, 0.0, 1.0);
    float roughness = perceptualRoughness * perceptualRoughness;

    float metallic  = clamp(metalRoughnessTex.b * material.metalMul     + material.metalAdd,     0.0, 1.0);

    // Occlusion texture.
    vec4 occlusionTex = tex(material.occlusionTexId, material.occlusionSampler, vsIn.uv0);
    float meshAo = clamp(occlusionTex.r, 0.0, 1.0);

    // Scene hdr color. r16g16b16a16 sfloat.
    outHDRSceneColor.rgb = emissiveColor.rgb; // Store emissive color in RGB channel.

    // Store object id.
    outGBufferId.r = packObjectId(inObjectId); 

    // GBufferA: r8g8b8a8 unorm.
    outGBufferA.rgb = baseColor.rgb; // Output base color in GBuffer A rgb channel.
    outGBufferA.a = packShadingModelId(material.shadingModel); // Shading model id.

    // GBuffer B: r10g10b10a2.
    outGBufferB.rgb = packWorldNormal(worldNormal); 

    // GBuffer S: r8g8b8a8 unorm.
    outGBufferS.r = metallic;  // Metalic
    outGBufferS.g = sqrt(geometricAA(worldNormal, roughness)); // Actually it is perceptualRoughness.

    outGBufferS.b = meshAo;    // Mesh Ao
    outGBufferS.a = length(fwidth(vertexWorldNormal)) / length(fwidth(vertexWorldPos)); // cheap mesh curvature.

    // Velocity output.
    outGBufferV = (vsIn.posNDCPrevNoJitter.xy / vsIn.posNDCPrevNoJitter.w) - (vsIn.posNDCCurNoJitter.xy / vsIn.posNDCCurNoJitter.w);

    // Also can do this if jitter:
    // const vec2 cancelJitter = frameData.jitterData.zw - frameData.jitterData.xy;
    // outGBufferV -= cancelJitter;

    // Transform motion vector from NDC space to UV space (+Y is top-down).
    outGBufferV *= vec2(0.5f, -0.5f);
}

#endif //////////////////////////// pixel shader end

#endif // STATIC_MESH_GBUFFER_PASS