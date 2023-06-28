#version 460
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable

struct VS2PS
{
    vec2 uv0;
    vec3 normal;
    vec3 worldPos;
    vec4 posNDCPrevNoJitter;
    vec4 posNDCCurNoJitter;
};

#define PMX_COMMON_SET
#include "pmx_common.glsl"

#ifdef VERTEX_SHADER ///////////// vertex shader start 


layout (location = 0) out VS2PS vsOut;



void main()
{
    const uint indicesId  = pmxParam.indicesArrayId;
    const uint positionId = pmxParam.positionsArrayId;
    const uint positionsPrevId = pmxParam.positionsPrevArrayId;
    const uint normalId = pmxParam.normalsArrayId;
    const uint uv0Id = pmxParam.uv0sArrayId;

    const uint indexId = gl_VertexIndex;
    const uint vertexId = indicesArray[nonuniformEXT(indicesId)].data[indexId];

    vec3 inPosition;
    vec3 inNormal;
    vec2 inUV0;
    vec3 inPositionLast;

    inPosition.x = verticesArray[nonuniformEXT(positionId)].data[vertexId * 3 + 0];
    inPosition.y = verticesArray[nonuniformEXT(positionId)].data[vertexId * 3 + 1];
    inPosition.z = verticesArray[nonuniformEXT(positionId)].data[vertexId * 3 + 2];

    inPositionLast.x = verticesArray[nonuniformEXT(positionsPrevId)].data[vertexId * 3 + 0];
    inPositionLast.y = verticesArray[nonuniformEXT(positionsPrevId)].data[vertexId * 3 + 1];
    inPositionLast.z = verticesArray[nonuniformEXT(positionsPrevId)].data[vertexId * 3 + 2];

    inNormal.x = verticesArray[nonuniformEXT(normalId)].data[vertexId * 3 + 0];
    inNormal.y = verticesArray[nonuniformEXT(normalId)].data[vertexId * 3 + 1];
    inNormal.z = verticesArray[nonuniformEXT(normalId)].data[vertexId * 3 + 2];

    inUV0.x = verticesArray[nonuniformEXT(uv0Id)].data[vertexId * 2 + 0];
    inUV0.y = verticesArray[nonuniformEXT(uv0Id)].data[vertexId * 2 + 1];

    float outlineWidth = 0.0025f;

 
    // PMX GL style uv flip.
    vsOut.uv0 = inUV0 * vec2(1.0f, -1.0f); 

    // Non-uniform scale need normal matrix convert.
    // see http://www.lighthouse3d.com/tutorials/glsl-12-tutorial/the-normal-matrix/.
    const mat3 normalMatrix = transpose(inverse(mat3(pmxParam.modelMatrix)));
    vec3 worldNormal = normalize(normalMatrix * normalize(inNormal));
    vsOut.normal  = worldNormal;



    // Local vertex position.
    const vec4 localPosition = vec4(inPosition, 1.0f);
    const vec4 worldPosition = pmxParam.modelMatrix * localPosition;
    vsOut.worldPos = worldPosition.xyz / worldPosition.w;

    {
        vec4 clipPos = frameData.camViewProj * worldPosition;
        vec3 clipNormal = normalize(mat3(frameData.camViewProj) * worldNormal);
        clipNormal.x /= frameData.camInfo.y;
        clipPos.xy +=  clipNormal.xy * (outlineWidth * clipPos.w);

        // Convert to clip space.
        gl_Position = clipPos;
    }

    {
        vec4 clipPosNoJitter = frameData.camViewProjNoJitter * worldPosition;
        vec3 clipNormalNoJitter = normalize(mat3(frameData.camViewProjPrevNoJitter) * worldNormal);
        clipNormalNoJitter.x /= frameData.camInfo.y;
        clipPosNoJitter.xy +=  clipNormalNoJitter.xy * (outlineWidth * clipPosNoJitter.w);

        vsOut.posNDCCurNoJitter  = clipPosNoJitter;
    }

    {
        vec4 worldPositionPrev =  pmxParam.modelMatrixPrev * vec4(inPositionLast, 1.0);
        vec4 clipPosPrevNoJitter = frameData.camViewProjPrevNoJitter * worldPositionPrev;

        const mat3 normalMatrixPrev = transpose(inverse(mat3(pmxParam.modelMatrixPrev)));
        vec3 worldNormalPrev = normalize(normalMatrixPrev * normalize(inNormal));
        vec3 clipNormaPrevlNoJitter = normalize(mat3(frameData.camViewProjPrevNoJitter) * worldNormalPrev);
        clipNormaPrevlNoJitter.x /= frameData.camInfo.y;
        clipPosPrevNoJitter.xy +=  clipNormaPrevlNoJitter.xy * (outlineWidth * clipPosPrevNoJitter.w);

        // Compute velocity for static mesh. https://github.com/GPUOpen-Effects/FidelityFX-FSR2
        // FSR2 will perform better quality upscaling when more objects provide their motion vectors. 
        // It is therefore advised that all opaque, alpha-tested and alpha-blended objects should write their motion vectors for all covered pixels.
        vsOut.posNDCPrevNoJitter = clipPosPrevNoJitter;
    }
}

#endif /////////////////////////// vertex shader end

#ifdef PIXEL_SHADER ////////////// pixel shader start 

vec4 tex(uint texId, vec2 uv)
{
    // PMX file use linear repeat to sample texture.
    return texture(sampler2D(texture2DBindlessArray[nonuniformEXT(texId)], linearRepeatMipFilterSampler), uv, frameData.basicTextureLODBias);
}

layout(location = 0) in VS2PS vsIn;

layout(location = 0) out vec4 outHDRSceneColor;
layout(location = 1) out vec2 outGBufferV;

void main()
{
    vec4 baseColor = tex(pmxParam.texID, vsIn.uv0);
    if(baseColor.a < 0.01f)
    {
        // Clip very small mask.
        discard;
    }

    const vec3 worldNormal = normalize(vsIn.normal);
    const vec3 worldPosition = vsIn.worldPos;
    const vec3 worldView = normalize(frameData.camWorldPos.xyz - worldPosition);

    bool bSSS = isInShadingModelRange(pmxParam.shadingModelId, kShadingModelSSSS);

    // Outline color.
    float powFactor = bSSS ? 2.0 : 4.0;
    float scaleFator = bSSS ? 0.05 : 0.01;
    outHDRSceneColor.xyz = pow(baseColor.xyz, vec3(powFactor)) * scaleFator;
    outHDRSceneColor.w = 0.0;

    // Velocity output.
    outGBufferV = (vsIn.posNDCPrevNoJitter.xy / vsIn.posNDCPrevNoJitter.w) - (vsIn.posNDCCurNoJitter.xy / vsIn.posNDCCurNoJitter.w);

    // Also can do this if jitter:
    // const vec2 cancelJitter = frameData.jitterData.zw - frameData.jitterData.xy;
    // outGBufferV -= cancelJitter;

    // Transform motion vector from NDC space to UV space (+Y is top-down).
    outGBufferV *= vec2(0.5f, -0.5f);
}

#endif //////////////////////////// pixel shader end