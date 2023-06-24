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

layout(location = 0) out VS2PS vsOut;

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

    // PMX GL style uv flip.
    vsOut.uv0 = inUV0 * vec2(1.0f, -1.0f); 

    // Local vertex position.
    const vec4 localPosition = vec4(inPosition, 1.0f);
    const vec4 worldPosition = pmxParam.modelMatrix * localPosition;
    vsOut.worldPos = worldPosition.xyz / worldPosition.w;

    // Convert to clip space.
    gl_Position = frameData.camViewProj * worldPosition;

    // Non-uniform scale need normal matrix convert.
    // see http://www.lighthouse3d.com/tutorials/glsl-12-tutorial/the-normal-matrix/.
    const mat3 normalMatrix = transpose(inverse(mat3(pmxParam.modelMatrix)));
    vsOut.normal  = normalize(normalMatrix * normalize(inNormal));

    // Compute velocity for static mesh. https://github.com/GPUOpen-Effects/FidelityFX-FSR2
    // FSR2 will perform better quality upscaling when more objects provide their motion vectors. 
    // It is therefore advised that all opaque, alpha-tested and alpha-blended objects should write their motion vectors for all covered pixels.
    vsOut.posNDCPrevNoJitter = frameData.camViewProjPrevNoJitter * pmxParam.modelMatrixPrev * vec4(inPositionLast, 1.0);
    vsOut.posNDCCurNoJitter  = frameData.camViewProjNoJitter * worldPosition;
}

#endif /////////////////////////// vertex shader end

#ifdef PIXEL_SHADER ////////////// pixel shader start 

vec4 tex(uint texId, vec2 uv)
{
    // PMX file use linear repeat to sample texture.
    return texture(sampler2D(texture2DBindlessArray[nonuniformEXT(texId)], linearRepeatMipFilterSampler), uv, frameData.basicTextureLODBias);
}

layout(location = 0) in VS2PS vsIn;

layout(location = 0) out vec4  outHDRSceneColor;
layout(location = 1) out float outRectiveMask;
layout(location = 2) out float outCompositionMask;
layout(location = 3) out uint  outId;
layout(location = 4) out vec2 outGBufferV;

void main()
{
    outId = packToIdBuffer(pmxParam.sceneNodeId, pmxParam.bSelected);

    // Mask material looks bad in fsr, need improve rective mask weight.
    // Still don't know a better solution, maybe use my own TAA is better.
    // outRectiveMask = 0.75; 

    vec4 baseColor = tex(pmxParam.texID, vsIn.uv0);

    const vec3 worldSpaceNormal = normalize(vsIn.normal);
    const vec3 worldPosition = vsIn.worldPos;
    const vec3 worldView = normalize(frameData.camWorldPos.xyz - worldPosition);

    // Directional light evaluate.
    vec3 directionalLit = vec3(0.0);

    if(frameData.skyValid > 0)
    {
        vec3 pointToLight = -frameData.sky.direction;

        AngularInfoPMX angularInfo = getAngularInfoPMX(pointToLight, worldSpaceNormal, worldView);

        if (angularInfo.NdotL > 0.0 || angularInfo.NdotV > 0.0)
        {
            directionalLit += angularInfo.NdotL * baseColor.rgb / kPI;
        }
    }


    outHDRSceneColor.rgb = baseColor.rgb * pmxParam.translucentUnlitScale;
    outHDRSceneColor.a   = baseColor.a;

    // Velocity output.
    outGBufferV = (vsIn.posNDCPrevNoJitter.xy / vsIn.posNDCPrevNoJitter.w) - (vsIn.posNDCCurNoJitter.xy / vsIn.posNDCCurNoJitter.w);

    // Also can do this if jitter:
    // const vec2 cancelJitter = frameData.jitterData.zw - frameData.jitterData.xy;
    // outGBufferV -= cancelJitter;

    // Transform motion vector from NDC space to UV space (+Y is top-down).
    outGBufferV *= vec2(0.5f, -0.5f);

    outRectiveMask = max(outHDRSceneColor.r, max(outHDRSceneColor.g, outHDRSceneColor.b)) * baseColor.a;

    float intervalNoise = interleavedGradientNoise(gl_FragCoord.xy, frameData.frameIndex.x % frameData.jitterPeriod);

    // Z fighting avoid, use pixel depth export, will break out early z function. TODO: Use static macro to change is performance better.
    gl_FragDepth = gl_FragCoord.z + gl_FragCoord.z * pmxParam.pixelDepthOffset * intervalNoise;

    // Select mask, don't need z test.
    if(pmxParam.bSelected != 0)
    {
        vec3 projPosUnjitter = vsIn.posNDCCurNoJitter.xyz / vsIn.posNDCCurNoJitter.w;

        projPosUnjitter.xy = 0.5 * projPosUnjitter.xy + 0.5;
        projPosUnjitter.y  = 1.0 - projPosUnjitter.y;

        ivec2 storeMaskPos = ivec2(projPosUnjitter.xy * imageSize(outSelectionMask));
        imageStore(outSelectionMask, storeMaskPos, vec4(1.0));
    }
}

#endif //////////////////////////// pixel shader end