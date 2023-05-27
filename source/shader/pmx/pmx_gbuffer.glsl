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

layout (location = 0) in vec3 inPosition;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inUV0;
layout (location = 3) in vec3 inPositionLast;

layout (location = 0) out VS2PS vsOut;

void main()
{
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
    return texture(sampler2D(texture2DBindlessArray[nonuniformEXT(texId)], linearRepeatSampler), uv, frameData.basicTextureLODBias);
}

layout(location = 0) in VS2PS vsIn;

// Scene hdr color. .rgb store emissive color.
layout(location = 0) out vec4 outHDRSceneColor;
layout(location = 1) out vec4 outGBufferA;
layout(location = 2) out vec4 outGBufferB;
layout(location = 3) out vec4 outGBufferS;
layout(location = 4) out vec2 outGBufferV;
layout(location = 5) out uint outId;
layout(location = 6) out float outRectiveMask;
layout(location = 7) out float outCompositionMask;

void main()
{
    outId = packToIdBuffer(pmxParam.sceneNodeId, pmxParam.bSelected);

    vec4 baseColor = tex(pmxParam.texID, vsIn.uv0);
    if(baseColor.a < 0.01f)
    {
        // Clip very small mask.
        discard;
    }

    const vec3 worldNormal = normalize(vsIn.normal);
    const vec3 worldPosition = vsIn.worldPos;
    const vec3 worldView = normalize(frameData.camWorldPos.xyz - worldPosition);

    outGBufferA.rgb = baseColor.rgb;
    outGBufferA.a   = pmxParam.shadingModelId;

    outGBufferB.rgb = worldNormal;

    outGBufferS.r = 0.0f; // metal
    outGBufferS.g = 0.6f; // roughness
    outGBufferS.b = 1.0f; // ao

    outHDRSceneColor = vec4(0.0, 0.0, 0.0, 1.0);

    // Velocity output.
    outGBufferV = (vsIn.posNDCPrevNoJitter.xy / vsIn.posNDCPrevNoJitter.w) - (vsIn.posNDCCurNoJitter.xy / vsIn.posNDCCurNoJitter.w);

    // Also can do this if jitter:
    // const vec2 cancelJitter = frameData.jitterData.zw - frameData.jitterData.xy;
    // outGBufferV -= cancelJitter;

    // Transform motion vector from NDC space to UV space (+Y is top-down).
    outGBufferV *= vec2(0.5f, -0.5f);


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