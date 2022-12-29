#version 460

/*
** Physical based render code, develop by engineer: qiutanguu.
*/

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

#include "PMX_Common.glsl"

#include "Common.glsl"
#include "ColorSpace.glsl"

layout (set = 0, binding = 0) uniform UniformPMXParam { UniformPMX pmxParam; };
layout (set = 1, binding = 0) uniform UniformView  { ViewData  viewData;  };
layout (set = 2, binding = 0) uniform UniformFrame { FrameData frameData; };

#define COMMON_SAMPLER_SET 3
#include "CommonSamplerSet.glsl"

#define BLUE_NOISE_TEXTURE_SET 4
#define BLUE_NOISE_BUFFER_SET 5
#include "BlueNoiseCommon.glsl"

layout (set = 6, binding = 0) uniform texture2D bindlessTexture2D[];

#ifdef VERTEX_SHADER ///////////// vertex shader start 

layout (location = 0) in vec3 inPosition;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inUV0;
layout (location = 3) in vec3 inPositionLast;

layout(location = 0) out VS2PS vsOut;

void main()
{
    // PMX GL style uv flip.
    vsOut.uv0 = inUV0 * vec2(1.0f, -1.0f); 

    // Local vertex position.
    const vec4 localPosition = vec4(inPosition, 1.0f);
    const vec4 worldPosition = pmxParam.modelMatrix * localPosition;
    vsOut.worldPos = worldPosition.xyz / worldPosition.w;

    // Convert to clip space.
    gl_Position = viewData.camViewProj * worldPosition;

    // Non-uniform scale need normal matrix convert.
    // see http://www.lighthouse3d.com/tutorials/glsl-12-tutorial/the-normal-matrix/.
    const mat3 normalMatrix = transpose(inverse(mat3(pmxParam.modelMatrix)));
    vsOut.normal  = normalize(normalMatrix * normalize(inNormal));

    // Compute velocity for static mesh. https://github.com/GPUOpen-Effects/FidelityFX-FSR2
    // FSR2 will perform better quality upscaling when more objects provide their motion vectors. 
    // It is therefore advised that all opaque, alpha-tested and alpha-blended objects should write their motion vectors for all covered pixels.
    vsOut.posNDCPrevNoJitter = viewData.camViewProjPrevNoJitter * pmxParam.modelMatrixPrev * * vec4(inPositionLast, 1.0);
    vsOut.posNDCCurNoJitter  = viewData.camViewProjNoJitter * worldPosition;
}

#endif /////////////////////////// vertex shader end

#ifdef PIXEL_SHADER ////////////// pixel shader start 

vec4 tex(uint texId, vec2 uv)
{
    // PMX file use linear repeat to sample texture.
    return texture(sampler2D(bindlessTexture2D[nonuniformEXT(texId)], linearRepeatSampler), uv);
}

layout(location = 0) in VS2PS vsIn;

// Scene hdr color. .rgb store emissive color.
layout(location = 0) out vec4 outHDRSceneColor;

layout(location = 1) out float outRectiveMask;

layout(location = 2) out float outCompositionMask;

// GBuffer V: r16g16 sfloat, store velocity.
layout(location = 3) out vec2 outGBufferV;

void main()
{
    // Mask material looks bad in fsr, need improve rective mask weight.
    // Still don't know a better solution, maybe use my own TAA is better.
    // outRectiveMask = 0.75; 

    vec4 baseColor = tex(pmxParam.texID, vsIn.uv0);
    if(baseColor.a < 0.01f)
    {
        // Clip very small mask.
        discard;
    }

    baseColor.rgb = inputColorPrepare(baseColor.rgb);

    const vec3 worldSpaceNormal = normalize(vsIn.normal);
    const vec3 worldPosition = vsIn.worldPos;
    const vec3 worldView = normalize(viewData.camWorldPos.xyz - worldPosition);

    // Directional light evaluate.
    vec3 directionalLit = vec3(0.0);
    {
        const DirectionalLightInfo evaluateLight = frameData.directionalLight;
        vec3 pointToLight = -evaluateLight.direction;

        AngularInfoPMX angularInfo = getAngularInfoPMX(pointToLight, worldSpaceNormal, worldView);

        if (angularInfo.NdotL > 0.0 || angularInfo.NdotV > 0.0)
        {
            directionalLit += angularInfo.NdotL * baseColor.rgb / kPI;
        }
    }


    outHDRSceneColor.rgb = baseColor.rgb;
    outHDRSceneColor.a   = baseColor.a;

    outRectiveMask = max(outHDRSceneColor.r, max(outHDRSceneColor.g, outHDRSceneColor.b)) * baseColor.a;

    // Velocity output.
    outGBufferV = (vsIn.posNDCPrevNoJitter.xy / vsIn.posNDCPrevNoJitter.w) - (vsIn.posNDCCurNoJitter.xy / vsIn.posNDCCurNoJitter.w);

    // Also can do this if jitter:
    // const vec2 cancelJitter = frameData.jitterData.zw - frameData.jitterData.xy;
    // outGBufferV -= cancelJitter;

    // Transform motion vector from NDC space to UV space (+Y is top-down).
    outGBufferV *= vec2(0.5f, -0.5f);
}

#endif //////////////////////////// pixel shader end