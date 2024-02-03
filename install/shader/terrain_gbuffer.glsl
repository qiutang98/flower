#version 460
 
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable

#define SHARED_SAMPLER_SET 1
#include "common_shader.glsl"

layout (set = 0, binding = 0) uniform UniformFrameData{ PerFrameData frameData; };
layout (set = 0, binding = 1) buffer SSBOPatchBuffer { TerrainPatch patches[]; } ssboPatchBuffer;
layout (set = 0, binding = 2) uniform texture2D heightmapTexture;
layout (set = 0, binding = 3) uniform texture2D lodTexture;
layout (set = 0, binding = 4) uniform texture2D normalMap;

struct VS2PS
{
    vec2 uv;
    vec3 worldPos;
    vec4 posNDCPrevNoJitter;
    vec4 posNDCCurNoJitter;
};

#ifdef VERTEX_SHADER ///////////// vertex shader start 

layout(location = 0) in  vec2  vsIn;
layout(location = 0) out flat uint lodLevel;
layout(location = 1) out VS2PS vsOut;


void main()
{
    TerrainPatch patchInfo = ssboPatchBuffer.patches[gl_InstanceIndex];

    vec3 localPos = vec3(patchInfo.position.x, 0.0, patchInfo.position.y);

    uint topDiff   = (patchInfo.patchCrossLOD & (0xff <<  0)) >>  0;
    uint downDiff  = (patchInfo.patchCrossLOD & (0xff <<  8)) >>  8;
    uint leftDiff  = (patchInfo.patchCrossLOD & (0xff << 16)) >> 16;
    uint rightDiff = (patchInfo.patchCrossLOD & (0xff << 24)) >> 24;

    const float kQuadSize = 1.0 / 16.0;
    const float kEpsilonQuad = kQuadSize * 0.5;

    vec2 snapPos = vsIn;

    // Top fix.
    if((vsIn.y < kEpsilonQuad) && (topDiff > 0))
    {
        float modSize = exp2(topDiff) * kQuadSize;
        float lessValue = mod(vsIn.x, modSize);
        if(lessValue > kEpsilonQuad)
        {
            snapPos.x = vsIn.x + (modSize - lessValue);
        }
    }

    // Down fix
    if((vsIn.y > 1.0 - kEpsilonQuad) && (downDiff > 0))
    {
        float modSize = exp2(downDiff) * kQuadSize;
        float lessValue = mod(vsIn.x, modSize);
        if(lessValue > kEpsilonQuad)
        {
            snapPos.x = vsIn.x - lessValue;
        }
    }

    // left fix
    if((vsIn.x < kEpsilonQuad) && (leftDiff > 0))
    {
        float modSize = exp2(leftDiff) * kQuadSize;
        float lessValue = mod(vsIn.y, modSize);
        if(lessValue > kEpsilonQuad)
        {
            snapPos.y = vsIn.y + (modSize - lessValue);
        }
    }

    // right fix
    if((vsIn.x > 1.0 - kEpsilonQuad) && (rightDiff > 0))
    {
        float modSize = exp2(rightDiff) * kQuadSize;
        float lessValue = mod(vsIn.y, modSize);
        if(lessValue > kEpsilonQuad)
        {
            snapPos.y = vsIn.y - lessValue;
        }
    }

    lodLevel = patchInfo.lod;

    float tileDim = getTerrainLODSizeFromLOD(patchInfo.lod);
    float patchDim = tileDim / 8.0; // Meter.
    localPos += vec3(snapPos.x, 0.0, snapPos.y) * patchDim;

    vec2 localUv = vec2(localPos.x - frameData.landscape.offsetX, localPos.z - frameData.landscape.offsetY) / vec2(frameData.landscape.terrainDimension);
    float heightmapValue = texture(sampler2D(heightmapTexture, linearClampEdgeSampler), localUv).x;

    localPos.y = mix(frameData.landscape.minHeight, frameData.landscape.maxHeight, heightmapValue);

    vec4 worldPos = vec4(localPos, 1.0);
    gl_Position   =  frameData.camViewProj * worldPos;

    vsOut.worldPos = worldPos.xyz / worldPos.w;
    vsOut.uv = localUv;
    vsOut.posNDCPrevNoJitter = frameData.camViewProjPrevNoJitter * vec4(localPos, 1.0);
    vsOut.posNDCCurNoJitter = frameData.camViewProjNoJitter * worldPos;
}

#endif /////////////////////////// vertex shader end

#ifdef PIXEL_SHADER ////////////// pixel shader start

layout(location = 0) in flat uint lodLevel;
layout(location = 1) in VS2PS vsIn;

layout(location = 0) out vec4 outHDRSceneColor; // Scene hdr color: r16g16b16a16. .rgb store emissive color.
layout(location = 1) out vec4 outGBufferA; // GBuffer A: r8g8b8a8 unorm, .rgb store base color, .a is shading model id.
layout(location = 2) out vec4 outGBufferB; // GBuffer B: r10g10b10a2. rgb store worldspace normal.
layout(location = 3) out vec4 outGBufferS; // GBuffer S: r8g8b8a8 unorm, .r is metal, .g is roughness, .b is mesh ao.
layout(location = 4) out vec2 outGBufferV; // GBuffer V: r16g16 sfloat, store velocity.
layout(location = 5) out float outGBufferId;

const vec3 kDebugColor[9] = 
{
    vec3(0.0, 0.0, 0.0),
    vec3(1.0, 0.0, 0.0),
    vec3(0.5, 0.5, 0.0),
    vec3(0.2, 0.8, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.8, 0.2),
    vec3(0.0, 0.5, 0.5),
    vec3(0.0, 0.2, 0.8),
    vec3(0.0, 0.0, 1.0),
};

void main()
{
    vec2 uv = vsIn.uv;

    const vec2 filterSize = 1.0f / vec2(textureSize(heightmapTexture, 0));
    const float dumpHeight = frameData.landscape.maxHeight - frameData.landscape.minHeight;

    vec3 worldNormal;

#if   0
    worldNormal = unpackWorldNormal(texture(sampler2D(normalMap, linearClampEdgeSampler), uv).xyz);
#elif 0
    {

        vec3 a = vec3(-filterSize.x, 0, 0);
        vec3 b = vec3( filterSize.x, 0, 0);
        vec3 c = vec3(0, 0,  filterSize.y);
        vec3 d = vec3(0, 0, -filterSize.y);

        a.y = textureLod(sampler2D(heightmapTexture, linearClampEdgeSampler), vsIn.uv + a.xz, 0.0).r; // 1
        b.y = textureLod(sampler2D(heightmapTexture, linearClampEdgeSampler), vsIn.uv + b.xz, 0.0).r; // 2
        c.y = textureLod(sampler2D(heightmapTexture, linearClampEdgeSampler), vsIn.uv + c.xz, 0.0).r; // 0
        d.y = textureLod(sampler2D(heightmapTexture, linearClampEdgeSampler), vsIn.uv + d.xz, 0.0).r; // 3

        worldNormal = -normalize(cross(b - a, c - d));
    }
#else
    {
        float h[4];

        h[0] = textureLod(sampler2D(heightmapTexture, linearClampEdgeSampler), vsIn.uv - vec2(0.0, filterSize.y), 0.0).r; // 1
        h[1] = textureLod(sampler2D(heightmapTexture, linearClampEdgeSampler), vsIn.uv - vec2(filterSize.x, 0.0), 0.0).r; // 2
        h[2] = textureLod(sampler2D(heightmapTexture, linearClampEdgeSampler), vsIn.uv + vec2(filterSize.x, 0.0), 0.0).r; // 0
        h[3] = textureLod(sampler2D(heightmapTexture, linearClampEdgeSampler), vsIn.uv + vec2(0.0, filterSize.y), 0.0).r; // 3

        worldNormal.z =  dumpHeight * (h[0] - h[3]);
        worldNormal.x =  dumpHeight * (h[1] - h[2]); // vulkan Y down.
        worldNormal.y =  2.0f;

        worldNormal = normalize(worldNormal); // = cross(vec3(2.0, a, 0.0), vec3(0.0, b, 2.0))
    }
#endif 



    outHDRSceneColor = vec4(vec3(0.0, 0.0, 0.0), 0.0);
    outGBufferB.rgb = packWorldNormal(worldNormal); 
    outGBufferS = vec4(0.0, 1.0, 1.0, 0.0);

    outGBufferA = vec4(vec3(0.50), packShadingModelId(EShadingModelType_DefaultLit));

    // Get object id.
    outGBufferId = packObjectId(frameData.landscape.terrainObjectId); 

    // Velocity output.
    outGBufferV = (vsIn.posNDCPrevNoJitter.xy / vsIn.posNDCPrevNoJitter.w) - (vsIn.posNDCCurNoJitter.xy / vsIn.posNDCCurNoJitter.w);

    // Also can do this if jitter:
    // const vec2 cancelJitter = frameData.jitterData.zw - frameData.jitterData.xy;
    // outGBufferV -= cancelJitter;

    // Transform motion vector from NDC space to UV space (+Y is top-down).
    outGBufferV *= vec2(0.5f, -0.5f);
}

#endif //////////////////////////// pixel shader end