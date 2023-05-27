#version 460

#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable

#include "common.glsl"

struct VS2PS
{
    vec2 uv;
    vec3 worldPos;
    vec4 posNDCPrevNoJitter;
    vec4 posNDCCurNoJitter;
};

#ifdef VERTEX_SHADER  ///////////// vertex shader start 

layout(location = 0) in  vec2 i_VertexPos;
layout(location = 0) out VS2PS vsOut;

void main()
{
    const int cbtID = 0;

    const uint nodeID = gl_InstanceIndex;

    cbt_Node node = cbt_DecodeNode(cbtID, nodeID);
    vec4 triangleVertices[3] = DecodeTriangleVertices(node);

    vec2 triangleTexCoords[3] = vec2[3](triangleVertices[0].xy, triangleVertices[1].xy, triangleVertices[2].xy);
    VertexAttribute attrib = TessellateTriangle(triangleTexCoords, i_VertexPos);

    const vec4 worldPos = u_ModelMatrix * attrib.position;
    gl_Position =  frameData.camViewProj * worldPos;


    vsOut.uv = attrib.texCoord;
    vsOut.worldPos = worldPos.xyz;

    // TODO: Terrain also import prev-model matrix.
    vsOut.posNDCPrevNoJitter = frameData.camViewProjPrevNoJitter * dynamicData.u_ModelMatrixPrev * attrib.position;
    vsOut.posNDCCurNoJitter = frameData.camViewProjNoJitter * worldPos;
}

#endif

#ifdef PIXEL_SHADER

layout(location = 0) in VS2PS vsIn;

layout(location = 0) out vec4 outHDRSceneColor; // Scene hdr color: r16g16b16a16. .rgb store emissive color.
layout(location = 1) out vec4 outGBufferA; // GBuffer A: r8g8b8a8 unorm, .rgb store base color, .a is shading model id.
layout(location = 2) out vec3 outGBufferB; // GBuffer B: r16g16b16a16. store worldspace normal.
layout(location = 3) out vec4 outGBufferS; // GBuffer S: r8g8b8a8 unorm, .r is metal, .g is roughness, .b is mesh ao.
layout(location = 4) out vec2 outGBufferV; // GBuffer V: r16g16 sfloat, store velocity.
layout(location = 5) out uint outId; // Id texture, [0 : 14] is sceneNode id, [15 : 15] is selection bit. 

void main()
{
    float filterSize = 1.0f / float(textureSize(inHeightmap, 0).x);

    vec4 terrainMask = texture(sampler2D(texture2DBindlessArray[nonuniformEXT(dynamicData.maskTexId)], linearClampEdgeMipFilterSampler), vsIn.uv, frameData.basicTextureLODBias);

    float sx0 = textureLod(sampler2D(inHeightmap, linearClampEdgeSampler), vsIn.uv - vec2(filterSize, 0.0), 0.0).r;
    float sx1 = textureLod(sampler2D(inHeightmap, linearClampEdgeSampler), vsIn.uv + vec2(filterSize, 0.0), 0.0).r;
    float sy0 = textureLod(sampler2D(inHeightmap, linearClampEdgeSampler), vsIn.uv - vec2(0.0, filterSize), 0.0).r;
    float sy1 = textureLod(sampler2D(inHeightmap, linearClampEdgeSampler), vsIn.uv + vec2(0.0, filterSize), 0.0).r;

    float grassIntensity = terrainMask.r;
    float sandIntensity = terrainMask.g;
    float mudIntenisty = terrainMask.b;
    float cliffIntensity = 1.0 - terrainMask.r - terrainMask.g - terrainMask.b;

    vec3 grassColor = vec3(69,138, 19) / 255.0 * 0.25 * grassIntensity;
    vec3 sandColor = vec3(196, 168, 79) / 255.0 * 0.25 * sandIntensity;
    vec3 mudColor = vec3(150, 140, 80) / 255.0 * 0.25 * mudIntenisty;
    vec3 cliffColor = vec3(109, 107, 80) / 255.0 * 0.25 * cliffIntensity;

    vec3 baseColor = grassColor + sandColor + mudColor + cliffColor;
    vec3 worldNormal;


    worldNormal.x = -u_DmapFactor * (sx1 - sx0);
    worldNormal.z =  u_DmapFactor * (sy1 - sy0); // vulkan Y down.
    worldNormal.y = 2.0f;
    worldNormal = normalize(worldNormal); // = cross(vec3(2.0, a, 0.0), vec3(0.0, b, 2.0))

    outHDRSceneColor = vec4(0.0f, 0.0f, 0.0f, 1.0f);
    outGBufferA.rgb = baseColor;
    outGBufferA.a = kShadingModelStandardPBR;

    outGBufferB.xyz = worldNormal;
    
    outGBufferS.r = 0.0f;
    outGBufferS.g = 0.95f;
    outGBufferS.b = terrainMask.a;
    
    outGBufferV = (vsIn.posNDCPrevNoJitter.xy / vsIn.posNDCPrevNoJitter.w) - (vsIn.posNDCCurNoJitter.xy / vsIn.posNDCCurNoJitter.w);
    outGBufferV *= vec2(0.5f, -0.5f);

    outId = packToIdBuffer(sceneNodeId, bSelected);
    if(bSelected != 0)
    {
        vec3 projPosUnjitter = vsIn.posNDCCurNoJitter.xyz / vsIn.posNDCCurNoJitter.w;

        projPosUnjitter.xy = 0.5 * projPosUnjitter.xy + 0.5;
        projPosUnjitter.y  = 1.0 - projPosUnjitter.y;

        ivec2 storeMaskPos = ivec2(projPosUnjitter.xy * imageSize(outSelectionMask));
        imageStore(outSelectionMask, storeMaskPos, vec4(1.0));
    }
}

#endif