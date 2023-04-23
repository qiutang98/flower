#version 460
#extension GL_GOOGLE_include_directive : enable

#include "common.glsl"

#ifdef VERTEX_SHADER  ///////////// vertex shader start 

layout(location = 0) in  vec2 i_VertexPos;

void main()
{
    const int cbtID = 0;
    const uint nodeID = gl_InstanceIndex;

    cbt_Node node = cbt_DecodeNode(cbtID, nodeID);
    vec4 triangleVertices[3] = DecodeTriangleVertices(node);

    vec2 triangleTexCoords[3] = vec2[3](triangleVertices[0].xy, triangleVertices[1].xy, triangleVertices[2].xy);
    VertexAttribute attrib = TessellateTriangle(triangleTexCoords, i_VertexPos);

    const vec4 worldPos = u_ModelMatrix * attrib.position;
    gl_Position = cascadeInfos[cascadeId].viewProj * worldPos;
}

#endif

#ifdef PIXEL_SHADER

layout(location = 0) out vec4 outColor;

void main()
{
    outColor = vec4(0.0f);
}

#endif