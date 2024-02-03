#version 460

// draw line for debug purpose.

#include "common_shader.glsl"

layout (set = 0, binding = 0) uniform UniformFrameData  { PerFrameData frameData; };
layout (set = 0, binding = 1) buffer  SSBOLineVertexBuffers  { LineDrawVertex vertices[]; };
layout (set = 0, binding = 2) buffer  SSBODrawCmdBuffer  { uint drawCmd[]; };
layout (set = 0, binding = 3) buffer  SSBODrawCmdCountBuffer  { uint count; };

#ifdef PREPARE_DRAW_CMD_PASS

void main()
{
    drawCmd[0] = count;
    drawCmd[1] = 1;
    drawCmd[2] = 0;
    drawCmd[3] = 0;
}

#endif // PREPARE_DRAW_CMD_PASS

#ifdef DRAW_PASS

#ifdef VERTEX_SHADER ///////////// vertex shader start 

void main()
{
    // Load by index.
    const vec4 inWorldPosition = vec4(vertices[gl_VertexIndex].worldPos, 1.0);
    gl_Position = frameData.camViewProjNoJitter * inWorldPosition;
}

#endif 

#ifdef PIXEL_SHADER

layout(location = 0) out vec4 outColor;

void main()
{
    outColor = vec4(1.0, 0.1, 0.0, 1.0);
}

#endif

#endif // DRAW_PASS

#ifdef WORLD_DRAW_LINE_BY_CPU

#ifdef VERTEX_SHADER

layout(location = 0) in vec4 inWorldPosition;
void main()
{
    gl_Position = frameData.camViewProjNoJitter * vec4(inWorldPosition.xyz, 1.0f);
}

#endif 

#ifdef PIXEL_SHADER

layout(location = 0) out vec4 outColor;
void main()
{
    outColor = vec4(1.0, 0.1, 0.0, 1.0);
}

#endif

#endif