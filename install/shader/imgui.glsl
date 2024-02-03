#version 460

struct VS2PS
{ 
    vec4 color; 
    vec2 uv; 
};

#ifdef VERTEX_SHADER

layout(location = 0) in  vec2  inPosition;
layout(location = 1) in  vec2  inUV;
layout(location = 2) in  vec4  inColor;
layout(location = 0) out VS2PS vsOut;

layout(push_constant) uniform VertexPushConstant 
{ 
    vec2 scale; 
    vec2 translate; 
};

void main()
{
    vsOut.color = inColor;
    vsOut.uv = inUV;

    gl_Position = vec4(inPosition * scale + translate, 0, 1);
}

#endif 

#ifdef PIXEL_SHADER

layout(location = 0) in  VS2PS psIn;
layout(location = 0) out vec4  outColor;

layout(set = 0, binding = 0) uniform sampler2D inTexture;

void main()
{
    outColor = psIn.color * texture(inTexture, psIn.uv);
}

#endif