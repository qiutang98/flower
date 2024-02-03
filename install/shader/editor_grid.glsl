#version 460

#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable

#define SHARED_SAMPLER_SET 1
#include "common_shader.glsl"

vec3 kNdcPoints[6] = vec3[](
    vec3( 1,  1, 0), 
    vec3(-1, -1, 0), 
    vec3(-1,  1, 0),
    vec3(-1, -1, 0), 
    vec3( 1,  1, 0), 
    vec3( 1, -1, 0)
);

layout (set = 0, binding = 0) uniform UniformFrameData { PerFrameData frameData; };
layout (set = 0, binding = 1) uniform  texture2D inDepth;

#ifdef VERTEX_SHADER ///////////// vertex shader start 

layout(location = 0) out vec3 nearPoint; 
layout(location = 1) out vec3 farPoint; 

vec3 deprojectNDC2World(vec2 pos, float z) 
{
    vec4 worldH = frameData.camInvertViewProjNoJitter * vec4(pos, z, 1.0);
    return worldH.xyz / worldH.w;
}

void main()
{
    nearPoint = deprojectNDC2World(kNdcPoints[gl_VertexIndex].xy, 1.0);
    farPoint  = deprojectNDC2World(kNdcPoints[gl_VertexIndex].xy, 0.0);

    gl_Position = vec4(kNdcPoints[gl_VertexIndex].xyz, 1.0);
}

#endif /////////////////////////// vertex shader end

#ifdef PIXEL_SHADER ////////////// pixel shader start 

layout(location = 0) in vec3 nearPoint; 
layout(location = 1) in vec3 farPoint; 

layout(location = 0) out vec4 outColor;

vec4 grid(vec3 fragPos3D) 
{
    float gray0 = 0.30;
    float scale0 = 1.0;

    float gray1 = 0.60;
    float scale1 = 0.1;

    vec2 coord0 = fragPos3D.xz * scale0; 
    vec2 derivative0 = fwidth(coord0);
    vec2 grid0 = abs(fract(coord0 - 0.5) - 0.5) / fwidth(coord0 * 2.0);
    float line0 = min(grid0.x, grid0.y);
    vec4 color = vec4(gray0, gray0, gray0, 1.0 - min(line0, 1.0));

    vec2 coord1 = fragPos3D.xz * scale1; 
    vec2 grid1 = abs(fract(coord1 - 0.5) - 0.5) / fwidth(coord1 * 2.0);
    float line1 = min(grid1.x, grid1.y);
    float a1 = 1.0 - min(line1, 1.0);
    if(a1 > 0.0f)
    {
        color = mix(color, vec4(gray1, gray1, gray1, a1), a1);
    }

    vec2 coord = fragPos3D.xz;
    bool xDraw = onRange(coord.x, -0.5f, 0.5f);
    bool yDraw = onRange(coord.y, -0.5f, 0.5f);
    if(xDraw || yDraw)
    {
        vec2 grid = abs(fract(coord - 0.5) - 0.5) / fwidth(coord * 2.0);
        vec2 a = vec2(1.0) - min(grid, vec2(1.0));

        if(xDraw)
        {
            color.xyz = mix(color.xyz, vec3(1.0f, 0.12f, 0.18f), a.x);
        }
        if(yDraw)
        {
            color.xyz = mix(color.xyz, vec3(0.1f, 0.3f, 1.0f), a.y);
        }
    }

    return color;
}

float computeDepth(vec3 pos) 
{
    vec4 posH = frameData.camViewProjNoJitter * vec4(pos.xyz, 1.0);
    return posH.z / posH.w;
}

vec4 getColor(vec3 fragPos3D, float t)
{
    float deviceZ = computeDepth(fragPos3D);

    vec2 uv = vec2(gl_FragCoord.xy) / vec2(frameData.postWidth, frameData.postHeight);

    float sceneZ = texture(sampler2D(inDepth, pointClampEdgeSampler), uv).r;

    float linearDepth = linearizeDepth(deviceZ, frameData);
    float fading = exp2(-linearDepth * 0.05);

    vec4 result = grid(fragPos3D) * float(t > 0);
    result.a = (deviceZ > sceneZ) ? result.a : 0.0;
    result.a *= fading * 0.75;

    return result;
}

void main()
{
    float t = -nearPoint.y / (farPoint.y - nearPoint.y);
    vec3 fragPos3D = nearPoint + t * (farPoint - nearPoint);
    outColor = getColor(fragPos3D, t);
}

#endif //////////////////////////// pixel shader end