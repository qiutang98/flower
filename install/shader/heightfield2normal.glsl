#version 460
#include "common_shader.glsl"

layout(set = 0, binding = 0) uniform texture2D textureHeightmap;
layout(set = 0, binding = 1) uniform writeonly image2D imageNormal;

vec3 getPosition(ivec2 coord, vec2 uv)
{
    float r = texelFetch(textureHeightmap, coord, 0).x;
    return vec3(uv.x, r, uv.y);
}

vec3 getNormal(vec3 v1, vec3 v2)
{
    return -normalize(cross(v1, v2));
}

// Compute normal from height field.
layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    uvec2 groupThreadId = remap8x8(gl_LocalInvocationIndex);
    uvec2 dispatchId = groupThreadId + gl_WorkGroupID.xy * 8;
    ivec2 workPos = ivec2(dispatchId);

    ivec2 workSize = imageSize(imageNormal);
    if(workPos.x >= workSize.x || workPos.y >= workSize.y)
    {
        return;
    }

    const vec2 texelSize = 1.0 / vec2(workSize);
    vec2 uv = (vec2(workPos) + 0.5) * texelSize;

    vec4 result;
    {
        vec3 c0   = getPosition(workPos,                                          uv);
        vec3 c10  = getPosition(workPos + ivec2( 1, 0), uv + vec2( 1, 0) * texelSize);
        vec3 c01  = getPosition(workPos + ivec2( 0, 1), uv + vec2( 0, 1) * texelSize);
        vec3 c_10 = getPosition(workPos + ivec2(-1, 0), uv + vec2(-1, 0) * texelSize);
        vec3 c_01 = getPosition(workPos + ivec2( 0,-1), uv + vec2( 0,-1) * texelSize);

        vec3 v1 = c10 - c0;
        vec3 v2 = c01 - c0;
        vec3 v3 = c_10 - c0;
        vec3 v4 = c_01 - c0;

        vec3 normal = getNormal(v1, v2) +  getNormal(v2, v3) +  getNormal(v3, v4) +  getNormal(v4, v1);
        normal = normalize(normal);

        result.xyz = packWorldNormal(normal);
    }

    imageStore(imageNormal, workPos, result);
}