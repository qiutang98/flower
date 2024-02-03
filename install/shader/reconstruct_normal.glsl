#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_EXT_samplerless_texture_functions : enable

#define SHARED_SAMPLER_SET    1
#include "common_shader.glsl"

layout (set = 0, binding = 0) uniform texture2D inDepth;
layout (set = 0, binding = 1, rgb10_a2) uniform image2D imageWorldNormal;
layout (set = 0, binding = 2) uniform UniformFrameData { PerFrameData frameData; };

layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    ivec2 workSize = imageSize(imageWorldNormal);
    uvec2 groupThreadId = remap8x8(gl_LocalInvocationIndex);
    uvec2 dispatchId = groupThreadId + gl_WorkGroupID.xy * 8;
    ivec2 workPos = ivec2(dispatchId);

    if(workPos.x >= workSize.x || workPos.y >= workSize.y)
    {
        return;
    }

    const vec2 texelSize = 1.0f / vec2(workSize);
    const vec2 uv = (vec2(workPos) + vec2(0.5f)) * texelSize;

    vec3 normalResult;
    const float depth = texture(sampler2D(inDepth, pointClampEdgeSampler), uv).r;

    if(depth <= 0.0)
    {
        imageStore(imageWorldNormal, ivec2(workPos), vec4(0.0));
        return;
    }


    vec2 delta = texelSize * 2.0f;

    vec3 l1 = vec3(uv + vec2(-1,  0) * delta, 0.0);
    vec3 l2 = vec3(uv + vec2(-2,  0) * delta, 0.0);
    vec3 r1 = vec3(uv + vec2( 1,  0) * delta, 0.0);
    vec3 r2 = vec3(uv + vec2( 2,  0) * delta, 0.0);
    vec3 u1 = vec3(uv + vec2( 0,  1) * delta, 0.0);
    vec3 u2 = vec3(uv + vec2( 0,  2) * delta, 0.0);
    vec3 d1 = vec3(uv + vec2( 0, -1) * delta, 0.0);
    vec3 d2 = vec3(uv + vec2( 0, -2) * delta, 0.0);

    float l1d = texture(sampler2D(inDepth, pointClampEdgeSampler), l1.xy).r; l1.z = linearizeDepth(l1d, frameData);
    float l2d = texture(sampler2D(inDepth, pointClampEdgeSampler), l2.xy).r; l2.z = linearizeDepth(l2d, frameData);
    float r1d = texture(sampler2D(inDepth, pointClampEdgeSampler), r1.xy).r; r1.z = linearizeDepth(r1d, frameData);
    float r2d = texture(sampler2D(inDepth, pointClampEdgeSampler), r2.xy).r; r2.z = linearizeDepth(r2d, frameData);
    float u1d = texture(sampler2D(inDepth, pointClampEdgeSampler), u1.xy).r; u1.z = linearizeDepth(u1d, frameData);
    float u2d = texture(sampler2D(inDepth, pointClampEdgeSampler), u2.xy).r; u2.z = linearizeDepth(u2d, frameData);
    float d1d = texture(sampler2D(inDepth, pointClampEdgeSampler), d1.xy).r; d1.z = linearizeDepth(d1d, frameData);
    float d2d = texture(sampler2D(inDepth, pointClampEdgeSampler), d2.xy).r; d2.z = linearizeDepth(d2d, frameData);

    float lZ = linearizeDepth(depth, frameData);

    const uint closestHorizontal = abs((2.0 * l1.z - l2.z) - lZ) < abs((2.0 * r1.z - r2.z) - lZ) ? 0 : 1;
    const uint closestVertical   = abs((2.0 * d1.z - d2.z) - lZ) < abs((2.0 * u1.z - u2.z) - lZ) ? 0 : 1;

    vec3 p1;
    vec3 p2;

    if(closestVertical == 0)
    {
        p1 = closestHorizontal == 0 ? vec3(l1.xy, l1d) : vec3(d1.xy, d1d);
        p2 = closestHorizontal == 0 ? vec3(d1.xy, d1d) : vec3(r1.xy, r1d);
    }
    else
    {
        p1 = closestHorizontal == 0 ? vec3(u1.xy, u1d) : vec3(r1.xy, r1d);
        p2 = closestHorizontal == 0 ? vec3(l1.xy, l1d) : vec3(u1.xy, u1d);
    }


    p1 = getWorldPos(p1.xy, p1.z, frameData);
    p2 = getWorldPos(p2.xy, p2.z, frameData);

    vec3 p0 = getWorldPos(uv, depth, frameData);
    normalResult = normalize(cross(p2 - p0, p1 - p0)) * 0.5 + 0.5;

    imageStore(imageWorldNormal, ivec2(workPos), vec4(normalResult, 0.0));
}

