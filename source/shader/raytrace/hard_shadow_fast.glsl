#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_ray_query : enable

#include "../common/shared_functions.glsl"

layout (set = 0, binding = 0) uniform writeonly image2D rayShadowMask;
layout (set = 0, binding = 1) uniform UniformFrameData { PerFrameData frameData; };
layout (set = 0, binding = 2) uniform accelerationStructureEXT topLevelAS;
layout (set = 0, binding = 3)  uniform texture2D inDepth;

#define SHARED_SAMPLER_SET 1
#include "../common/shared_sampler.glsl"

#define BLUE_NOISE_BUFFER_SET 2
#include "../common/shared_bluenoise.glsl"

// fast but no sample mask.
layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    ivec2 colorSize = imageSize(rayShadowMask);

    uvec2 groupThreadId = remap8x8(gl_LocalInvocationIndex);
    uvec2 dispatchId = groupThreadId + gl_WorkGroupID.xy * 8;
    ivec2 workPos = ivec2(dispatchId);
    if(workPos.x >= colorSize.x || workPos.y >= colorSize.y)
    {
        return;
    }
    const vec2 uv = (vec2(workPos) + vec2(0.5f)) / vec2(colorSize);

    float shadow = 1.0f;
    const float deviceZ = texelFetch(inDepth, workPos, 0).r;
    if(deviceZ > 0.0f)
    {
        vec3 worldPos = getWorldPos(uv, deviceZ, frameData);

        rayQueryEXT rayQuery;
        rayQueryInitializeEXT(rayQuery, topLevelAS, gl_RayFlagsTerminateOnFirstHitEXT, 0xFF, worldPos, 0.01, -normalize(frameData.sky.direction), 1000.0);

        // Traverse the acceleration structure and store information about the first intersection (if any)
        rayQueryProceedEXT(rayQuery);

        // If the intersection has hit a triangle, the fragment is shadowed
        if (rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCommittedIntersectionTriangleEXT) 
        {
            shadow *= 0.1;
        }
    }

    // Final store.
    imageStore(rayShadowMask, workPos, vec4(shadow));
}