#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable

#define SHARED_SAMPLER_SET    1
#include "common_sampler.glsl"
#include "common_shader.glsl"

layout (push_constant) uniform PushConsts 
{  
    vec2 mixWeight;
};

layout (set = 0, binding = 0) uniform texture2D inCloudShadowDepth;
layout (set = 0, binding = 1) uniform texture2D inCloudShadowDepthHistory;
layout (set = 0, binding = 2) writeonly uniform image2D imageCloudShadowDepth;

layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    ivec2 texSize = imageSize(imageCloudShadowDepth);
    ivec2 workPos = ivec2(gl_GlobalInvocationID.xy);

    if(workPos.x >= texSize.x || workPos.y >= texSize.y)
    {
        return;
    }

    const vec2 uv = (vec2(workPos) + vec2(0.5f)) / vec2(texSize);


    vec2 currentZ = texture(sampler2D(inCloudShadowDepth, linearClampEdgeSampler), uv).xy;

    // TODO: History project.
    vec2 prevZ    = texture(sampler2D(inCloudShadowDepthHistory, linearClampEdgeSampler), uv).xy;

    currentZ = mix(prevZ, currentZ, mixWeight);

	imageStore(imageCloudShadowDepth, workPos, vec4(currentZ, 1.0, 1.0));
}