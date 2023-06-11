#version 460

layout(set = 0, binding = 0) uniform texture2D inA;
layout(set = 0, binding = 1) uniform writeonly image2D outA;

#define SHARED_SAMPLER_SET 1
#include "../common/shared_sampler.glsl"

#include "post_common.glsl"

layout (push_constant) uniform PushConsts 
{  
    vec2  kDirection;
};

#define kSigma 7.0
#define kFusionRadius 8.0

float gaussianWeight(float x) 
{
    const float mu = 0; // From center.
    const float dx = x - mu;
    const float sigma2 = kSigma * kSigma;
    return 0.398942280401 / kSigma * exp(- (dx * dx) * 0.5 / sigma2);
}


layout(local_size_x = 8, local_size_y = 8) in;
void main()
{
    ivec2 colorSize = imageSize(outA);

    uvec2 groupThreadId = remap8x8(gl_LocalInvocationIndex);
    uvec2 dispatchId = groupThreadId + gl_WorkGroupID.xy * 8;
    ivec2 workPos = ivec2(dispatchId);
    if(workPos.x >= colorSize.x || workPos.y >= colorSize.y)
    {
        return;
    }
    const vec2 texelSize = 1.0 / vec2(colorSize);
    const vec2 uv = (vec2(workPos) + vec2(0.5f)) * texelSize;

    vec4 sum = vec4(0.0);
    float weight = 0.0;

#if 0
    int index = 0;
    for(float i = -kFusionGaussianTapRadius; i <= kFusionGaussianTapRadius; i++)
    {
        float sampleWeight = kFusionGaussianWeight[index];
        sum += sampleWeight * texture(sampler2D(inA, pointClampEdgeSampler), uv + i * kDirection * texelSize);

        weight += sampleWeight;
        index ++;
    }
#else
    for(float i = -kFusionRadius; i <= kFusionRadius; i++)
    {
        float sampleWeight = gaussianWeight(i);
        sum += sampleWeight * texture(sampler2D(inA, pointClampEdgeSampler), uv + i * kDirection * texelSize);

        weight += sampleWeight;
    }
#endif
    sum /= weight;

    imageStore(outA, workPos, sum);
}