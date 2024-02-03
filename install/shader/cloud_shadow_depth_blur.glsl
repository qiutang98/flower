#version 460

#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable

#define SHARED_SAMPLER_SET 1
#include "common_shader.glsl"

layout(set = 0, binding = 0) uniform texture2D inputTexture;
layout(set = 0, binding = 1) uniform writeonly image2D blurImage;

layout (push_constant) uniform PushConsts 
{  
    vec2  kBlurDirection;
    float kRadius;
    float kSigma; 
    float kRadiusLow;
};

float gaussianWeight(float x) 
{
    const float mu = 0; // From center.
    const float dx = x - mu;
    const float sigma2 = kSigma * kSigma;
    return 0.398942280401 / kSigma * exp(- (dx * dx) * 0.5 / sigma2);
}

layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    ivec2 workSize = imageSize(blurImage);
    ivec2 workPos = ivec2(gl_GlobalInvocationID.xy);
    if(workPos.x >= workSize.x || workPos.y >= workSize.y)
    {
        return;
    }

    vec2 uv = (vec2(workPos) + vec2(0.5)) / vec2(workSize);
    vec2 inputTexelSize = 1.0 / vec2(workSize);
    
    vec2 blurDirection = kBlurDirection;
    blurDirection *= inputTexelSize;

    vec2 result;

    {
        float depthSum = 0.0;
        float weightSum = 0.0f;
        for(float i = -kRadius; i <= kRadius; i++)
        {
            float weight = gaussianWeight(i / kRadius);
            depthSum += weight * texture(sampler2D(inputTexture, pointClampEdgeSampler), uv + i * blurDirection).r;
            weightSum += weight;
        }
        depthSum /= weightSum;

        result.x = depthSum;
    }

    {
        vec2 depthSum = vec2(0.0);
        float weightSum = 0.0f;
        for(float i = -kRadiusLow; i <= kRadiusLow; i++)
        {
            float weight = gaussianWeight(i / kRadiusLow);
            depthSum += weight * texture(sampler2D(inputTexture, pointClampEdgeSampler), uv + i * blurDirection).xy;
            weightSum += weight;
        }
        depthSum /= weightSum;

        const bool bFirstPass = kBlurDirection.x > 0.5f;

        result.y = bFirstPass ? depthSum.x : depthSum.y;
    }

    imageStore(blurImage, workPos, vec4(result, 0.0, 0.0));
}