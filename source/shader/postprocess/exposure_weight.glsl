#version 460

#include "../common/shared_functions.glsl"

layout (set = 0, binding = 0) uniform texture2D exposureColor0;
layout (set = 0, binding = 1) uniform texture2D exposureColor1;
layout (set = 0, binding = 2) uniform texture2D exposureColor2;
layout (set = 0, binding = 3) uniform texture2D exposureColor3;
layout (set = 0, binding = 4) uniform UniformFrameData { PerFrameData frameData; };
layout (set = 0, binding = 5) uniform writeonly image2D weightColor;

#define SHARED_SAMPLER_SET 1
#include "../common/shared_sampler.glsl"

layout(push_constant) uniform PushConstants
{
    float kSigma; // 0.2
    float kWellExposureValue; // 0.5
    float kContrastPow; // 1.0
    float kSaturationPow; // 1.0
    float kExposurePow; // 1.0
};


float saturationWeight(vec3 color)
{
    float mu = (color.x + color.y + color.z) / 3.0;
    vec3 dis = color - vec3(mu);
    float disDot = dot(dis, dis);

    return sqrt(disDot / 3.0);
}

float exposureWeight(vec3 color)
{
    float sigma2 = kSigma * kSigma;

    vec3 colorMu = (color - kWellExposureValue);
    vec3 expC = exp(-0.5 * colorMu * colorMu / sigma2);

    return expC.r * expC.g * expC.b;
}

float computeWeight(texture2D tex, vec2 uv, vec2 texelSize, vec3 param)
{
    vec3 center = texture(sampler2D(tex, pointClampEdgeSampler), uv).xyz; 
    float weight = 1.0;

#if 1
    // contrast, laplace filter.
    if(param.x > 0.0)
    {
        vec3 up     = texture(sampler2D(tex, pointClampEdgeSampler), uv + texelSize * vec2( 0.0,  1.0)).xyz; 
        vec3 down   = texture(sampler2D(tex, pointClampEdgeSampler), uv + texelSize * vec2( 0.0, -1.0)).xyz; 
        vec3 left   = texture(sampler2D(tex, pointClampEdgeSampler), uv + texelSize * vec2(-1.0,  0.0)).xyz; 
        vec3 right  = texture(sampler2D(tex, pointClampEdgeSampler), uv + texelSize * vec2( 1.0,  0.0)).xyz; 

        float centerGray = dot(vec3(0.1,0.7,0.2), center);
        float upGray     = dot(vec3(0.1,0.7,0.2), up);
        float downGray   = dot(vec3(0.1,0.7,0.2), down);
        float leftGray   = dot(vec3(0.1,0.7,0.2), left);
        float rightGray  = dot(vec3(0.1,0.7,0.2), right);

        weight *= pow(abs(-centerGray * 4.0 + upGray + downGray + leftGray + rightGray), param.x);
    } 
#endif
    // saturation
    if(param.y > 0.0)
    {
        weight *= pow(saturationWeight(center), param.y);
    }


    // exposedness
    if(param.z > 0.0)
    {
        weight *= pow(exposureWeight(center), param.z);
    }
    return weight;
}

layout(local_size_x = 8, local_size_y = 8) in;
void main()
{
    ivec2 colorSize = imageSize(weightColor);

    uvec2 groupThreadId = remap8x8(gl_LocalInvocationIndex);
    uvec2 dispatchId = groupThreadId + gl_WorkGroupID.xy * 8;
    ivec2 workPos = ivec2(dispatchId);
    if(workPos.x >= colorSize.x || workPos.y >= colorSize.y)
    {
        return;
    }

    const vec2 texelSize = 1.0 / vec2(colorSize);
    const vec2 uv = (vec2(workPos) + vec2(0.5f)) * texelSize;

    vec3 param = vec3(kContrastPow, kSaturationPow, kExposurePow);

    vec4 weights;
    weights.x = computeWeight(exposureColor0, uv, texelSize, param);
    weights.y = computeWeight(exposureColor1, uv, texelSize, param);
    weights.z = computeWeight(exposureColor2, uv, texelSize, param);
    weights.w = computeWeight(exposureColor3, uv, texelSize, param);

    weights /= dot(weights, vec4(1.0)) + 1e-12f;

    imageStore(weightColor, workPos, weights);
}