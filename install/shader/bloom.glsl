#version 460

#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable

#define SHARED_SAMPLER_SET 1
#include "common_shader.glsl"

vec3 prefilter(vec3 c, vec4 prefilterFactor) 
{
    float brightness = max(c.r, max(c.g, c.b));

    float soft = brightness - prefilterFactor.y;

    soft = clamp(soft, 0, prefilterFactor.z);
    soft = soft * soft * prefilterFactor.w;
    
    float contribution = max(soft, brightness - prefilterFactor.x);
    contribution /= max(brightness, 0.00001);

    return c * contribution;
}

#ifdef BLOOM_DOWNSAMPLE_PASS

layout(set = 0, binding = 0) uniform texture2D inputTexture;
layout(set = 0, binding = 1, rgba16f)  uniform image2D hdrDownSample;
layout(set = 0, binding = 2) uniform texture2D inAdaptedLumTex;
layout(set = 0, binding = 3) uniform UniformFrameData { PerFrameData frameData; };

layout (push_constant) uniform PushConsts 
{  
    vec4 prefilterFactor;
    uint mipLevel;
};

// 13 tap downsample kernal.
const uint kDownSampleCount = 13;
const vec2 kDownSampleCoords[kDownSampleCount] = 
{
    { 0.0,  0.0},
	{-1.0, -1.0}, {1.0, -1.0}, {1.0,  1.0}, { -1.0, 1.0},
	{-2.0, -2.0}, {0.0, -2.0}, {2.0, -2.0}, {  2.0, 0.0}, {2.0, 2.0}, {0.0, 2.0}, {-2.0, 2.0}, {-2.0, 0.0}
};

const float kWeights[kDownSampleCount] = 
{
    0.125, 
    0.125, 0.125, 0.125, 0.125, 
    0.03125, 0.0625, 0.03125, 0.0625, 0.03125, 0.0625, 0.03125, 0.0625
};

const int kDownSampleGroupCnt = 5;
const int kSamplePerGroup = 4;
const int kDownSampleGroups[kDownSampleGroupCnt][kSamplePerGroup] = 
{
	{ 1, 2,  3,  4},
    { 5, 6,  0, 12},
    { 6, 7,  8,  0},
    { 0, 8,  9, 10},
    {12, 0, 10, 11}
};
const float kDownSampleGroupWeights[kDownSampleGroupCnt] = 
{
	0.5, 0.125, 0.125, 0.125, 0.125
};

layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    ivec2 downsampleSize = imageSize(hdrDownSample);
    ivec2 workPos = ivec2(gl_GlobalInvocationID.xy);
    if(workPos.x >= downsampleSize.x || workPos.y >= downsampleSize.y)
    {
        return;
    }

    vec2 uv = (vec2(workPos) + vec2(0.5)) / vec2(downsampleSize);
    vec3 outColor = vec3(0.0);

    const bool bFirstDownsample = (mipLevel == 0);

    // Get src texture size and compute it's pixel size.
    uvec2 srcSize = textureSize(inputTexture, 0);
    vec2 pixelSize = 1.0f / vec2(srcSize);

    vec3 samples[kDownSampleCount]; 
    for(uint i = 0; i < kDownSampleCount; i ++)
    {
        vec2 sampleUv = uv + kDownSampleCoords[i] * pixelSize;

        // When downsample, we should not use clamp to edge sampler.
        // Evaluate some bright pixel on the edge, if clamp to edge, down sample level edge pixel will capture it in multi sample.
        // And accumulate all of them then get a bright pixel.
        samples[i] = texture(sampler2D(inputTexture, linearClampBorder0000Sampler), sampleUv).rgb;

        if(bFirstDownsample)
        {
            samples[i] = prefilter(samples[i], prefilterFactor);
        }
    }

    // Downsample
    if(bFirstDownsample)
    {
        float sampleKarisWeight[kDownSampleCount];
        for(uint i = 0; i < kDownSampleCount; i ++)
        {
            sampleKarisWeight[i] = 1.0 / (1.0 + luminance(samples[i]));
        }
        
        for(int i = 0; i < kDownSampleGroupCnt; i++)
        {
			float sumedKarisWeight = 0; 
			for(int j = 0; j < kSamplePerGroup; j++)
            {
				sumedKarisWeight += sampleKarisWeight[kDownSampleGroups[i][j]];
			}

            // Anti AA filter.
			for(int j = 0; j < kSamplePerGroup; j++)
            {
				outColor += kDownSampleGroupWeights[i] * sampleKarisWeight[kDownSampleGroups[i][j]] / sumedKarisWeight * samples[kDownSampleGroups[i][j]];
			}
		}
    }
    else
    {
        for(uint i = 0; i < kDownSampleCount; i ++)
        {
            outColor += samples[i] * kWeights[i];
        }
    }

    imageStore(hdrDownSample, workPos, vec4(outColor, 1.0f));
}

#endif // BLOOM_DOWNSAMPLE_PASS

#ifdef BLOOM_UPSCALE_PASS

layout(set = 0, binding = 0) uniform texture2D inputTexture;
layout(set = 0, binding = 1) uniform texture2D inputCurTexture;
layout(set = 0, binding = 2, rgba16f) uniform image2D hdrUpscale;

layout (push_constant) uniform PushConsts 
{  
    uint  bBlurX;
    uint  bFinalBlur;
    uint  upscaleTime;
    float blurRadius;
};

// TODO: Bake me.
const float kRadius = 16.0;
const float kSigma = 0.2; 
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
    ivec2 upscaleSize = imageSize(hdrUpscale);
    ivec2 workPos = ivec2(gl_GlobalInvocationID.xy);
    if(workPos.x >= upscaleSize.x || workPos.y >= upscaleSize.y)
    {
        return;
    }

    vec2 uv = (vec2(workPos) + vec2(0.5)) / vec2(upscaleSize);

    vec2 inputTexelSize = 1.0 / vec2(upscaleSize);
    
    const bool bBlurXDirection = bBlurX > 0;
    vec2 blurDirection = bBlurXDirection ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    blurDirection *= inputTexelSize;

    vec4 sum = vec4(0.0);
    for(float i = -kRadius; i <= kRadius; i++)
    {
        float weight = gaussianWeight(i / kRadius);
        sum.xyz += weight * texture(sampler2D(inputTexture, pointClampEdgeSampler), uv + i * blurDirection).xyz;
        sum.w += weight;
    }
    sum.xyz /= sum.w;

    if((!bBlurXDirection) && (bFinalBlur == 0))
    {
        vec3 currentColor = texture(sampler2D(inputCurTexture, pointClampEdgeSampler), uv).rgb;
        sum.xyz = mix(currentColor, sum.xyz, vec3(blurRadius));
    }

    imageStore(hdrUpscale, workPos, vec4(sum.xyz, 1.0f));
}

#endif