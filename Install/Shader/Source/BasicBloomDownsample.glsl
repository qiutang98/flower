#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable

layout(set = 0, binding = 0) uniform texture2D inputTexture;
layout(set = 0, binding = 1, rgba16f)  uniform image2D hdrDownSample;
layout(set = 0, binding = 2) uniform texture2D inAdaptedLumTex;

#define COMMON_SAMPLER_SET 1
#include "CommonSamplerSet.glsl"

#include "Common.glsl"

layout (set = 2, binding = 0) uniform UniformView  { ViewData  viewData;  };
layout (set = 3, binding = 0) uniform UniformFrame { FrameData frameData; };


layout (push_constant) uniform PushConsts 
{  
    vec4 prefilterFactor;
    uint mipLevel;
};

// 13 tap downsample kernal.
const uint kDownSampleCount = 13;
const vec2 kDownSampleCoords[kDownSampleCount] = 
{
    {0.0,0.0},
	{-1.0,-1.0},{1.0,-1.0},{1.0,1.0},{-1.0,1.0},
	{-2.0,-2.0},{0.0,-2.0},{2.0,-2.0},{2.0,0.0},{2.0,2.0},{0.0,2.0},{-2.0,2.0},{-2.0,0.0}
};
const float kWeights[kDownSampleCount] = 
{
    0.125, 0.125, 0.125, 0.125, 0.125, 0.03125, 0.0625,
	0.03125, 0.0625, 0.03125, 0.0625, 0.03125, 0.0625
};

const int kDownSampleGroupCnt = 5;
const int kSamplePerGroup = 4;
const int kDownSampleGroups[kDownSampleGroupCnt][kSamplePerGroup] = 
{
	{1,2,3,4},
    {5,6,0,12},
    {6,7,8,0},
    {0,8,9,10},
    {12,0,10,11}
};
const float kDownSampleGroupWeights[kDownSampleGroupCnt] = 
{
	0.5,0.125,0.125,0.125,0.125
};

#include "BasicBloomCommon.glsl"

layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    ivec2 downsampleSize = imageSize(hdrDownSample);
    ivec2 workPos = ivec2(gl_GlobalInvocationID.xy);
    if(workPos.x >= downsampleSize.x || workPos.y >= downsampleSize.y)
    {
        return;
    }

    float exposure = 1.0f;

    #if 1
        exposure = texelFetch(inAdaptedLumTex, ivec2(0, 0), 0).r, viewData.evCompensation;
    #elif 0
        exposure = viewData.exposure;
    #endif
    // TODO: Bloom ev compensation, we need this?
    // exposure *= pow(2.0, ev100 + compensation - 3.0);


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
            samples[i] = prefilter(samples[i] * exposure, prefilterFactor);
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
            // TODO: Can be pre compute.
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