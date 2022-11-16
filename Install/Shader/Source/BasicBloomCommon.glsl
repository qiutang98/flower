#ifndef BasicBloomCommon_GLSL
#define BasicBloomCommon_GLSL

#extension GL_GOOGLE_include_directive : enable

#define MIX_BLOOM_UPSCALE 1
#define MIX_BLOOM_OUTPUT 0

// 9 tap upscale kernal.
const uint kUpscaleCount = 9;
const vec2 kUpscaleSampleCoords[kUpscaleCount] = 
{
    {0,0},{-1,-1},{0,-1},{1,-1},{1,0},{1,1},{0,1},{-1,1},{-1,0}
};

const float kUpsampleWeights[kUpscaleCount] = 
{
    0.25,0.0625,0.125,0.0625,0.125,0.0625,0.125,0.0625,0.125
};

vec3 upscampleTentFilter(vec2 uv, texture2D inTex, sampler samplerTex, float blurRadius)
{
    vec3 outColor = vec3(0);

    // Get src texture size and compute it's pixel size.
    uvec2 srcSize = textureSize(inTex, 0);
    vec2 pixelSize = 1.0f / vec2(srcSize);

    for(uint i = 0; i < kUpscaleCount; i ++)
    {
        vec2 sampleUv = uv + kUpscaleSampleCoords[i] * pixelSize;
        outColor += kUpsampleWeights[i] * texture(sampler2D(inTex, samplerTex), sampleUv).rgb;
    }

#if MIX_BLOOM_UPSCALE
    return outColor;
#else
    return outColor * blurRadius;
#endif
    
}

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

#endif