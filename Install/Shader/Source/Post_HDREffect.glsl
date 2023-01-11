#version 460

#extension GL_EXT_samplerless_texture_functions : enable

#include "Common.glsl"
#include "PoissonDisk.glsl"

layout (set = 0, binding = 0, rgba16f) uniform image2D HDRSceneColorImage; 
layout (set = 0, binding = 1) uniform texture2D inHDRSceneColorImage; 

// Other common set.
layout (set = 1, binding = 0) uniform UniformView  { ViewData  viewData;  };
layout (set = 2, binding = 0) uniform UniformFrame { FrameData frameData; };

// Common sampler set.
#define COMMON_SAMPLER_SET 3
#include "CommonSamplerSet.glsl"

#define BLUE_NOISE_TEXTURE_SET 4
#define BLUE_NOISE_BUFFER_SET 5
#include "BlueNoiseCommon.glsl"

layout(push_constant) uniform PushConsts
{   
    int bEnableVignette;
    float vignetteFalloff;

    int bFringeMode; // 0 is off, 1 is Conrady, 2 is barrel.

    float fringe_barrelStrength;
    float fringe_zoomStrength;
    float fringe_lateralShift;

} HDREffectPush;

float getVignette(vec2 uv, float rcpAspect)
{
    vec2 coord = (uv - 0.5) * vec2(rcpAspect, 1.0) * 2;
    float rf = sqrt(dot(coord, coord)) * HDREffectPush.vignetteFalloff;
    float rf2_1 = rf * rf + 1.0;

    return 1.0 / (rf2_1 * rf2_1);
}

// 

vec2 remap( vec2 t, vec2 a, vec2 b ) 
{
	return clamp( (t - a) / (b - a), 0.0, 1.0 );
}

vec3 spectrumOffsetRGB(float t)
{
    float t0 = 3.0 * t - 1.5;
	vec3 ret = clamp( vec3( -t0, 1.0 - abs(t0), t0), 0.0, 1.0);
    
    return ret;
}

vec2 barrelDistortion(vec2 p, vec2 amt)
{
    p = 2.0 * p - 1.0;

    const float maxBarrelPower = 5.0;
    float theta  = atan(p.y, p.x);
    vec2 radius = vec2( length(p) );

    radius = pow(radius, 1.0 + maxBarrelPower * amt);
    p.x = radius.x * cos(theta);
    p.y = radius.y * sin(theta);

    return p * 0.5 + 0.5;
}

vec2 brownConradyDistortion(vec2 uv, float dist)
{
    uv = uv * 2.0 - 1.0;

    float barrelDistortion1 =  0.100 * dist; 
    float barrelDistortion2 = -0.025 * dist; 

    float r2 = dot(uv,uv);
    uv *= 1.0 + barrelDistortion1 * r2 + barrelDistortion2 * r2 * r2;
    
    return uv * 0.5 + 0.5;
}

vec2 distort(vec2 uv, float t, vec2 minDistort, vec2 maxDistort)
{
    vec2 dist = mix(minDistort, maxDistort, t);
    
    if(HDREffectPush.bFringeMode == 1)
    {
        return brownConradyDistortion(uv, 75.0 * dist.x);
    }
    else if(HDREffectPush.bFringeMode == 2)
    {
        return barrelDistortion(uv, 1.75 * dist);
    }

    return uv;
}

vec3 applyFringeEffect(vec2 uv, uvec2 res, uvec2 workPos)
{
    const float kMaxDistPX = 50.0;

    float maxDistortPx = kMaxDistPX * HDREffectPush.fringe_barrelStrength;

    // Get min max distort.
	vec2 maxDistort = vec2(maxDistortPx) / vec2(res);
    vec2 minDistort = 0.5 * maxDistort;
    vec2 oversiz = distort( vec2(1.0), 1.0, minDistort, maxDistort);

    // Get distort sample uv.
    uv = mix(uv, remap(uv, 1.0 - oversiz, oversiz), HDREffectPush.fringe_zoomStrength);
    
    const int kSampleCount = 7;
    const float stepsiz = 1.0 / (float(kSampleCount) - 1.0);

    uvec2 offset = uvec2(vec2(0.754877669, 0.569840296) * frameData.frameIndex.x * res);
    uvec2 offsetId = workPos.xy + offset;
    offsetId.x = offsetId.x % res.x;
    offsetId.y = offsetId.y % res.y;
    float blueNoise = samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d(offsetId.x, offsetId.y, 0, 0u); 

    float t = blueNoise * stepsiz;
    
    vec3 sumcol = vec3(0.0);
	vec3 sumw = vec3(0.0);

	for ( int i = 0; i < kSampleCount; ++i )
	{
		vec3 w = spectrumOffsetRGB(t);
		sumw += w;

        vec2 uvd = distort(uv, t, minDistort, maxDistort);
		sumcol += w * texture(sampler2D(inHDRSceneColorImage, pointClampEdgeSampler), uvd).rgb;

        t += stepsiz;
	}
    sumcol.rgb /= sumw;
    
    return sumcol.rgb;
}

vec4 applyFringeEffect(vec2 uv, float rcpAspect, vec4 inSrc)
{
    vec2 spc = (uv - 0.5) * vec2(rcpAspect, 1.0);
    float r2 = dot(spc, spc);


    float f0 = 1.0 - r2 * HDREffectPush.fringe_lateralShift * 0.02;
    float f1 = 1.0 + r2 * HDREffectPush.fringe_lateralShift * 0.02;

    vec4 src = inSrc;

    src.r = texture(sampler2D(inHDRSceneColorImage, pointClampEdgeSampler), (uv - 0.5) * f0 + 0.5).r;
    src.b = texture(sampler2D(inHDRSceneColorImage, pointClampEdgeSampler), (uv - 0.5) * f1 + 0.5).b;

    return src;
}

// HDR Effect before all post.
layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    ivec2 colorSize = imageSize(HDRSceneColorImage);

    uvec2 groupThreadId = remap8x8(gl_LocalInvocationIndex);
    uvec2 dispatchId = groupThreadId + gl_WorkGroupID.xy * 8;
    ivec2 workPos = ivec2(dispatchId);

    if(workPos.x >= colorSize.x || workPos.y >= colorSize.y)
    {
        return;
    }

    float aspect = float(colorSize.x) / float(colorSize.y);
    float rcpAspect = 1.0 / aspect;

    const vec2 texelSize = 1.0f / vec2(colorSize);
    const vec2 uv = (vec2(workPos) + vec2(0.5f)) * texelSize;

    vec4 processColor = texelFetch(inHDRSceneColorImage, workPos, 0);

    // Apply Chromatic Aberration.
    if(HDREffectPush.bFringeMode > 0)
    {
        if(HDREffectPush.bFringeMode == 3)
        {
            processColor = applyFringeEffect(uv, rcpAspect, processColor);
        }
        else
        {
            processColor.rgb = applyFringeEffect(uv, uvec2(colorSize), dispatchId);
        }
    }

    // Apply vignette.
    if(HDREffectPush.bEnableVignette > 0)
    {
        processColor.rgb *= getVignette(uv, rcpAspect);
    }


    imageStore(HDRSceneColorImage, workPos, processColor);
}