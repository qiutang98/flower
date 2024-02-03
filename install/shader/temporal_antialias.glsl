#version 460

#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_samplerless_texture_functions : enable


#define SHARED_SAMPLER_SET 1
#include "common_shader.glsl"

layout (set = 0, binding = 0) uniform UniformFrameData { PerFrameData frameData; };
layout (set = 0, binding = 1) uniform texture2D inHDRSceneColor;
layout (set = 0, binding = 2) uniform texture2D inSceneDepth;
layout (set = 0, binding = 3) uniform texture2D inVelocity;
layout (set = 0, binding = 4) uniform texture2D inAdaptedLumTex;
layout (set = 0, binding = 5, rgba16f)  uniform writeonly image2D outTAAImage;
layout (set = 0, binding = 6) uniform texture2D inHistory;

layout (push_constant) uniform PushConsts 
{  
    float kAntiFlickerIntensity;
    float kContrastForMaxAntiFlicker;
    float kSampleHistorySharpening;
    float kHistoryContrastBlendLerp;

    float kBaseBlendFactor;
    float kFilterWeights[9];
};

// 3x3 neighbor sample pattern.
const vec2 kNeighbourOffsets[9] =
{
/*
    vec2( 0.0f,  0.0f),
    vec2( 0.0f,  1.0f),
    vec2( 1.0f,  0.0f),
    vec2(-1.0f,  0.0f),
    vec2( 0.0f, -1.0f),
    vec2( 1.0f,  1.0f),
    vec2( 1.0f, -1.0f),
    vec2(-1.0f,  1.0f),
    vec2(-1.0f, -1.0f),
*/
    vec2( 0,  0),
    vec2( 0,  1),
    vec2( 1,  0),
    vec2(-1,  0),
    vec2( 0, -1),
    vec2(-1,  1), // ****
    vec2( 1, -1),
    vec2( 1,  1), // ****
    vec2(-1, -1)
};



// [Karis 2014]: find closest cross depth.
vec3 getClosestDepthCross(ivec2 samplePos)
{
    float d0 = texelFetch(inSceneDepth, samplePos, 0).r;

    float d1 = texelFetch(inSceneDepth, samplePos + ivec2( 1,  1), 0).r;
    float d2 = texelFetch(inSceneDepth, samplePos + ivec2(-1,  1), 0).r;
    float d3 = texelFetch(inSceneDepth, samplePos + ivec2( 1, -1), 0).r;
    float d4 = texelFetch(inSceneDepth, samplePos + ivec2(-1, -1), 0).r;

    vec3 closest = vec3(0.0, 0.0, d0);

    closest = closest.z < d1 ? vec3(vec2( 1,  1), d1) : closest;
    closest = closest.z < d2 ? vec3(vec2(-1,  1), d2) : closest;
    closest = closest.z < d3 ? vec3(vec2( 1, -1), d3) : closest;
    closest = closest.z < d4 ? vec3(vec2(-1, -1), d4) : closest;

    return closest;
}

// Filmic SMAA presentation[Jimenez 2016]
vec4 sampleHistoryBicubic5Tap(vec2 uv, float sharpening)
{
    const vec2 historySize = vec2(textureSize(inHistory, 0));
    const vec2 historyTexelSize = 1.0f / historySize;

    vec2 samplePos = uv * historySize;
    vec2 tc1 = floor(samplePos - 0.5) + 0.5;
    vec2 f  = samplePos - tc1;
    vec2 f2 = f * f;
    vec2 f3 = f * f2;

    const float c = sharpening;
    vec2 w0  = -c         * f3 +  2.0 * c         * f2 - c * f;
    vec2 w1  =  (2.0 - c) * f3 - (3.0 - c)        * f2          + 1.0;
    vec2 w2  = -(2.0 - c) * f3 + (3.0 - 2.0 * c)  * f2 + c * f;
    vec2 w3  = c          * f3 - c                * f2;
    vec2 w12 = w1 + w2;

    vec2 tc0  = historyTexelSize   * (tc1 - 1.0);
    vec2 tc3  = historyTexelSize   * (tc1 + 2.0);
    vec2 tc12 = historyTexelSize  * (tc1 + w2 / w12);

    vec4 s0 = texture(sampler2D(inHistory, linearClampEdgeSampler), vec2(tc12.x,  tc0.y));
    vec4 s1 = texture(sampler2D(inHistory, linearClampEdgeSampler), vec2(tc0.x,  tc12.y));
    vec4 s2 = texture(sampler2D(inHistory, linearClampEdgeSampler), vec2(tc12.x, tc12.y));
    vec4 s3 = texture(sampler2D(inHistory, linearClampEdgeSampler), vec2(tc3.x,   tc0.y));
    vec4 s4 = texture(sampler2D(inHistory, linearClampEdgeSampler), vec2(tc12.x,  tc3.y));

    float cw0 = (w12.x * w0.y);
    float cw1 = (w0.x  * w12.y);
    float cw2 = (w12.x * w12.y);
    float cw3 = (w3.x  * w12.y);
    float cw4 = (w12.x * w3.y);

    // Anti-ring from unity
    vec4 minColor = min(min(min(min(s0, s1), s2), s3), s4);
    vec4 maxColor = max(max(max(max(s0, s1), s2), s3), s4);

    s0 *= cw0;
    s1 *= cw1;
    s2 *= cw2;
    s3 *= cw3;
    s4 *= cw4;

    vec4 historyFiltered = s0 + s1 + s2 + s3 + s4;
    float weightSum = cw0 + cw1 + cw2 + cw3 + cw4;

    vec4 filteredVal = historyFiltered / weightSum;

    // Anti-ring from unity.
    // This sortof neighbourhood clamping seems to work to avoid the appearance of overly dark outlines in case
    // sharpening of history is too strong.
    filteredVal = clamp(filteredVal, minColor, maxColor);

    // Final output clamp.
    return clamp(filteredVal, vec4(0.0), vec4(kMaxHalfFloat));
}

// Lumiance aware perceptual weight.
float perceptualWeight(vec4 colorYcocg)
{
    return 1.0f / (1.0 + colorYcocg.x);
}

float perceptualInvWeight(vec4 colorYcocg)
{
    return 1.0 / (1.0 - colorYcocg.x);
}

// 3x3 neighborhood samples.
struct NeighbourhoodSamples
{
    vec4 neighbours[8];

    vec4 central;
    vec4 minNeighbour;
    vec4 maxNeighbour;
    vec4 avgNeighbour;
};

const vec3  kAmbientBiasc = vec3(0.0f);
const float kExposureScale = 1.0f;

// Convert to ycocg color space and add lumiance aware weight premul.
vec4 colorInputPostProcess(vec4 color)
{
    color.xyz  = RGBToYCoCg(kExposureScale * color.xyz + kAmbientBiasc);
    color.xyz *= perceptualWeight(color);
    return color;
}

vec4 colorOutputPostProcess(vec4 color)
{
    color.xyz *= perceptualInvWeight(color);
    color.xyz  = (YCoCgToRGB(color.xyz) - kAmbientBiasc) / kExposureScale;
    return color;
}

void varianceClip(inout NeighbourhoodSamples samples, float historyLuma, float colorLuma, float velocityLengthInPixels, out float aggressiveClampedHistoryLuma)
{
    // Prepare moments.
    vec4 moment1 = vec4(0);
    vec4 moment2 = vec4(0);

    for (int i = 0; i < 8; ++i)
    {
        moment1 += samples.neighbours[i];
        moment2 += samples.neighbours[i] * samples.neighbours[i];
    }
    samples.avgNeighbour = moment1 / 8.0f;

    // Also accumulate center.
    moment1 += samples.central;
    moment2 += samples.central * samples.central;

    // Average moments.
    moment1 /= 9.0f;
    moment2 /= 9.0f;

    // Get std dev.
    vec4 stdDev = sqrt(abs(moment2 - moment1 * moment1));

    // Luma based anti filcker, from unity hdrp.
    float stDevMultiplier = 1.5;
    {
        float aggressiveStdDevLuma = stdDev.x * 0.5;

        aggressiveClampedHistoryLuma = clamp(historyLuma, moment1.x - aggressiveStdDevLuma, moment1.x + aggressiveStdDevLuma);
        float temporalContrast = saturate(abs(colorLuma - aggressiveClampedHistoryLuma) / max(max(0.15, colorLuma), aggressiveClampedHistoryLuma));

        const float maxFactorScale = 2.25f; // when stationary
        const float minFactorScale = 0.80f; // when moving more than slightly

        float localizedAntiFlicker = mix(
            kAntiFlickerIntensity * minFactorScale, 
            kAntiFlickerIntensity * maxFactorScale, 
            saturate(1.0f - 2.0f * (velocityLengthInPixels)));

        stDevMultiplier += mix(0.0, localizedAntiFlicker, smoothstep(0.05, kContrastForMaxAntiFlicker, temporalContrast));
        stDevMultiplier = mix(stDevMultiplier, 0.75, saturate(velocityLengthInPixels / 50.0f));
    }

    samples.minNeighbour = moment1 - stdDev * stDevMultiplier;
    samples.maxNeighbour = moment1 + stdDev * stDevMultiplier;
}

// From Playdead's TAA
vec4 historyClipAABB(vec4 history, vec4 minimum, vec4 maximum)
{
    // Note: only clips towards aabb center (but fast!)
    vec4 center  = 0.5 * (maximum + minimum);
    vec4 extents = 0.5 * (maximum - minimum);

    // This is actually `distance`, however the keyword is reserved
    vec4 offset  = history - center;

    // Check out of range state.
    float maxUnit = max3(abs(offset.xyz / extents.xyz));
    if (maxUnit > 1.0)
    {
        return center + (offset / maxUnit);
    }

    // Otherwise just use history.
    return history;
}

float historyContrast(float historyLuma, float minNeighbourLuma, float maxNeighbourLuma, float baseBlendFactor)
{
    float lumaContrast = max(maxNeighbourLuma - minNeighbourLuma, 0) / historyLuma;
    float blendFactor = baseBlendFactor;
    return saturate(blendFactor / (1.0 + lumaContrast));
}

layout (local_size_x = 8, local_size_y = 8) in;
void main()
{  
    const ivec2 outTAAImageSize = imageSize(outTAAImage);

    const uvec2 groupThreadId = remap8x8(gl_LocalInvocationIndex);
    const uvec2 dispatchId = groupThreadId + gl_WorkGroupID.xy * 8;
    const ivec2 workPos = ivec2(dispatchId);

    if (workPos.x >= outTAAImageSize.x || workPos.y >= outTAAImageSize.y)
    {
        return; 
    }

    const vec2 texelSize = 1.0f / vec2(outTAAImageSize);
    const vec2 uv = (vec2(workPos) + vec2(0.5f)) * texelSize;

    // Load scene color.
    vec4 color = texture(sampler2D(inHDRSceneColor, linearClampEdgeSampler), uv);
    color = clamp(color, vec4(0.0), vec4(kMaxHalfFloat));

    // Load velocity form closest cross 3x3.
    vec2 velocity;
    {
        vec2 sampleVelocityUv = uv + getClosestDepthCross(workPos).xy * texelSize;
        velocity = texture(sampler2D(inVelocity, pointClampEdgeSampler), sampleVelocityUv).xy;
    }
    float velocityFactor = 0.0f;

    const bool bHistoryValid = (frameData.bCameraCut == 0);
    if (bHistoryValid)
    {
        const vec2 reprojectUv = uv + velocity;

        // Sample filtered history.
        vec4 history = colorInputPostProcess(sampleHistoryBicubic5Tap(reprojectUv, kSampleHistorySharpening));

        // Prepare neighbor.
        NeighbourhoodSamples samples;
        {
            samples.central = colorInputPostProcess(color);

            [[unroll]] 
            for(int i = 0; i < 8; ++i)
            {
                const vec2 sampleUv = uv + kNeighbourOffsets[i + 1] * texelSize;
                samples.neighbours[i] = colorInputPostProcess(texture(sampler2D(inHDRSceneColor, linearClampEdgeSampler), sampleUv));
            }
        }

        // Filter color.
        vec4 filteredColor = samples.central;
        if(frameData.postprocessing.bTAAEnableColorFilter != 0)
        {
            float totalWeight = 1.0f;

            for (int i = 0; i < 8; ++i)
            {
                float w = kFilterWeights[i + 1];

                // Accumulate.
                filteredColor += samples.neighbours[i] * w;
                totalWeight += w;
            }

            // Average.
            filteredColor /= totalWeight;
        }

        // History outof range.
        if(!onRange(reprojectUv, vec2(0), vec2(1)))
        {
            history = filteredColor;
        }

        const float colorLumiance = filteredColor.x;
        const float historyLumiance = history.x;

        const float velocityLength = length(velocity);
        const float velocityLengthInPixels = velocityLength * length(vec2(outTAAImageSize));
    
        // Variance clip history clamp color(for min & max).
        float aggressivelyClampedHistoryLuma = 0;
        varianceClip(samples, historyLumiance, colorLumiance, velocityLengthInPixels, aggressivelyClampedHistoryLuma);

        // Clip history with aabb.
        history = historyClipAABB(history, samples.minNeighbour, samples.maxNeighbour);

        // Compute blend factor.
        float blendFactor;
        {
            float historyLum = historyContrast(aggressivelyClampedHistoryLuma, samples.minNeighbour.x, samples.maxNeighbour.x, kBaseBlendFactor);
            blendFactor = mix(colorLumiance, historyLum, kHistoryContrastBlendLerp);

            // Velocity factor.
            blendFactor = mix(blendFactor, saturate(2.0f * blendFactor), saturate(velocityLengthInPixels  / 50.0f));
        }

        blendFactor = clamp(blendFactor, 0.03f, 0.98f);

        color.xyz = mix(history.xyz, filteredColor.xyz, blendFactor);
        color = colorOutputPostProcess(color);

        color = clamp(color, vec4(0.0), vec4(kMaxHalfFloat));
    }

    imageStore(outTAAImage, workPos, color);
}