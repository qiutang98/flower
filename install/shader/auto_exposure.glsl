#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable

#define SHARED_SAMPLER_SET 1
#include "common_shader.glsl"

layout (set = 0, binding = 0) uniform texture2D inHDRSceneColor;
layout (set = 0, binding = 1, r16f) uniform image2D adaptedLumImage;
layout (set = 0, binding = 2, r32ui) uniform uimage2D histogramImage;
layout (set = 0, binding = 3) uniform utexture2D inHistogramImage;
layout (set = 0, binding = 4) uniform texture2D inPrevLumImage;
layout (set = 0, binding = 5) uniform UniformFrameData { PerFrameData frameData; };

const uint kHistogramBin = 128;
const uint kHistogramThreadDim = 16;
const uint kHistogramReductionThreadDimX = kHistogramThreadDim;
const uint kHistogramReductionThreadDimY = kHistogramBin / kHistogramThreadDim;

float getHistogramBinFromLuminance(float value)
{
    return saturate(log2(value) * frameData.postprocessing.autoExposureScale + frameData.postprocessing.autoExposureOffset);
}

float getLuminanceFromHistogramBin(float bin)
{
    return exp2((bin - frameData.postprocessing.autoExposureOffset) / frameData.postprocessing.autoExposureScale);
}

#ifdef EXPOSURE_HISTOGRAM_PASS

const uint kDimBlockReduce = 3;
shared uint histogramShared[kHistogramBin];

layout (local_size_x = kHistogramThreadDim, local_size_y = kHistogramThreadDim) in;
void main() 
{
    const uint threadId = gl_LocalInvocationIndex;
    if(threadId < kHistogramBin)
    {
        // Init shared memory.
        histogramShared[threadId] = 0;
    }

    groupMemoryBarrier();
    barrier();

    ivec2 hdrColorSize = textureSize(inHDRSceneColor, 0);
    ivec2 workPosBasic = ivec2(gl_GlobalInvocationID.xy) * int(kDimBlockReduce);

    for(uint i = 0; i < kDimBlockReduce; i ++)
    {
        for(uint j = 0; j < kDimBlockReduce; j ++)
        {
            ivec2 workPos = workPosBasic + ivec2(i, j);
            if (workPos.x < hdrColorSize.x && workPos.y < hdrColorSize.y) 
            {
                uint weight = 1;
                vec2 uv = (vec2(workPos) + vec2(0.5)) / vec2(hdrColorSize);

                vec3 hdrColor = texture(sampler2D(inHDRSceneColor, pointClampEdgeSampler), uv).rgb;
                float lum = luminance(hdrColor);

                // Get log lum in [0,1]
                float logLum = getHistogramBinFromLuminance(lum);

                // Map to histogram buffer.
                uint idx = uint(logLum * (kHistogramBin - 1u));
                atomicAdd(histogramShared[idx], weight);
            }
        }
    }

    groupMemoryBarrier();
    barrier();

    if(threadId < kHistogramBin)
    {
        imageAtomicAdd(histogramImage, ivec2(threadId, 0), histogramShared[threadId]);
    }
}

#endif // EXPOSURE_HISTOGRAM_PASS

#ifdef EXPOSURE_AVERAGE_PASS

#extension GL_GOOGLE_include_directive : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_shader_subgroup_basic : enable

shared uint lumMaxShared;
shared uint lumAllShared;
shared uint histogramShared[kHistogramBin];

float getBinValue(uint index, float maxHistogramValue)
{
    return maxHistogramValue * histogramShared[index];
}

void filterLuminance(uint i, float maxHistogramValue, inout vec4 filterResult)
{
    float binValue = getBinValue(i, maxHistogramValue);

    // filter dark areas
    float offset = min(filterResult.z, binValue);
    binValue -= offset;
    filterResult.zw -= offset.xx;

    // filter highlights
    binValue = min(filterResult.w, binValue);
    filterResult.w -= binValue;

    // luminance at the bin
    float luminance = getLuminanceFromHistogramBin(float(i) / float(kHistogramBin));

    filterResult.xy += vec2(luminance * binValue, binValue);
}

float getAverageLuminance(float maxHistogramValue)
{
    // Sum of all bins
    uint i;
    float totalSum = float(lumAllShared) * maxHistogramValue;

    // Skip darker and lighter parts of the histogram to stabilize the auto exposure
    // x: filtered sum
    // y: accumulator
    // zw: fractions
    vec4 filterResult = vec4(0.0, 0.0, totalSum * vec2(frameData.postprocessing.autoExposureLowPercent, frameData.postprocessing.autoExposureHighPercent));

    // Filter one by one, total 128 times. 
    for (i = 0; i < kHistogramBin; i++)
    {
        filterLuminance(i, maxHistogramValue, filterResult);
    }

    // Clamp to user brightness range
    return clamp(filterResult.x / max(filterResult.y, 1e-4), frameData.postprocessing.autoExposureMinBrightness, frameData.postprocessing.autoExposureMaxBrightness);
}

float getExposureMultiplier(float avgLuminance)
{
    avgLuminance = max(1e-4, avgLuminance);

#if 0
    // https://knarkowicz.wordpress.com/2016/01/09/automatic-exposure/
    float keyValue = 1.03 - (2.0 / (2.0 + log2(avgLuminance + 1.0)));
    keyValue += frameData.postprocessing.autoExposureExposureCompensation;
#else
    float keyValue = frameData.postprocessing.autoExposureExposureCompensation;
#endif

    float exposure = keyValue / avgLuminance;
    return exposure;
}

float interpolateExposure(float newExposure, float oldExposure)
{
    float delta = newExposure - oldExposure;
    float speed = delta > 0.0 ? frameData.postprocessing.autoExposureSpeedDown : frameData.postprocessing.autoExposureSpeedUp;

    // Time delta from https://knarkowicz.wordpress.com/2016/01/09/automatic-exposure/
    float exposure = oldExposure + delta * saturate(1.0 - exp2(-frameData.postprocessing.autoExposureDeltaTime * speed));
    return exposure;
}

layout (local_size_x = kHistogramReductionThreadDimX, local_size_y = kHistogramReductionThreadDimY) in;
void main() 
{
    const uint threadId = gl_LocalInvocationIndex;
    const uint sampleLum = texelFetch(inHistogramImage, ivec2(threadId, 0), 0).r;

    // Clear and init.
    lumMaxShared = 0;
    lumAllShared = 0;
    histogramShared[threadId] = sampleLum;

    // Find max lum in subgroups.
    const uint maxLumWave = subgroupMax(sampleLum);
    const uint totalLumWave = subgroupAdd(sampleLum);

    groupMemoryBarrier();
    barrier();

    // Find max lum in all thread groups.
    if(subgroupElect())
    {
        atomicMax(lumMaxShared, maxLumWave);
        atomicAdd(lumAllShared, totalLumWave);
    }

    groupMemoryBarrier();
    barrier();

    // Filter in thread id 0.
    if(threadId == 0)
    {
        float maxValue = 1.0 / float(lumMaxShared);

        float avgLuminance = getAverageLuminance(maxValue);
        float exposure = getExposureMultiplier(avgLuminance);

        if(frameData.bCameraCut == 0)
        {
            // Get prev frame's lum.
            float prevExposure = texelFetch(inPrevLumImage, ivec2(0, 0), 0).x;
            exposure = interpolateExposure(exposure, prevExposure);
        }

        imageStore(adaptedLumImage, ivec2(0, 0), vec4(exposure, 0.0, 0.0, 0.0));
    }
}

#endif // EXPOSURE_AVERAGE_PASS