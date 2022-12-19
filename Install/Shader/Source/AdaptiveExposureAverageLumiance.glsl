#version 460

/*
** Physical based render code, develop by engineer: qiutanguu.
*/

#extension GL_GOOGLE_include_directive : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_shader_subgroup_basic : enable

#include "AdaptiveExposureCommon.glsl"

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
    vec4 filterResult = vec4(0.0, 0.0, totalSum * vec2(autoExposurePush.lowPercent, autoExposurePush.highPercent));

    // Filter one by one, total 128 times. 
    for (i = 0; i < kHistogramBin; i++)
    {
        filterLuminance(i, maxHistogramValue, filterResult);
    }

    // Clamp to user brightness range
    return clamp(filterResult.x / max(filterResult.y, 1e-4), autoExposurePush.minBrightness, autoExposurePush.maxBrightness);
}

float getExposureMultiplier(float avgLuminance)
{
    avgLuminance = max(1e-4, avgLuminance);

    // https://knarkowicz.wordpress.com/2016/01/09/automatic-exposure/
    float keyValue = 1.03 - (2.0 / (2.0 + log2(avgLuminance + 1.0)));

    // Compensate add here. diff from unity.
    keyValue += autoExposurePush.exposureCompensation;

    float exposure = keyValue / avgLuminance;
    return exposure;
}

float interpolateExposure(float newExposure, float oldExposure)
{
    float delta = newExposure - oldExposure;
    float speed = delta > 0.0 ? autoExposurePush.speedDown : autoExposurePush.speedUp;

    // Time delta from https://knarkowicz.wordpress.com/2016/01/09/automatic-exposure/
    float exposure = oldExposure + delta * saturate(1.0 - exp(-autoExposurePush.deltaTime * speed));
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

        if(!cameraCut(frameData))
        {
            // Get prev frame's lum.
            float prevExposure = texelFetch(inPrevLumImage, ivec2(0, 0), 0).x;
            exposure = interpolateExposure(exposure, prevExposure);
        }

        imageStore(adaptedLumImage, ivec2(0, 0), vec4(exposure, 0.0, 0.0, 0.0));
    }
}
