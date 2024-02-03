#version 460
#extension GL_GOOGLE_include_directive : enable

#include "gtao_common.glsl"

// 5x5 filter tent, need 2 border pixels.
const uint kThreadCountX = 16;
const uint kThreadCountY = 16;
const uint kThreadCount = kThreadCountX * kThreadCountY;

const uint kBorderSize = 2;

const uint kSampleCountX = kThreadCountX + kBorderSize * 2;
const uint kSampleCountY = kThreadCountY + kBorderSize * 2;
const uint kSampleCount = kSampleCountX * kSampleCountY;

shared float sharedDeviceZ[kSampleCount];
shared float sharedAo[kSampleCount];

vec2 getDeviceZandAO(ivec2 threadIdPos)
{
    const ivec2 locatePos = threadIdPos + ivec2(kBorderSize);
    const uint sampleFlat = locatePos.y * kSampleCountX + locatePos.x;

    const float ao = sharedAo[sampleFlat];
    const float depth = sharedDeviceZ[sampleFlat];

    return vec2(depth, ao);
}

layout (local_size_x = kThreadCountX, local_size_y = kThreadCountY) in;
void main()
{
    ivec2 gtaoSize = imageSize(GTAOFilterImage);
    ivec2 workPos = ivec2(gl_GlobalInvocationID.xy);

    // Offset to border pos and fill whole tile.
    ivec2 basicSamplePos = ivec2(gl_WorkGroupID.xy) * ivec2(kThreadCountX, kThreadCountY) - ivec2(kBorderSize);
    uint fillID = gl_LocalInvocationIndex;
    while(fillID < kSampleCount)
    {
        ivec2 samplePos = basicSamplePos + ivec2(fillID % kSampleCountX, fillID / kSampleCountX);
        samplePos = clamp(samplePos, ivec2(0), gtaoSize - ivec2(1));
        
        float deviceZ = texelFetch(inDepth, samplePos, 0).r;
        float ao = texelFetch(inGTAO, samplePos, 0).r;

        sharedDeviceZ[fillID] = deviceZ;
        sharedAo[fillID] = ao;

        // Step thread count.
        fillID += kThreadCount;
    }

    if(workPos.x >= gtaoSize.x || workPos.y >= gtaoSize.y)
    {
        return;
    }

    // Cache shared memory data.
    groupMemoryBarrier();
    barrier();

    const ivec2 kThreadId = ivec2(gl_LocalInvocationID.xy);
    const vec2 texelSize = 1.0f / vec2(gtaoSize);
    const vec2 uv = (vec2(workPos) + vec2(0.5f)) * texelSize;

    float filterAO;
    {
        vec2 zDiff;

        float thisZ = getDeviceZandAO(kThreadId).x;
        {
            ivec2 X1Offset = ivec2(1, 0);
            ivec2 X2Offset = ivec2(2, 0);

            float XM2Z = getDeviceZandAO(kThreadId - X2Offset).x;
            float XM1Z = getDeviceZandAO(kThreadId - X1Offset).x;
            float XP1Z = getDeviceZandAO(kThreadId + X1Offset).x;
            float XP2Z = getDeviceZandAO(kThreadId + X2Offset).x;

            // Get extrapolated point either side
            float C1 = abs((XM1Z + (XM1Z - XM2Z)) - thisZ);
            float C2 = abs((XP1Z + (XP1Z - XP2Z)) - thisZ);

            if (C1 < C2)
            {
                zDiff.x = XM1Z - XM2Z;
            }
            else
            {
                zDiff.x = XP2Z - XP1Z;
            }
        }
        {
            ivec2 Y2Offset = ivec2(0, 2);
            ivec2 Y1Offset = ivec2(0, 1);

            float YM2Z = getDeviceZandAO(kThreadId - Y2Offset).x;
            float YM1Z = getDeviceZandAO(kThreadId - Y1Offset).x;
            float YP1Z = getDeviceZandAO(kThreadId + Y1Offset).x;
            float YP2Z = getDeviceZandAO(kThreadId + Y2Offset).x;

            // Get extrapolated point either side
            float C1 = abs((YM1Z + (YM1Z - YM2Z)) - thisZ);
            float C2 = abs((YP1Z + (YP1Z - YP2Z)) - thisZ);

            if (C1 < C2)
            {
                zDiff.y = YM1Z - YM2Z;
            }
            else
            {
                zDiff.y = YP2Z - YP1Z;
            }
        }

        float sumAO = 0;
        float sumWeight = 0;

        int x, y;

        // Get the Z Value to compare against 
        float depthBase = thisZ - (zDiff.x * 2) - (zDiff.y * 2);

        for (y = -2; y <= 2; y++)
        {
            float planeZ = depthBase;

            for (x = -2; x <= 2; x++)
            {
                // Get value and see how much it compares to the centre with the gradients
                float XDiff = abs(x);

                vec2 SampleZAndAO = getDeviceZandAO(kThreadId + ivec2(x, y));
                float Weight = 1.0f;

                {
                    // Get the bilateral weight. This is a function of the difference in height between the plane equation and the base depth
                    // Compare the Z at this sample with the gradients 
                    float SampleZDiff = abs(planeZ - SampleZAndAO.x);

                    const float SpatialFilterWeight = 20000;
                    Weight = 1.0f - saturate(SampleZDiff * SpatialFilterWeight);
                }

                sumAO += SampleZAndAO.y * Weight;
                sumWeight += Weight;

                planeZ += zDiff.x;
            }
            depthBase += zDiff.y;
        }

        // Weight normalize.
        sumAO /= sumWeight;
        sumAO *= (kPI * 0.5f);

        // Style ao.
        sumAO = 1.0 - (1.0 - pow(sumAO, GTAOPush.power)) * GTAOPush.intensity;

        filterAO = sumAO;
    }
    imageStore(GTAOFilterImage, workPos, vec4(filterAO, 0.0, 0.0, 0.0));
}