#version 460

#extension GL_GOOGLE_include_directive : enable

#include "AdaptiveExposureCommon.glsl"

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
