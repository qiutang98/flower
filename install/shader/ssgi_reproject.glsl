#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_EXT_samplerless_texture_functions : enable

#define SHARED_SAMPLER_SET    1

#include "common_shader.glsl"
#include "common_lighting.glsl"

layout (set = 0, binding = 0) uniform texture2D inDepth;
layout (set = 0, binding = 1) uniform texture2D inGbufferV;
layout (set = 0, binding = 2) uniform UniformFrameData { PerFrameData frameData; };

layout (set = 0, binding = 3) uniform texture2D inSSGI; // 

layout (set = 0, binding = 4) uniform texture2D inSSGIHistory;
layout (set = 0, binding = 5, rgba16f) uniform image2D imageReproject;
layout (set = 0, binding = 6) uniform texture2D inMomentHistory;
layout (set = 0, binding = 7, rgba16f) uniform image2D imageMomentHistory;

layout (set = 0, binding = 8) uniform texture2D inGBufferB; // 
layout (set = 0, binding = 9) uniform texture2D inPrevGBufferB; // 
layout (set = 0, binding = 10) uniform texture2D inPrevDepth;
layout (set = 0, binding = 11) uniform texture2D inGbufferID;
layout (set = 0, binding = 12) uniform texture2D inPrevGbufferID;

layout(push_constant) uniform PushConsts
{   
    float kMinAlpha;
    float kMinAlphaMoment;
};

#define kAngleDiff 0.9f
#define kPlaneDiff 5.0f

vec3 clipAABB(vec3 aabbMin, vec3 aabbMax, vec3 historySample)
{
    vec3 aabbCenter = 0.5f * (aabbMax + aabbMin);
    vec3 extentClip = 0.5f * (aabbMax - aabbMin) + 0.001f;

    vec3 colorVector = historySample - aabbCenter;
    vec3 colorVectorClip = colorVector / extentClip;

    colorVectorClip  = abs(colorVectorClip);
    float maxAbsUnit = max(max(colorVectorClip.x, colorVectorClip.y), colorVectorClip.z);

    if (maxAbsUnit > 1.0)
    {
        return aabbCenter + colorVector / maxAbsUnit; 
    }

    return historySample; 
}

layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    ivec2 workSize = imageSize(imageReproject);

    uvec2 groupThreadId = remap8x8(gl_LocalInvocationIndex);
    uvec2 dispatchId = groupThreadId + gl_WorkGroupID.xy * 8;
    ivec2 workPos = ivec2(dispatchId);

    if(workPos.x >= workSize.x || workPos.y >= workSize.y)
    {
        return;
    }

    const vec2 texelSize = 1.0f / vec2(workSize);
    const vec2 uv = (vec2(workPos) + vec2(0.5f)) * texelSize;
    const float depth = texture(sampler2D(inDepth, pointClampEdgeSampler), uv).r;

    // Basic history valid state.
    bool bHistoryValid = 
        depth > 0.0 && // Non sky pixel.
        frameData.bCameraCut == 0; // Non camera cut frame.

    vec2 historyUv = uv + texture(sampler2D(inGbufferV, pointClampEdgeSampler), uv).rg;

    // History uv on range.
    bHistoryValid = bHistoryValid && onRange(historyUv, vec2(0.0), vec2(1.0));

    // Current point sample geometry reject is no confidence due to TAA jitter.
    // TODO: Bilinear sample pattern compare.
#if 0
    // Reject by compare prev frame geometry info.
    if(bHistoryValid)
    {
        float currentId = texture(sampler2D(inGbufferID, pointClampEdgeSampler), uv).r;
        float prevId = texture(sampler2D(inPrevGbufferID, pointClampEdgeSampler), historyUv).r;
        if(currentId != prevId)
        {
            bHistoryValid = false;
        }
    }

    vec3 currentNormal = unpackWorldNormal(texture(sampler2D(inGBufferB, pointClampEdgeSampler), uv).rgb);
    if(bHistoryValid)
    {
        vec3 prevNormal = unpackWorldNormal(texture(sampler2D(inPrevGBufferB, pointClampEdgeSampler), historyUv).rgb);
        if(dot(currentNormal, prevNormal) < kAngleDiff)
        {
            bHistoryValid = false;
        }
    }

    if(bHistoryValid)
    {
        vec3 worldPosCur = getWorldPos(uv, depth, frameData);
        vec3 worldPosPrev = getWorldPos(uv, texture(sampler2D(inDepth, pointClampEdgeSampler), historyUv).r, frameData);

        vec3 toCur = worldPosCur - worldPosPrev;
        float dis2Plane = abs(dot(toCur, currentNormal));

        if(dis2Plane > kPlaneDiff)
        {
            bHistoryValid = false;
        }
    }
#endif

    // Load current frame SSGI intersect result.
    vec3 currentColor = texture(sampler2D(inSSGI, pointClampEdgeSampler), uv).xyz;

    vec3  historyColor = currentColor;

    vec2  moments;
    moments.r = dot(currentColor, vec3(1.0 / 3.0));
    moments.g = moments.r * moments.r;

    vec2  historyMoment = moments;

    float historyLen = 1.0f;
    if(bHistoryValid)
    {
        const vec4 historySSGI = texture(sampler2D(inSSGIHistory, pointClampEdgeSampler), historyUv);
        const vec4 historyMomentLen = texture(sampler2D(inMomentHistory, pointClampEdgeSampler), historyUv);

        historyColor  = historySSGI.xyz;
        historyMoment = historyMomentLen.xy;
        historyLen    = historyMomentLen.z;

        // Step history length.
        historyLen = min(32.0, historyLen + 1.0);
    }

    const float alpha       = bHistoryValid ? max(kMinAlpha,       1.0 / historyLen) : 1.0;
    const float momentAlpha = bHistoryValid ? max(kMinAlphaMoment, 1.0 / historyLen) : 1.0;

    moments = mix(historyMoment, moments, momentAlpha);
    float variance = max(0.0, moments.g - moments.r * moments.r);

    // Clamp history.
    if (bHistoryValid)
    {
        vec3 stdDev;
        vec3 mean;

        vec3 m0 = vec3(0.0f);
        vec3 m1 = vec3(0.0f);

        const int radius = 5; // Crazy config man.
        for(int x = -radius; x <= radius; x++)
        {
            for(int y = -radius; y <= radius; y++)
            {
                ivec2 sampleCoord =  workPos + ivec2(x, y);
                vec3 sampleCurrentColor = texelFetch(inSSGI, sampleCoord, 0).xyz;
                m0 += sampleCurrentColor;
                m1 += sampleCurrentColor * sampleCurrentColor;
            }
        }

        const float weight = (radius * 2.0 + 1.0) * (radius * 2.0 + 1.0);
        mean = m0 / weight;

        vec3 variance = (m1 / weight) - (mean * mean);
        stdDev = sqrt(max(variance, 0.0f));

        vec3 radianceMin = mean - stdDev;
        vec3 radianceMax = mean + stdDev;

        historyColor = clipAABB(radianceMin, radianceMax, historyColor);
    }
    else
    {
        // NOTE: History reproject fail will cast screen edge exist some noise, use a netrual value as variance.
        variance = mix(variance, 0.5, 0.5);
    }

    vec4  outSSGI;
    vec4  outMomemtHistory;

    outSSGI.xyz = mix(historyColor, currentColor, alpha);
    outSSGI.w = variance;
    outMomemtHistory.xy = moments;
    outMomemtHistory.z = historyLen;

    imageStore(imageReproject, ivec2(workPos), outSSGI);
    imageStore(imageMomentHistory, ivec2(workPos), outMomemtHistory);
}