#version 460

#include "Sample.glsl"
#include "KinoBokehDof_Common.glsl"

layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    if(shouldSkipRenderDof())
    {
        return;
    }

    ivec2 colorSize = imageSize(HDRSceneColorImage);

    uvec2 groupThreadId = remap8x8(gl_LocalInvocationIndex);
    uvec2 dispatchId = groupThreadId + gl_WorkGroupID.xy * 8;
    ivec2 workPos = ivec2(dispatchId);

    if(workPos.x >= colorSize.x || workPos.y >= colorSize.y)
    {
        return;
    }

    const vec2 texelSize = 1.0f / vec2(colorSize);
    const vec2 uv = (vec2(workPos) + vec2(0.5f)) * texelSize;

    vec4 srcColor = imageLoad(HDRSceneColorImage, workPos);

    srcColor.rgb = max(srcColor.rgb, vec3(0.0f)); // avoid pow nan.
    srcColor.rgb = pow(srcColor.rgb, vec3(kBokehWorkingGamma));

       vec4 blurColor = catmullRom9Sample(inExpandFillImage, linearClampEdgeSampler, uv, vec2(colorSize));
    // vec4 blurColor = texture(sampler2D(inExpandFillImage, linearClampEdgeSampler), uv);

    // Slight jitter alpha by blue noise. 
    uvec2 offset = uvec2(vec2(0.754877669, 0.569840296) * frameData.frameIndex.x * uvec2(colorSize));
    uvec2 offsetId = dispatchId.xy + offset;
    offsetId.x = offsetId.x % colorSize.x;
    offsetId.y = offsetId.y % colorSize.y;
    float blueNoise = samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d(offsetId.x, offsetId.y, 0, 0u);
    float mulNoiseFactor = 2.0 * (0.5 - abs(saturate(blurColor.a) - 0.5));
    blurColor.a += mulNoiseFactor * (blueNoise * 2.0 - 1.0) * 0.1;
    blurColor.a = saturate(blurColor.a);
    

    vec3 colorResult = srcColor.rgb * blurColor.a + blurColor.rgb;

    colorResult.rgb = pow(colorResult.rgb, vec3(1.0 / kBokehWorkingGamma));

    imageStore(HDRSceneColorImage, workPos, vec4(colorResult, srcColor.a));
}