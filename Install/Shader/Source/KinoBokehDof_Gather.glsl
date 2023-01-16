#version 460

#include "KinoBokehDof_Common.glsl"

layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    if(shouldSkipRenderDof())
    {
        return;
    }

    ivec2 colorSize = imageSize(gatherImage);

    uvec2 groupThreadId = remap8x8(gl_LocalInvocationIndex);
    uvec2 dispatchId = groupThreadId + gl_WorkGroupID.xy * 8;
    ivec2 workPos = ivec2(dispatchId);

    if(workPos.x >= colorSize.x || workPos.y >= colorSize.y)
    {
        return;
    }

    const vec2 texelSize = 1.0f / vec2(colorSize);
    const vec2 uv = (vec2(workPos) + vec2(0.5f)) * texelSize;

    const ivec2 downsampleHDRSize = textureSize(inDownSampleHDRImage, 0);
    const vec2 downsampleTexelSize = 1.0f / vec2(downsampleHDRSize);

    // Load center sample first.
    vec4 centerSample = texture(sampler2D(inDownSampleHDRImage, linearClampEdgeSampler), uv);


    //
    vec4 backGroundAcc = vec4(0);
    vec4 foreGroundAcc = vec4(0);

    // TODO: Two pass gather optimize.
    const int kSampleCount = 71;
    for (int i = 0; i < kSampleCount; i++)
    {
        // TODO: Tile max coc instead of use max coc.
        vec2 offset = DofPush.maxCoc * kVeogelDisk_71[i];
        float offsetLen = length(offset);

        vec2 sampleUv = uv + vec2(offset.x * DofPush.aspectRcp, offset.y);

        vec4 tap = texture(sampler2D(inDownSampleHDRImage, linearClampEdgeSampler), sampleUv);

        float bgCoC =  max(min(centerSample.a, tap.a), 0.0f);
        float fgCoc = -tap.a;

        // Compare the CoC to the sample distance.
        // Add a small margin to smooth out.
        float margin = downsampleTexelSize.y * 2;
        float bgWeight = saturate((bgCoC - offsetLen + margin) / margin);
        float fgWeight = saturate((fgCoc - offsetLen + margin) / margin);

        // Cut influence from focused areas because they're darkened by CoC
        // premultiplying. This is only needed for near field.
        fgWeight *= step(downsampleTexelSize.y, fgCoc);

        // Accumulation
        backGroundAcc += vec4(tap.rgb, 1) * bgWeight;
        foreGroundAcc += vec4(tap.rgb, 1) * fgWeight;
    }

    // Get the weighted average.
    backGroundAcc.rgb /= backGroundAcc.a + ((backGroundAcc.a == 0) ? 1.0 : 0.0);
    foreGroundAcc.rgb /= foreGroundAcc.a + ((foreGroundAcc.a == 0) ? 1.0 : 0.0);

    // BG: Calculate the alpha value only based on the center CoC.
    // This is a rather aggressive approximation but provides stable results.
    backGroundAcc.a = smoothstep(downsampleTexelSize.y, downsampleTexelSize.y * 2, centerSample.a);

    // FG: Normalize the total of the weights.
    foreGroundAcc.a *= kPI / float(kSampleCount);

    // Alpha premultiplying
    vec3 rgb = vec3(0);
    rgb = mix(rgb, backGroundAcc.rgb, saturate(backGroundAcc.a));
    rgb = mix(rgb, foreGroundAcc.rgb, saturate(foreGroundAcc.a));

    // Combined alpha value
    float alpha = (1 - saturate(backGroundAcc.a)) * (1 - saturate(foreGroundAcc.a));

    imageStore(gatherImage, workPos, vec4(rgb, alpha));
}