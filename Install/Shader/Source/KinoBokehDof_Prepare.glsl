#version 460

#include "KinoBokehDof_Common.glsl"


layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    if(shouldSkipRenderDof())
    {
        return;
    }

    ivec2 colorSize = imageSize(downSampleHDRImage);

    uvec2 groupThreadId = remap8x8(gl_LocalInvocationIndex);
    uvec2 dispatchId = groupThreadId + gl_WorkGroupID.xy * 8;
    ivec2 workPos = ivec2(dispatchId);

    if(workPos.x >= colorSize.x || workPos.y >= colorSize.y)
    {
        return;
    }

    // Work pos uv.
    const vec2 texelSize = 1.0f / vec2(colorSize);
    const vec2 uv = (vec2(workPos) + vec2(0.5f)) * texelSize;

    // 4x src color load.
    const ivec2 hdrSrcSize = textureSize(inHDRSceneColor, 0);
    const vec2 hdrSrcTexelSize = 1.0f / vec2(hdrSrcSize);
    vec3 c0 = texture(sampler2D(inHDRSceneColor, linearClampEdgeSampler), uv + vec2(-0.5, -0.5) * hdrSrcTexelSize.xy).rgb; // 0.75 linear
    vec3 c1 = texture(sampler2D(inHDRSceneColor, linearClampEdgeSampler), uv + vec2( 0.5, -0.5) * hdrSrcTexelSize.xy).rgb;
    vec3 c2 = texture(sampler2D(inHDRSceneColor, linearClampEdgeSampler), uv + vec2(-0.5,  0.5) * hdrSrcTexelSize.xy).rgb;
    vec3 c3 = texture(sampler2D(inHDRSceneColor, linearClampEdgeSampler), uv + vec2( 0.5,  0.5) * hdrSrcTexelSize.xy).rgb;

    c0 = pow(max(c0, vec3(0)), vec3(kBokehWorkingGamma));
    c1 = pow(max(c1, vec3(0)), vec3(kBokehWorkingGamma));
    c2 = pow(max(c2, vec3(0)), vec3(kBokehWorkingGamma));
    c3 = pow(max(c3, vec3(0)), vec3(kBokehWorkingGamma));

    // 4x depth load.
    const ivec2 depthSrcSize = textureSize(inDepth, 0);
    const vec2 depthSrcTexelSize = 1.0f / vec2(depthSrcSize);
    const float d0 = linearizeDepth(texture(sampler2D(inDepth, linearClampEdgeSampler), uv + vec2(-0.5, -0.5) * depthSrcTexelSize.xy).r, viewData);
    const float d1 = linearizeDepth(texture(sampler2D(inDepth, linearClampEdgeSampler), uv + vec2( 0.5, -0.5) * depthSrcTexelSize.xy).r, viewData);
    const float d2 = linearizeDepth(texture(sampler2D(inDepth, linearClampEdgeSampler), uv + vec2(-0.5,  0.5) * depthSrcTexelSize.xy).r, viewData);
    const float d3 = linearizeDepth(texture(sampler2D(inDepth, linearClampEdgeSampler), uv + vec2( 0.5,  0.5) * depthSrcTexelSize.xy).r, viewData);
    const vec4 depths = vec4(d0, d1, d2, d3);

    // Calculate the radiuses of CoCs at these sample points.
    float focusDepth = DofPush.distanceF;
    float lensCoeff =  DofPush.lensCoeff;

    if(DofPush.bFocusPMXCharacter > 0)
    {
        float minDepthF = uintBitsToFloat(depthRange.minDepth);
        float maxDepthF = uintBitsToFloat(depthRange.maxDepth);

        float sumDepthF = uintBitsToFloat(depthRange.sumPmxDepth);
        uint pmxPixelNum = depthRange.pmxPixelCount;

        // TODO: Add some filter and configable.
             if(DofPush.bFocusPMXCharacter == 1) { focusDepth = minDepthF; } 
        else if(DofPush.bFocusPMXCharacter == 2) { focusDepth = maxDepthF; }
        else {  focusDepth = sumDepthF / float(pmxPixelNum); }

        focusDepth += DofPush.pmxFoucusMinOffset;

        focusDepth = max(focusDepth, DofPush.focusLen);
        lensCoeff = DofPush.focusLen * DofPush.focusLen / (DofPush.fStop * (focusDepth - DofPush.focusLen) * DofPush.filmHeight * 2.0);
    }

    vec4 cocs = (depths - vec4(focusDepth)) * lensCoeff / depths;

    float minCoc = DofPush.bNearBlur > 0 ? -DofPush.maxCoc : -1e-5f;
    cocs = clamp(cocs, minCoc, DofPush.maxCoc);

    // [0.0, 1.0] weight.
    vec4 weights = saturate(abs(cocs) * DofPush.maxCoCRcp);

    // Anti flicker.
    weights.x *= 1.0f / (max3(c0) + 1.0f);
    weights.y *= 1.0f / (max3(c1) + 1.0f);
    weights.z *= 1.0f / (max3(c2) + 1.0f);
    weights.w *= 1.0f / (max3(c3) + 1.0f);

    float sumweights = sum(weights);

    // Output CoC = average of CoCs
    float coc = mean(cocs);

    // Weighted average of the color samples
    vec3 avg = (c0 * weights.x + c1 * weights.y + c2 * weights.z + c3 * weights.w) / sumweights;

    // Premultiply CoC again.
    avg *= smoothstep(0, hdrSrcTexelSize.y * 2, abs(coc));

    imageStore(downSampleHDRImage, workPos, vec4(avg, coc));
}