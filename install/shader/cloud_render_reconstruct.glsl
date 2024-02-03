#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable

#include "cloud_render_common.glsl"

layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    ivec2 texSize = imageSize(imageCloudReconstructionTexture);
    ivec2 workPos = ivec2(gl_GlobalInvocationID.xy);

    if(workPos.x >= texSize.x || workPos.y >= texSize.y)
    {
        return;
    }

    const vec2 uv = (vec2(workPos) + vec2(0.5f)) / vec2(texSize);
    const float traceCloudDepth = texelFetch(inCloudDepthTexture, workPos / 4, 0).r;
    
    const vec2 curEvaluateCloudTexelSize = 1.0f / vec2(textureSize(inCloudRenderTexture, 0));

    // Reproject to get prev uv.
    vec3 worlPosCur = getWorldPos(uv, traceCloudDepth, frameData);
    
    vec4 projPosPrev  = frameData.camViewProjPrev * vec4(worlPosCur, 1.0);
    vec3 projPosPrevH = projPosPrev.xyz / projPosPrev.w;

    vec2 uvPrev = projPosPrevH.xy * 0.5 + 0.5;
    uvPrev.y = 1.0 - uvPrev.y;

    bool bCameraCut = frameData.bCameraCut != 0;

    // Valid check.
    bool bPrevUvValid = onRange(uvPrev, vec2(0.0), vec2(1.0)) && (!bCameraCut);
    if(bPrevUvValid)
    {
        bPrevUvValid  = textureLod(sampler2D(inDepthPrev, pointClampEdgeSampler), uvPrev, 0.0).r <= 0.0;
    }

    vec4 color = vec4(0.0);
    vec4 fog = vec4(0.0);
    float depthZ = 0.0; 

    if(bPrevUvValid)
    {
        // Evaluate, fetch it.
        vec4 curColor   = texelFetch(inCloudRenderTexture, workPos / 4, 0);
        float curDepthZ = texelFetch(inCloudDepthTexture,  workPos / 4, 0).r;
        float preDepthZ = texture(sampler2D(inCloudDepthReconstructionTextureHistory,  linearClampEdgeSampler), uvPrev).r;

        // Evaluate state check.
        uint  bayerIndex  = frameData.frameIndex.x % 16;
        ivec2 bayerOffset = ivec2(kBayerMatrix16[bayerIndex] % 4, kBayerMatrix16[bayerIndex] / 4);
        ivec2 workDeltaPos = workPos % 4;
        const bool bUpdateEvaluate = (workDeltaPos.x == bayerOffset.x) && (workDeltaPos.y == bayerOffset.y);
        if(bUpdateEvaluate)
        {
            // Just update color is good enough.
            color = curColor;
            depthZ = curDepthZ;
        }
        else
        {
            // Prev uv valid, sample history with prev Uv.
            color  = texture(sampler2D(inCloudReconstructionTextureHistory,  linearClampEdgeSampler), uvPrev);
            depthZ = preDepthZ;
        }
    }
    else
    {
        // No history valid, no evaluate, bilinear sample current.
        color  = texture(sampler2D(inCloudRenderTexture, linearClampEdgeSampler), uv);
        depthZ = texture(sampler2D(inCloudDepthTexture,  linearClampEdgeSampler), uv).r;
    }

    imageStore(imageCloudReconstructionTexture, workPos, color);
    imageStore(imageCloudDepthReconstructionTexture, workPos, vec4(depthZ));
}