#version 460

/*
** Physical based render code, develop by engineer: qiutanguu.
*/

// Single scatter version volumetric cloud renderering.
// Q: Why no use multi-scatter version? 
// A: Ugly and no fit flower render style.

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable
#include "Cloud_Common.glsl"
#include "Noise.glsl"
#include "Phase.glsl"

// Evaluate quater resolution.
layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    ivec2 texSize = imageSize(imageCloudRenderTexture);
    ivec2 workPos = ivec2(gl_GlobalInvocationID.xy);

    if(workPos.x >= texSize.x || workPos.y >= texSize.y)
    {
        return;
    }

    // Get bayer offset matrix.
    uint bayerIndex = frameData.frameIndex.x % 16;
    ivec2 bayerOffset = ivec2(bayerFilter4x4[bayerIndex] % 4, bayerFilter4x4[bayerIndex] / 4);

    // Get evaluate position in full resolution.
    ivec2 fullResSize = texSize * 4;
    ivec2 fullResWorkPos = workPos * 4 + ivec2(bayerOffset);

    // Get evaluate uv in full resolution.
    const vec2 uv = (vec2(fullResWorkPos) + vec2(0.5f)) / vec2(fullResSize);

    // Offset retarget for new seeds each frame
    uvec2 offset = uvec2(vec2(0.754877669, 0.569840296) * (frameData.frameIndex.x) * uvec2(fullResSize));
    uvec2 offsetId = fullResWorkPos.xy + offset;
    offsetId.x = offsetId.x % fullResSize.x;
    offsetId.y = offsetId.y % fullResSize.y;
    float blueNoise = samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d(offsetId.x, offsetId.y, 0, 0u); 

     // We are revert z.
    vec4 clipSpace = vec4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, 0.0, 1.0);
    vec4 viewPosH = viewData.camInvertProj * clipSpace;
    vec3 viewSpaceDir = viewPosH.xyz / viewPosH.w;
    vec3 worldDir = normalize((viewData.camInvertView * vec4(viewSpaceDir, 0.0)).xyz);


    AtmosphereParameters atmosphere = getAtmosphereParameters(frameData);

    float depth = 0.0; // reverse z.
    vec4 cloudColor = cloudColorCompute(atmosphere, uv, blueNoise, depth, workPos, worldDir);

	imageStore(imageCloudRenderTexture, workPos, cloudColor);
    imageStore(imageCloudDepthTexture, workPos, vec4(depth));
}