
#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable
#extension GL_EXT_nonuniform_qualifier : enable

#include "cloud_render_common.glsl"

void filterColor(in const vec4 color, in const float weightConst, inout vec4 colorSum, inout float weightSum)
{
    float weight = color.w < -0.5 ? 0.0 : weightConst;

    colorSum  += weight * color;
    weightSum += weight;
}

layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    ivec2 texSize = imageSize(imageHdrSceneColor);
    ivec2 depthTextureSize = textureSize(inDepth, 0);
    ivec2 workPos = ivec2(gl_GlobalInvocationID.xy);

    if(workPos.x >= texSize.x || workPos.y >= texSize.y)
    {
        return;
    }
    AtmosphereParameters atmosphere = getAtmosphereParameters(frameData);
    const vec2 uv = (vec2(workPos) + vec2(0.5f)) / vec2(texSize);

    // Offset retarget for new seeds each frame
    uvec2 offset = uvec2(vec2(0.754877669, 0.569840296) * (frameData.frameIndex.x) * uvec2(texSize));
    uvec2 offsetId = workPos.xy + offset;
    offsetId.x = offsetId.x % texSize.x;
    offsetId.y = offsetId.y % texSize.y;
    float blueNoise2 = samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d(offsetId.x, offsetId.y, 0, 0u); 

    vec4 clipSpace = vec4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, 0.0, 1.0);
	vec4 viewPosH = frameData.camInvertProj * clipSpace;
    vec3 viewDir = viewPosH.xyz / viewPosH.w;
    vec3 worldDir = normalize((frameData.camInvertView * vec4(viewDir, 0.0)).xyz);

    vec4 srcColor = imageLoad(imageHdrSceneColor, workPos);
    float sceneZ = texture(sampler2D(inDepth, pointClampEdgeSampler), uv).r;
    // vec4 cloudColor = kuwaharaFilter(inCloudReconstructionTexture, linearClampEdgeSampler,uv);

    vec4 cloudColor = texture(sampler2D(inCloudReconstructionTexture, linearClampEdgeSampler), uv);
    vec3 result = srcColor.rgb;

    float w = 0.0;
    if(sceneZ <= 0.0f) // reverse z.
    {
        result = srcColor.rgb * cloudColor.a + cloudColor.rgb;  
    }

    if(frameData.cloud.cloudGodRay != 0)
    {
        vec4 fog = texture(sampler2D(inFog, linearClampEdgeSampler), uv);
        result.rgb = result.rgb * fog.w + fog.xyz;
    }




    // 
    imageStore(imageHdrSceneColor, workPos, vec4(result.rgb, w));
}