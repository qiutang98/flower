#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : require

#include "common_shader.glsl"

layout (set = 0, binding = 0) uniform texture2D inHeightFiledTexture;
layout (set = 0, binding = 1) uniform writeonly image2D bakeHeightFieldMinMax0; 

layout (push_constant) uniform PushConsts 
{  
    uint bFromSrcDepth;
};

void genMinMax(ivec2 samplePos, inout vec2 minMax)
{
    // Always set 0 is fine because already set mipcount as 1 in cpp.
    vec2 src = texelFetch(inHeightFiledTexture, samplePos, 0).xy;

    // Load from original heightmap.
    src.y = (bFromSrcDepth != 0) ? src.x : src.y;

    // op.
    minMax.x = min(minMax.x, src.x);
    minMax.y = max(minMax.y, src.y);
}

layout (local_size_x = 8, local_size_y = 8) in;
void main()
{

    // Src dimension.
    ivec2 srcSize = textureSize(inHeightFiledTexture, 0);

    uvec2 groupThreadId = remap8x8(gl_LocalInvocationIndex);
    uvec2 dispatchId = groupThreadId + gl_WorkGroupID.xy * 8;
    ivec2 workPos = ivec2(dispatchId);

    ivec2 workSize = imageSize(bakeHeightFieldMinMax0);
    if(workPos.x >= workSize.x || workPos.y >= workSize.y)
    {
        return;
    }

    ivec2 samplePos0 = clamp(workPos * 2 + ivec2(0, 0), ivec2(0), srcSize - 1);
    ivec2 samplePos1 = clamp(workPos * 2 + ivec2(0, 1), ivec2(0), srcSize - 1);
    ivec2 samplePos2 = clamp(workPos * 2 + ivec2(1, 0), ivec2(0), srcSize - 1);
    ivec2 samplePos3 = clamp(workPos * 2 + ivec2(1, 1), ivec2(0), srcSize - 1);

    vec2 minMax = vec2(9999999.0f, -9999999.0f);

    genMinMax(samplePos0, minMax);
    genMinMax(samplePos1, minMax);
    genMinMax(samplePos2, minMax);
    genMinMax(samplePos3, minMax);

    vec2 ratio = vec2(srcSize) / vec2(workSize);
    bool needExtraSampleX = ratio.x > 2.0;
    bool needExtraSampleY = ratio.y > 2.0;

    if(needExtraSampleX)
    {
        genMinMax(clamp(workPos * 2 + ivec2(2, 1), ivec2(0), srcSize - 1), minMax);
        genMinMax(clamp(workPos * 2 + ivec2(2, 0), ivec2(0), srcSize - 1), minMax);
    }
    if(needExtraSampleY)
    {
        genMinMax(clamp(workPos * 2 + ivec2(0, 2), ivec2(0), srcSize - 1), minMax);
        genMinMax(clamp(workPos * 2 + ivec2(1, 2), ivec2(0), srcSize - 1), minMax);
    }
    if(needExtraSampleX && needExtraSampleY)
    {
        genMinMax(clamp(workPos * 2 + ivec2(2, 2), ivec2(0), srcSize - 1), minMax);
    }

    imageStore(bakeHeightFieldMinMax0, workPos, vec4(minMax, 0.0f, 0.0f));
}
