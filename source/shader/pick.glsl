#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : require

#include "common/shared_functions.glsl"

layout (set = 0, binding = 0) buffer SSBOPickId { uint pickSceneNodeId; };
layout (set = 0, binding = 1) uniform utexture2D inSceneNodeIdTexture; 

#define SHARED_SAMPLER_SET 1
#include "common/shared_sampler.glsl"

layout(push_constant) uniform PushConsts
{   
    vec2 pickUv;
};

void main()
{
    uint loadId = texture(usampler2D(inSceneNodeIdTexture, pointClampEdgeSampler), pickUv).r;

    pickSceneNodeId = unpackToSceneNodeId(loadId);
}