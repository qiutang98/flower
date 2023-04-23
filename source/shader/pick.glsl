#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : require

#include "common/shared_functions.glsl"

layout (set = 0, binding = 0) buffer SSBOPickId { uint pickSceneNodeId; };
layout (set = 0, binding = 1) uniform utexture2D inSceneNodeIdTexture; 

layout(push_constant) uniform PushConsts
{   
    ivec2 pickPos;
};

void main()
{
    uint loadId = texelFetch(inSceneNodeIdTexture, pickPos, 0).r;

    pickSceneNodeId = unpackToSceneNodeId(loadId);
}