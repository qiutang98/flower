#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : require

#define SHARED_SAMPLER_SET 1
#include "common_shader.glsl"

layout (set = 0, binding = 0) buffer SSBOPickId { uint pickSceneNodeId; };
layout (set = 0, binding = 1)  uniform texture2D inGBufferId;
layout (set = 0, binding = 2) readonly buffer SSBOPerObject { PerObjectInfo objectDatas[]; };
layout (set = 0, binding = 3) uniform UniformFrameData { PerFrameData frameData; };

layout(push_constant) uniform PushConsts
{   
    vec2 pickUv;
};

void main()
{
    uint objectId = unpackFrom16bitObjectId(texture(sampler2D(inGBufferId, pointClampEdgeSampler), pickUv).r);

    if(objectId <= kMaxObjectId)
    {
        PerObjectInfo object = objectDatas[objectId];
        pickSceneNodeId = object.sceneNodeId;
    }
    else if(objectId == kSkyObjectId)
    {
        pickSceneNodeId = frameData.skyComponentSceneNodeId;
    }


}