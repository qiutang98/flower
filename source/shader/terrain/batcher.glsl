#version 460
#extension GL_GOOGLE_include_directive : enable

#define CBT_SET_INDEX 0
#define CBT_BINDING_INDEX 0
#include "../cbt/cbt.glsl"

layout (set = 0, binding = 1) buffer SSBODrawElementsIndirectCommandBuffer { uint u_DrawElementsIndirectCommand[]; };
layout (set = 0, binding = 2) buffer SSBODispatchIndirectCommandBuffer { uint u_DispatchIndirectCommand[]; };

layout (push_constant) uniform PushConsts 
{  
    int u_CbtID;
    int u_MeshletIndexCount;
};

#include "../leb/leb.glsl"

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main()
{
    const int cbtID = u_CbtID;
    uint nodeCount = cbt_NodeCount(cbtID);

    u_DispatchIndirectCommand[0] = nodeCount / 256u + 1u;
    u_DrawElementsIndirectCommand[0] = u_MeshletIndexCount;
    u_DrawElementsIndirectCommand[1] = nodeCount;
}