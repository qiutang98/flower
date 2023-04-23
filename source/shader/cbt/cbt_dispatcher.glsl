#version 460
#extension GL_GOOGLE_include_directive : enable

#define CBT_SET_INDEX 0
#define CBT_BINDING_INDEX 0
#include "cbt.glsl"

layout (set = 0, binding = 1) buffer SSBODispatchIndirectCommandBuffer { uint u_CbtDispatchBuffer[]; };

layout (push_constant) uniform PushConsts 
{  
    int u_CbtID;
};

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main()
{
    const int cbtID = u_CbtID;
    uint nodeCount = cbt_NodeCount(cbtID);

    u_CbtDispatchBuffer[0] = max(nodeCount >> 8, 1u);
}
