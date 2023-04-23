#version 460
#extension GL_GOOGLE_include_directive : enable

#include "shared_cbt.glsl"



layout (local_size_x = 256) in;
void main(void)
{
    const int cbtID = u_CbtID;
    uint cnt = (1u << u_PassID);
    uint threadID = gl_GlobalInvocationID.x;

    if (threadID < cnt) 
    {
        uint nodeID = threadID + cnt;
        uint x0 = cbt_HeapRead(cbtID, cbt_CreateNode(nodeID << 1u     , u_PassID + 1));
        uint x1 = cbt_HeapRead(cbtID, cbt_CreateNode(nodeID << 1u | 1u, u_PassID + 1));

        cbt_a_HeapWrite(cbtID, cbt_CreateNode(nodeID, u_PassID), x0 + x1);
    }
}
