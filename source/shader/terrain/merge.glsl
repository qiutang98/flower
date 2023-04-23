#version 460
#extension GL_GOOGLE_include_directive : enable

#include "common.glsl"

layout(local_size_x = 256) in;
void main(void)
{
    const int cbtID = 0;
    uint threadID = gl_GlobalInvocationID.x;

    if (threadID < cbt_NodeCount(cbtID)) 
    {
        // and extract triangle vertices
        cbt_Node node = cbt_DecodeNode(cbtID, threadID);
        vec4 triangleVertices[3] = DecodeTriangleVertices(node);

        // compute target LoD
        vec2 targetLod = LevelOfDetail(triangleVertices);

        leb_DiamondParent diamond = leb_DecodeDiamondParent_Square(node);
        bool shouldMergeBase = LevelOfDetail(DecodeTriangleVertices(diamond.base)).x < 1.0;
        bool shouldMergeTop = LevelOfDetail(DecodeTriangleVertices(diamond.top)).x < 1.0;

        if (shouldMergeBase && shouldMergeTop) 
        {
            leb_MergeNode_Square(cbtID, node, diamond);
        }
    }
}