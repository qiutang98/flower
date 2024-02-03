#version 460
#include "common_shader.glsl"

layout (set = 0, binding = 0)  uniform UniformFrameData { PerFrameData frameData; };

layout (set = 0, binding = 1) buffer lodNodeContinueSSBO
{
    uint data[];
} lodNodeContinue;


layout(set = 0, binding = 2) uniform writeonly image2D imageLODPatchMap;

layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    uvec2 groupThreadId = remap8x8(gl_LocalInvocationIndex);
    uvec2 dispatchId = groupThreadId + gl_WorkGroupID.xy * 8;
    ivec2 workPos = ivec2(dispatchId);

    ivec2 workSize = imageSize(imageLODPatchMap);
    if(workPos.x >= workSize.x || workPos.y >= workSize.y)
    {
        return;
    }

    vec2 uv = (vec2(workPos) + 0.5) / vec2(workSize);

    int lodLevel = 0;
    int dim = 4;
    while(lodLevel < frameData.landscape.lodCount - 1)
    {
        ivec2 posInLod = ivec2(dim * uv);
        int lodNodeBase = int((16 * (1 - pow(4, int(lodLevel)))) / -3);
        int lodNodePos = lodNodeBase + posInLod.y * dim + posInLod.x;

        const bool bSplit = (lodNodeContinue.data[lodNodePos] != 0);
        if(bSplit)
        {
            // Update dim.
            lodLevel ++;
            dim *= 2;
        }
        else
        {
            // No split, just return.
            break;
        }
    }

    imageStore(imageLODPatchMap, workPos, vec4((lodLevel + 0.5) / 255.0f));
}