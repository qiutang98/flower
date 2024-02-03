#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_ballot : enable

#include "sssr_common.glsl"

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main()
{
    // Prepare intersection args.
    uint rayCount = ssboRayCounter.rayCount; // Ray count.

    ssboIntersectCommand.args.x = (rayCount + 63) / 64;
    ssboIntersectCommand.args.y = 1;
    ssboIntersectCommand.args.z = 1;

    // Prepare denoise args.
    uint tileCount = ssboRayCounter.denoiseTileCount; // 8x8 tile count.
    
    ssboDenoiseCommand.args.x = tileCount;
    ssboDenoiseCommand.args.y = 1;
    ssboDenoiseCommand.args.z = 1;
}