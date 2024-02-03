#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_ray_query : enable

// SLAM TSDF reconstruction from depth buffer.

#define SHARED_SAMPLER_SET    1
#include "common_shader.glsl"

layout (set = 0, binding = 0) uniform texture2D inDepth;
layout (set = 0, binding = 1) buffer  SSBOPerObject    { PerObjectInfo objectDatas[];              };
layout (set = 0, binding = 2) uniform UniformFrameData { PerFrameData frameData; };
layout (set = 0, binding = 3, rgba16f) uniform image3D imageTruncSDF;

layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main()
{
    
}