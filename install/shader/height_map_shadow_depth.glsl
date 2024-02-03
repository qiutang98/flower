#version 460

#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable

#define SHARED_SAMPLER_SET 1
#include "common_shader.glsl"

layout(set = 0, binding = 0) uniform texture2D textureHeightmap;
layout(set = 0, binding = 1) uniform writeonly image2D imageShadowDepth;

// Heightmap raymarching shadow depth and build ESM effect.
layout (local_size_x = 8, local_size_y = 8) in;
void main()
{




}