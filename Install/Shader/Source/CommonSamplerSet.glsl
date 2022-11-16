#ifndef COMMON_SAMPLER_SET_GLSL
#define COMMON_SAMPLER_SET_GLSL


// These samplers' mipmapMode are point. Commonly use for RT read.
layout(set = COMMON_SAMPLER_SET, binding = 0) uniform sampler pointClampEdgeSampler;
layout(set = COMMON_SAMPLER_SET, binding = 1) uniform sampler pointClampBorder0000Sampler;
layout(set = COMMON_SAMPLER_SET, binding = 2) uniform sampler pointRepeatSampler;
layout(set = COMMON_SAMPLER_SET, binding = 3) uniform sampler linearClampEdgeSampler;
layout(set = COMMON_SAMPLER_SET, binding = 4) uniform sampler linearClampBorder0000Sampler;
layout(set = COMMON_SAMPLER_SET, binding = 5) uniform sampler linearRepeatSampler;
layout(set = COMMON_SAMPLER_SET, binding = 6) uniform sampler linearClampBorder1111Sampler;
layout(set = COMMON_SAMPLER_SET, binding = 7) uniform sampler pointClampBorder1111Sampler;
// Sampler mipmapMode.

#endif