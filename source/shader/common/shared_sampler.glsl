#ifndef SHARED_SAMPLER_GLSL
#define SHARED_SAMPLER_GLSL

#ifndef SHARED_SAMPLER_SET
#error "Must define SHARED_SAMPLER_SET before include this file!"
#endif

// These samplers' mipmapMode are point. Commonly use for RT read.
layout(set = SHARED_SAMPLER_SET, binding = 0) uniform sampler pointClampEdgeSampler;
layout(set = SHARED_SAMPLER_SET, binding = 1) uniform sampler pointClampBorder0000Sampler;
layout(set = SHARED_SAMPLER_SET, binding = 2) uniform sampler pointRepeatSampler;
layout(set = SHARED_SAMPLER_SET, binding = 3) uniform sampler linearClampEdgeSampler;
layout(set = SHARED_SAMPLER_SET, binding = 4) uniform sampler linearClampBorder0000Sampler;
layout(set = SHARED_SAMPLER_SET, binding = 5) uniform sampler linearRepeatSampler;
layout(set = SHARED_SAMPLER_SET, binding = 6) uniform sampler linearClampBorder1111Sampler;
layout(set = SHARED_SAMPLER_SET, binding = 7) uniform sampler pointClampBorder1111Sampler;

// With mip filter
layout(set = SHARED_SAMPLER_SET, binding = 8) uniform sampler linearClampEdgeMipFilterSampler;
layout(set = SHARED_SAMPLER_SET, binding = 9) uniform sampler linearRepeatMipFilterSampler;

#endif