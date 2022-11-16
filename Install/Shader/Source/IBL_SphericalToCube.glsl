#version 460

#extension GL_GOOGLE_include_directive : enable

#include "IBL_Common.glsl"

// Spherical map to cube map.

vec2 cubeSampleDirectionToSphericalMapUv(vec3 v)
{
	const vec2 kInvAtan = vec2(0.1591, 0.3183);
    vec2 result = kInvAtan * vec2(atan(v.z, v.x), asin(v.y)) + 0.5;
    return result;
}

layout (set = 0, binding = 0, rgba16f) uniform imageCube imageCubeEnv;
layout (set = 0, binding = 1) uniform texture2D inHdr;

// Common sampler set.
#define COMMON_SAMPLER_SET 1
#include "CommonSamplerSet.glsl"

#include "ColorSpace.glsl"

// 
layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main()
{
    ivec3 cubeCoord = ivec3(gl_GlobalInvocationID);
    ivec2 cubeSize = imageSize(imageCubeEnv);

    if(cubeCoord.x >= cubeSize.x || cubeCoord.y >= cubeSize.y || cubeCoord.z >= 6)
    {
        return;
    }

    const vec2 pixPos = vec2(cubeCoord) + vec2(0.5f);
    const vec2 uv = pixPos / vec2(cubeSize);

    // Get sample direction. flip direction.
    vec3 sampleVector = -getSamplingVector(cubeCoord.z, uv);

    vec3 sampleColor = inputColorPrepare(textureLod(sampler2D(inHdr, linearClampEdgeSampler), cubeSampleDirectionToSphericalMapUv(sampleVector), 0).rgb);
    
    imageStore(imageCubeEnv, cubeCoord, vec4(sampleColor, 1.0));
    return;
}