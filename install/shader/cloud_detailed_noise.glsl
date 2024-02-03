#version 460
#extension GL_GOOGLE_include_directive : enable

#include "cloud_noise_common.glsl"

// Detail frequency is 8.0f
#define kDetailFrequency 8.0

layout (set = 0, binding = 0, r8) uniform image3D imageWorleyNoise; 
layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main()
{
    ivec3 texSize = imageSize(imageWorleyNoise);
    ivec3 workPos = ivec3(gl_GlobalInvocationID.xyz);

    if(workPos.x >= texSize.x || workPos.y >= texSize.y || workPos.z >= texSize.z)
    {
        return;
    }

    const vec3 uvw = (vec3(workPos) + vec3(0.5f)) / vec3(texSize);

    float detailNoise = 
		worleyFbm(uvw, kDetailFrequency * 1.0) * 0.625 +
    	worleyFbm(uvw, kDetailFrequency * 2.0) * 0.250 +
    	worleyFbm(uvw, kDetailFrequency * 4.0) * 0.125;

	imageStore(imageWorleyNoise, workPos, vec4(detailNoise));
}