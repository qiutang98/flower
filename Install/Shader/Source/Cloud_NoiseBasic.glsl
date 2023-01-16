#version 460
#extension GL_GOOGLE_include_directive : enable

/*
** Physical based render code, develop by engineer: qiutanguu.
*/

#include "Noise.glsl"

#define kBasicFrequency 4.0

layout (set = 0, binding = 0, r8) uniform image3D imageBasicNoise; // 128 x 128 x 128

float remap(float x, float a, float b, float c, float d)
{
    return (((x - a) / (b - a)) * (d - c)) + c;
}

float basicNoiseComposite(vec4 v)
{
    float wfbm = v.y * 0.625 + v.z * 0.25 + v.w * 0.125; 
    
    // cloud shape modeled after the GPU Pro 7 chapter
    return remap(v.x, wfbm - 1.0, 1.0, 0.0, 1.0);
}

layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main()
{
    ivec3 texSize = imageSize(imageBasicNoise);
    ivec3 workPos = ivec3(gl_GlobalInvocationID.xyz);

    if(workPos.x >= texSize.x || workPos.y >= texSize.y || workPos.z >= texSize.z)
    {
        return;
    }

    const vec3 uvw = (vec3(workPos) + vec3(0.5f)) / vec3(texSize);

    float pfbm = mix(1.0, perlinfbm(uvw, kBasicFrequency, 7), 0.5);
    pfbm = abs(pfbm * 2.0 - 1.0); // billowy perlin noise
    
    vec4 col = vec4(0.0);
    col.g += worleyFbm(uvw, kBasicFrequency);
    col.b += worleyFbm(uvw, kBasicFrequency * 2.0);
    col.a += worleyFbm(uvw, kBasicFrequency * 4.0);

    col.r += remap(pfbm, 0.0, 1.0, col.g, 1.0); // perlin-worley

	imageStore(imageBasicNoise, workPos,  vec4(basicNoiseComposite(col)));
}