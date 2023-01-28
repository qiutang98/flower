#version 460

/*
** Physical based render code, develop by engineer: qiutanguu.
*/

#extension GL_GOOGLE_include_directive : enable

// Capture 360 cube map with low resolution.
// Use for scene capture sphere fallback.

#extension GL_EXT_samplerless_texture_functions : enable

#include "Cloud_Common.glsl"
#include "Noise.glsl"
#include "Phase.glsl"


layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main()
{
    ivec3 cubeCoord = ivec3(gl_GlobalInvocationID.xy, frameData.earthAtmosphere.updateFaceIndex);
    ivec2 cubeSize = imageSize(imageCubeEnv);

    if(cubeCoord.x >= cubeSize.x || cubeCoord.y >= cubeSize.y)
    {
        return;
    }

    const vec2 pixPos = vec2(cubeCoord) + vec2(0.5f);
    const vec2 uv = pixPos / vec2(cubeSize);

    float depth = 0.0;
    vec3 worldDir = getSamplingVector(cubeCoord.z, uv);
    
    vec4 cloudColor = cloudColorCompute(uv, 0.0f, depth, cubeCoord.xy, worldDir);

    vec4 srcColor = imageLoad(imageCubeEnv, cubeCoord);
    cloudColor = vec4(mix(srcColor.rgb, cloudColor.rgb, 1.0 - cloudColor.a), srcColor.a);

    imageStore(imageCubeEnv, cubeCoord, cloudColor);
}