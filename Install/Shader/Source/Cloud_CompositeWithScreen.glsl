
#version 460

/*
** Physical based render code, develop by engineer: qiutanguu.
*/

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable
#include "Cloud_Common.glsl"

layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    ivec2 texSize = imageSize(imageHdrSceneColor);
    ivec2 workPos = ivec2(gl_GlobalInvocationID.xy);

    if(workPos.x >= texSize.x || workPos.y >= texSize.y)
    {
        return;
    }

    const vec2 uv = (vec2(workPos) + vec2(0.5f)) / vec2(texSize);

    vec4 srcColor = imageLoad(imageHdrSceneColor, workPos);
    float sceneZ = texture(sampler2D(inDepth, pointClampEdgeSampler), uv).r;

    vec4 cloudColor = texture(sampler2D(inCloudReconstructionTexture, linearClampEdgeSampler), uv);
    float cloudDepth = texture(sampler2D(inCloudDepthReconstructionTexture, linearClampEdgeSampler), uv).r;
    cloudDepth = max(1e-5f, cloudDepth); // very far cloud may be negative, use small value is enough.

    if(sceneZ <= cloudDepth) // reverse z.
    {
        vec3 result = mix(srcColor.rgb, cloudColor.rgb, 1.0 - cloudColor.a);
	    imageStore(imageHdrSceneColor, workPos, vec4(result, 1.0));

        // TODO: Sometimes the cloud is ghost and fsr can't clip it. Need to write depth and velocity. :(
        //       Need fix.
        // imageStore(imageGbufferReactiveMask, workPos, 0.5);
    }
}