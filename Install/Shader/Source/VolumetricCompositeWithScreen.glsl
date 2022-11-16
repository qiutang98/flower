
#version 460

#extension GL_GOOGLE_include_directive : enable

#include "VolumetricCloudCommon.glsl"

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
    vec4 cloudColor = texture(sampler2D(inCloudRenderTexture, linearClampEdgeSampler), uv);

    vec3 result = srcColor.rgb * cloudColor.a + cloudColor.rgb;
	imageStore(imageHdrSceneColor, workPos, vec4(result, 1.0));
}