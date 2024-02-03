#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : require

layout (set = 0, binding = 0) uniform writeonly image2D imageDepth;
layout (set = 0, binding = 1) uniform texture2D inDepth;

layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    ivec2 texSize = imageSize(imageDepth);
    ivec2 workPos = ivec2(gl_GlobalInvocationID.xy);

    if(workPos.x >= texSize.x || workPos.y >= texSize.y)
    {
        return;
    }

    const float d00 = texelFetch(inDepth, workPos * 2 + ivec2(0, 0), 0).x;
    const float d01 = texelFetch(inDepth, workPos * 2 + ivec2(0, 1), 0).x;
    const float d10 = texelFetch(inDepth, workPos * 2 + ivec2(1, 0), 0).x;
    const float d11 = texelFetch(inDepth, workPos * 2 + ivec2(1, 1), 0).x;

    float minD = min(min(min(d00, d01), d10), d11);
    float maxD = max(max(max(d00, d01), d10), d11);

    ivec2 position = workPos % 2;

	int index = position.x + position.y;
	float z = index == 1 ? minD : maxD;

    imageStore(imageDepth, workPos, vec4(z));
}