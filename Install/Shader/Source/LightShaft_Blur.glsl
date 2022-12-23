#version 460

layout (set = 0, binding = 0, rgba16f) uniform image2D outColor;
layout (set = 0, binding = 1) uniform sampler2D inputColorSampler;

#include "lightshaft_common.glsl"

layout(push_constant) uniform block
{
	LightShaftPushConstant pushConstant;
};

#define NUM_SAMPLES 12

layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    uvec2 pos = gl_GlobalInvocationID.xy;
    ivec2 imageSize = imageSize(outColor);

    if(gl_GlobalInvocationID.x < uint(imageSize.x) && gl_GlobalInvocationID.y < uint(imageSize.y))
    {
        vec2 texelSize = vec2(1.0f) / vec2(imageSize);
        vec2 uv = (vec2(pos) + vec2(0.5)) * texelSize;
        vec4 outColorFinal = vec4(0.0f, 0.0f, 0.0f, 1.0f);

        vec4 blurredValues = vec4(0);

        // Scale the UVs so that the blur will be the same pixel distance in x and y
        vec2 aspectCorrectedUV = uv * pushConstant.aspectRatioAndInvAspectRatio.zw;

        // Increase the blur distance exponentially in each pass
        float passScale = pow(.4f * NUM_SAMPLES, pushConstant.radialBlurParameters.z);

        vec2 aspectCorrectedBlurVector = (pushConstant.textureSpaceBlurOrigin.xy - aspectCorrectedUV)
            // Prevent reading past the light position
            * min(pushConstant.radialBlurParameters.y * passScale, 1);

        for (int sampleIndex = 0; sampleIndex < NUM_SAMPLES; sampleIndex++)
        {
            vec2 sampleUVs = (aspectCorrectedUV + aspectCorrectedBlurVector * sampleIndex / float(NUM_SAMPLES)) * pushConstant.aspectRatioAndInvAspectRatio.xy;
            vec2 clampedUVs = clamp(sampleUVs, vec2(0.0f), vec2(1.0f));
            vec4 sampleValue = texture(inputColorSampler, clampedUVs);
            blurredValues += sampleValue;
        }

        outColorFinal = blurredValues / float(NUM_SAMPLES);
        imageStore(outColor, ivec2(gl_GlobalInvocationID.xy), outColorFinal);
    }
}