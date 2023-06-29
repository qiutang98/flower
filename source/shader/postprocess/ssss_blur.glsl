#version 450

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable



#include "post_common.glsl"

layout (set = 0, binding = 0) uniform texture2D diffuseTexture;
layout (set = 0, binding = 1) uniform texture2D depthTexture;
layout (set = 0, binding = 2, rgba16f)  uniform image2D resultImage;
layout (set = 0, binding = 3) uniform UniformFrameData { PerFrameData frameData; };


#define SHARED_SAMPLER_SET 1
#include "../common/shared_sampler.glsl"

layout (push_constant) uniform PushConsts 
{  
    vec2  ssss_direction;
    float ssss_width;
    float ssss_maxScale;
    int finalPass;
};

#define kSampleKernelNum 25
#define kSampleRange 3.0
vec4 kSampleKernel[] = 
{
vec4(0.022094,0.902209,0.990221,0),
vec4(0.00202874,0.000202874,2.02874e-05,-3),
vec4(0.00695426,0.000695426,6.95426e-05,-2.52083),
vec4(0.0104242,0.00104242,0.000104242,-2.08333),
vec4(0.0146037,0.00146037,0.000146037,-1.6875),
vec4(0.0196644,0.00196644,0.000196644,-1.33333),
vec4(0.0267701,0.00267701,0.000267701,-1.02083),
vec4(0.0373416,0.00373416,0.000373416,-0.75),
vec4(0.0549254,0.00549254,0.000549254,-0.520833),
vec4(0.0854524,0.00854524,0.000854524,-0.333333),
vec4(0.102831,0.0102831,0.00102831,-0.1875),
vec4(0.0839133,0.00839133,0.000839133,-0.0833333),
vec4(0.0440441,0.00440441,0.000440441,-0.0208333),
vec4(0.0440441,0.00440441,0.000440441,0.0208333),
vec4(0.0839133,0.00839133,0.000839133,0.0833333),
vec4(0.102831,0.0102831,0.00102831,0.1875),
vec4(0.0854524,0.00854524,0.000854524,0.333333),
vec4(0.0549254,0.00549254,0.000549254,0.520833),
vec4(0.0373416,0.00373416,0.000373416,0.75),
vec4(0.0267701,0.00267701,0.000267701,1.02083),
vec4(0.0196644,0.00196644,0.000196644,1.33333),
vec4(0.0146037,0.00146037,0.000146037,1.6875),
vec4(0.0104242,0.00104242,0.000104242,2.08333),
vec4(0.00695426,0.000695426,6.95426e-05,2.52083),
vec4(0.00202874,0.000202874,2.02874e-05,3),
};

layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    ivec2 colorSize = imageSize(resultImage);

    uvec2 groupThreadId = remap8x8(gl_LocalInvocationIndex);
    uvec2 dispatchId = groupThreadId + gl_WorkGroupID.xy * 8;
    ivec2 workPos = ivec2(dispatchId);
    if(workPos.x >= colorSize.x || workPos.y >= colorSize.y)
    {
        return;
    }
    const vec2 texelSize = vec2(1.0) / vec2(colorSize);
    const vec2 uv = (vec2(workPos) + vec2(0.5f)) * texelSize;

    bool bFinalPass = finalPass != 0;
    vec4 srcColor = vec4(0.0);
    if(bFinalPass)
    {
        srcColor += imageLoad(resultImage, workPos);
    }

	vec4 colorM = texture(sampler2D(diffuseTexture, pointClampEdgeSampler), uv);

    if(colorM.a <= 0.0f)
    {
        imageStore(resultImage, workPos, colorM + srcColor);
        return;
    }

	float depthM = linearizeDepth(texture(sampler2D(depthTexture,  pointClampEdgeSampler), uv).x, frameData);

    float distanceToProjectionWindow = 1.0 / tan(0.5 * frameData.camInfo.x);
    float scale = ssss_maxScale * distanceToProjectionWindow / depthM;

    vec2 finalStep = ssss_width * scale * ssss_direction;
    finalStep *= colorM.a; // Modulate it using the alpha channel.
    finalStep *= 1.0 / kSampleRange; // Divide by 3 as the kernels range from -3 to 3.

	// accumulate the center sample:
	vec4 colorBlurred = colorM;
	colorBlurred.rgb *= kSampleKernel[0].rgb;
	
	for (int i = 1; i < kSampleKernelNum; ++i)
	{
		vec2 sampleUv = uv + kSampleKernel[i].a * finalStep;

		vec4 color = texture(sampler2D(diffuseTexture, pointClampEdgeSampler), sampleUv);

#if 1
		float depth = linearizeDepth(texture(sampler2D(depthTexture, pointClampEdgeSampler), sampleUv).x, frameData);
        float s = saturate(300.0f * distanceToProjectionWindow * ssss_width * abs(depthM - depth));
        color.rgb = mix(color.rgb, colorM.rgb, s);
#endif

		colorBlurred.rgb += kSampleKernel[i].rgb * color.rgb;
	}

	imageStore(resultImage, workPos, colorBlurred + srcColor);
}