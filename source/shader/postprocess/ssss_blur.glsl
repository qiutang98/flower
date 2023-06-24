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

#define kSampleKernelNum 64
#define kSampleRange 3.0
vec4 kSampleKernel[] = 
{
vec4(0.00405711,0.502029,0.950203,0.00075586),
vec4(0.000803385,0.000401693,4.01693e-05,-3),
vec4(0.00199587,0.000997936,9.97936e-05,-2.81255),
vec4(0.00240442,0.00120221,0.000120221,-2.63114),
vec4(0.00285592,0.00142796,0.000142796,-2.45578),
vec4(0.00334835,0.00167418,0.000167418,-2.28647),
vec4(0.0038806,0.0019403,0.00019403,-2.1232),
vec4(0.0044523,0.00222615,0.000222615,-1.96599),
vec4(0.00506351,0.00253176,0.000253176,-1.81481),
vec4(0.00571532,0.00285766,0.000285766,-1.66969),
vec4(0.00641306,0.00320653,0.000320653,-1.53061),
vec4(0.00717192,0.00358596,0.000358596,-1.39758),
vec4(0.00802185,0.00401093,0.000401093,-1.2706),
vec4(0.00900537,0.00450269,0.000450269,-1.14966),
vec4(0.0101645,0.00508224,0.000508224,-1.03477),
vec4(0.0115222,0.00576108,0.000576108,-0.925926),
vec4(0.0130818,0.0065409,0.00065409,-0.823129),
vec4(0.0148755,0.00743776,0.000743776,-0.726379),
vec4(0.0170504,0.0085252,0.00085252,-0.635677),
vec4(0.0198878,0.0099439,0.00099439,-0.55102),
vec4(0.023633,0.0118165,0.00118165,-0.472411),
vec4(0.0281908,0.0140954,0.00140954,-0.399849),
vec4(0.0329525,0.0164762,0.00164762,-0.333333),
vec4(0.0369572,0.0184786,0.00184786,-0.272865),
vec4(0.0392923,0.0196462,0.00196462,-0.218443),
vec4(0.0394442,0.0197221,0.00197221,-0.170068),
vec4(0.0374045,0.0187022,0.00187022,-0.12774),
vec4(0.0335457,0.0167729,0.00167729,-0.0914588),
vec4(0.0284028,0.0142014,0.00142014,-0.0612245),
vec4(0.0224871,0.0112436,0.00112436,-0.037037),
vec4(0.016185,0.00809249,0.000809249,-0.0188965),
vec4(0.00973371,0.00486685,0.000486685,-0.00680271),
vec4(0.0040571,0.00202855,0.000202855,-0.00075586),
vec4(0.00973371,0.00486685,0.000486685,0.00680274),
vec4(0.016185,0.0080925,0.00080925,0.0188965),
vec4(0.0224871,0.0112436,0.00112436,0.0370371),
vec4(0.0284028,0.0142014,0.00142014,0.0612245),
vec4(0.0335457,0.0167729,0.00167729,0.0914589),
vec4(0.0374045,0.0187022,0.00187022,0.12774),
vec4(0.0394441,0.0197221,0.00197221,0.170068),
vec4(0.0392923,0.0196462,0.00196462,0.218443),
vec4(0.0369572,0.0184786,0.00184786,0.272865),
vec4(0.0329524,0.0164762,0.00164762,0.333333),
vec4(0.0281908,0.0140954,0.00140954,0.399849),
vec4(0.023633,0.0118165,0.00118165,0.472411),
vec4(0.0198878,0.00994389,0.000994389,0.55102),
vec4(0.0170504,0.00852521,0.000852521,0.635677),
vec4(0.0148755,0.00743777,0.000743777,0.72638),
vec4(0.0130818,0.0065409,0.00065409,0.823129),
vec4(0.0115221,0.00576107,0.000576107,0.925926),
vec4(0.0101645,0.00508223,0.000508223,1.03477),
vec4(0.00900538,0.00450269,0.000450269,1.14966),
vec4(0.00802186,0.00401093,0.000401093,1.2706),
vec4(0.00717191,0.00358596,0.000358596,1.39758),
vec4(0.00641305,0.00320652,0.000320652,1.53061),
vec4(0.00571533,0.00285766,0.000285766,1.66969),
vec4(0.00506352,0.00253176,0.000253176,1.81482),
vec4(0.0044523,0.00222615,0.000222615,1.96599),
vec4(0.0038806,0.0019403,0.00019403,2.1232),
vec4(0.00334836,0.00167418,0.000167418,2.28647),
vec4(0.00285593,0.00142796,0.000142796,2.45578),
vec4(0.00240442,0.00120221,0.000120221,2.63114),
vec4(0.00199587,0.000997934,9.97934e-05,2.81255),
vec4(0.000803385,0.000401693,4.01693e-05,3),
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