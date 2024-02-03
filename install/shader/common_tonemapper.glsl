#ifndef TONE_COMMON_GLSL
#define TONE_COMMON_GLSL

#include "aces.glsl"
#include "common_shader.glsl"

vec3 filmToneMap(
	vec3 colorAP1, 
	float filmSlope, 
	float filmToe, 
	float filmShoulder, 
	float filmBlackClip, 
	float filmWhiteClip,
	float filmPreDesaturate,
	float filmPostDesaturate,
	float filmRedModifier,
	float filmGlowScale);
vec3 uchimuraTonemapper(vec3 color, float P, float a, float m, float l0, float c, float b, float S0, float S1, float CP);

vec3 tonemapperAp1(vec3 acescg, const in PerFrameData frameData)
{
    vec3 toneAp1;
    
    if(frameData.postprocessing.tonemapper_type == ETonemapperType_GT)
    {
        toneAp1 = uchimuraTonemapper(acescg,
            frameData.postprocessing.tonemapper_P,
            frameData.postprocessing.tonemapper_a,
            frameData.postprocessing.tonemapper_m,
            frameData.postprocessing.tonemapper_l0,
            frameData.postprocessing.tonemapper_c,
            frameData.postprocessing.tonemapper_b,
            frameData.postprocessing.tonemapper_S0,
            frameData.postprocessing.tonemapper_S1,
            frameData.postprocessing.tonemapper_CP);
    }
    else if(frameData.postprocessing.tonemapper_type == ETonemapperType_FilmACES)
    {
        toneAp1 = filmToneMap(acescg,
            frameData.postprocessing.tonemapper_filmACESSlope,
            frameData.postprocessing.tonemapper_filmACESToe,
            frameData.postprocessing.tonemapper_filmACESShoulder,
            frameData.postprocessing.tonemapper_filmACESBlackClip,
            frameData.postprocessing.tonemapper_filmACESWhiteClip,
			frameData.postprocessing.tonemapper_filmACESPreDesaturate,
			frameData.postprocessing.tonemapper_filmACESPostDesaturate,
			frameData.postprocessing.tonemapper_filmACESRedModifier,
			frameData.postprocessing.tonemapper_filmACESGlowScale
        );
    }

    return toneAp1;
}


// Customize film tonemapper.
vec3 filmToneMap(
	vec3 colorAP1, 
	float filmSlope, 
	float filmToe, 
	float filmShoulder, 
	float filmBlackClip, 
	float filmWhiteClip,
	float filmPreDesaturate,
	float filmPostDesaturate,
	float filmRedModifier,
	float filmGlowScale) 
{
	vec3 colorAP0 = colorAP1 *  AP1_2_XYZ_MAT * XYZ_2_AP0_MAT;

	// "Glow" module constants
	const float RRT_GLOW_GAIN = 0.05;
	const float RRT_GLOW_MID = 0.08;
	float saturation = rgb_2_saturation(colorAP0);
	float ycIn = rgb_2_yc(colorAP0, 1.75);
	float s = sigmoid_shaper((saturation - 0.4) / 0.2);
	float addedGlow = 1 + glow_fwd(ycIn, RRT_GLOW_GAIN * s, RRT_GLOW_MID) * filmGlowScale;
	colorAP0 *= addedGlow;

	// --- Red modifier --- //
	const float RRT_RED_SCALE = 0.82;
	const float RRT_RED_PIVOT = 0.03;
	const float RRT_RED_HUE = 0;
	const float RRT_RED_WIDTH = 135;
	float hue = rgb_2_hue(colorAP0);
	float centeredHue = center_hue(hue, RRT_RED_HUE);
	float hueWeight = smoothstep(0, 1, 1 - abs(2 * centeredHue / RRT_RED_WIDTH));
    hueWeight = hueWeight * hueWeight;
	colorAP0.r += mix(0.0, hueWeight * saturation * (RRT_RED_PIVOT - colorAP0.r) * (1.0 - RRT_RED_SCALE), filmRedModifier);

	// Use ACEScg primaries as working space
	vec3 workingColor = colorAP0 * AP0_2_AP1_MAT;
	workingColor = max(vec3(0), workingColor);

	// Pre desaturate
	workingColor = mix(vec3(dot(workingColor, AP1_RGB2Y)), workingColor, filmPreDesaturate);

	const float toeScale	    = 1.0 + filmBlackClip - filmToe;
	const float shoulderScale	= 1.0 + filmWhiteClip - filmShoulder;
	
	const float inMatch = 0.18;
	const float outMatch = 0.18;

	float toeMatch;
	if(filmToe > 0.8)
	{
		// 0.18 will be on straight segment
		toeMatch = (1 - filmToe  - outMatch) / filmSlope + log10(inMatch);
	}
	else
	{
		// 0.18 will be on toe segment

		// Solve for toeMatch such that input of inMatch gives output of outMatch.
		const float bt = (outMatch + filmBlackClip) / toeScale - 1;
		toeMatch = log10(inMatch) - 0.5 * log((1+bt) / (1-bt)) * (toeScale / filmSlope);
	}

	float straightMatch = (1.0 - filmToe) / filmSlope - toeMatch;
	float shoulderMatch = filmShoulder / filmSlope - straightMatch;
	
	vec3 logColor = log10(workingColor);
	vec3 straightColor = filmSlope * (logColor + straightMatch);
	
	vec3 toeColor		= (    - filmBlackClip ) + (2.0 *      toeScale) / (1.0 + exp( (-2.0 * filmSlope /      toeScale) * (logColor -      toeMatch)));
	vec3 shoulderColor	= (1.0 + filmWhiteClip ) - (2.0 * shoulderScale) / (1.0 + exp( ( 2.0 * filmSlope / shoulderScale) * (logColor - shoulderMatch)));

	toeColor.x		= logColor.x <      toeMatch ?      toeColor.x : straightColor.x;
    toeColor.y		= logColor.y <      toeMatch ?      toeColor.y : straightColor.y;
    toeColor.z		= logColor.z <      toeMatch ?      toeColor.z : straightColor.z;

	shoulderColor.x	= logColor.x > shoulderMatch ? shoulderColor.x : straightColor.x;
	shoulderColor.y	= logColor.y > shoulderMatch ? shoulderColor.y : straightColor.y;
    shoulderColor.z	= logColor.z > shoulderMatch ? shoulderColor.z : straightColor.z;

	vec3 t = saturate((logColor - toeMatch) / (shoulderMatch - toeMatch));
	t = shoulderMatch < toeMatch ? 1 - t : t;
	t = (3.0 - 2.0 * t) * t * t;

	vec3 toneColor = mix(toeColor, shoulderColor, t);

	// Post desaturate
	toneColor = mix(vec3(dot(vec3(toneColor), AP1_RGB2Y)), toneColor, filmPostDesaturate);

	// Returning positive AP1 values
	return max(vec3(0), toneColor);
}


// Uchimura 2017, "HDR theory and practice"
// Math: https://www.desmos.com/calculator/gslcdxvipg
// Source: https://www.slideshare.net/nikuque/hdr-theory-and-practicce-jp
float uchimuraTonemapperComponent(float x, float P, float a, float m, float l0, float c, float b, float S0, float S1, float CP) 
{
    float w0 = 1.0 - smoothstep(0.0, m, x);
    float w2 = step(m + l0, x);
    float w1 = 1.0 - w0 - w2;
    float T  = m * pow(x / m, c) + b;
    float S  = P - (P - S1) * exp(CP * (x - S0));
    float L  = m + a * (x - m);

    return T * w0 + L * w1 + S * w2;
}

vec3 uchimuraTonemapper(vec3 color, float P, float a, float m, float l0, float c, float b, float S0, float S1, float CP)
{
    vec3 result;

    result.x = uchimuraTonemapperComponent(color.x, P, a, m, l0, c, b, S0, S1, CP);
    result.y = uchimuraTonemapperComponent(color.y, P, a, m, l0, c, b, S0, S1, CP);
    result.z = uchimuraTonemapperComponent(color.z, P, a, m, l0, c, b, S0, S1, CP);

    return result;
}

// Gamma curve encode to srgb.
vec3 encodeSRGB(vec3 linearRGB)
{
    // Most PC Monitor is 2.2 Gamma, maybe this function is enough.
    // return pow(linearRGB, vec3(1.0 / 2.2));

    // TV encode Rec709 encode.
    vec3 a = 12.92 * linearRGB;
    vec3 b = 1.055 * pow(linearRGB, vec3(1.0 / 2.4)) - 0.055;
    vec3 c = step(vec3(0.0031308), linearRGB);
    return mix(a, b, c);
}

#endif 