#ifndef SHARED_ACES_GLSL
#define SHARED_ACES_GLSL

/****************************************************************************************

	ACES: Academy Color Encoding System
	https://github.com/ampas/aces-dev/tree/v1.0

	License Terms for Academy Color Encoding System Components

	Academy Color Encoding System (ACES) software and tools are provided by the Academy under
	the following terms and conditions: A worldwide, royalty-free, non-exclusive right to copy, modify, create
	derivatives, and use, in source and binary forms, is hereby granted, subject to acceptance of this license.

	Copyright © 2013 Academy of Motion Picture Arts and Sciences (A.M.P.A.S.). Portions contributed by
	others as indicated. All rights reserved.

	Performance of any of the aforementioned acts indicates acceptance to be bound by the following
	terms and conditions:

	 *	Copies of source code, in whole or in part, must retain the above copyright
		notice, this list of conditions and the Disclaimer of Warranty.
	 *	Use in binary form must retain the above copyright notice, this list of
		conditions and the Disclaimer of Warranty in the documentation and/or other
		materials provided with the distribution.
	 *	Nothing in this license shall be deemed to grant any rights to trademarks,
		copyrights, patents, trade secrets or any other intellectual property of
		A.M.P.A.S. or any contributors, except as expressly stated herein.
	 *	Neither the name "A.M.P.A.S." nor the name of any other contributors to this
		software may be used to endorse or promote products derivative of or based on
		this software without express prior written permission of A.M.P.A.S. or the
		contributors, as appropriate.

	This license shall be construed pursuant to the laws of the State of California,
	and any disputes related thereto shall be subject to the jurisdiction of the courts therein.

	Disclaimer of Warranty: THIS SOFTWARE IS PROVIDED BY A.M.P.A.S. AND CONTRIBUTORS "AS
	IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
	IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
	NON-INFRINGEMENT ARE DISCLAIMED. IN NO EVENT SHALL A.M.P.A.S., OR ANY
	CONTRIBUTORS OR DISTRIBUTORS, BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
	SPECIAL, EXEMPLARY, RESITUTIONARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
	NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
	DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
	OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
	NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
	EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

	WITHOUT LIMITING THE GENERALITY OF THE FOREGOING, THE ACADEMY SPECIFICALLY
	DISCLAIMS ANY REPRESENTATIONS OR WARRANTIES WHATSOEVER RELATED TO PATENT OR
	OTHER INTELLECTUAL PROPERTY RIGHTS IN THE ACADEMY COLOR ENCODING SYSTEM, OR
	APPLICATIONS THEREOF, HELD BY PARTIES OTHER THAN A.M.P.A.S.,WHETHER DISCLOSED
	OR UNDISCLOSED.

****************************************************************************************/

// Linear sRGB gamut convert to XYZ gamut
const mat3 sRGB_2_XYZ_MAT = mat3
(
	vec3(0.4124564, 0.3575761, 0.1804375),
	vec3(0.2126729, 0.7151522, 0.0721750),
	vec3(0.0193339, 0.1191920, 0.9503041)
);

// XYZ gamut to linear sRGB gamut
const mat3 XYZ_2_sRGB_MAT = mat3
( 
	vec3(3.2409699419, -1.5373831776, -0.4986107603),
	vec3(-0.9692436363,  1.8759675015,  0.0415550574),
	vec3(0.0556300797, -0.2039769589,  1.0569715142)
);

// D65 to D60 White Point
const mat3 D65_2_D60_CAT = mat3
( 
	vec3(1.01303, 0.00610531, -0.014971),
	vec3(0.00769823, 0.998165, -0.00503203),
	vec3(-0.00284131, 0.00468516, 0.924507)
);

// D60 to D65 White Point
const mat3 D60_2_D65_CAT = mat3
( 
	vec3(0.987224,   -0.00611327, 0.0159533),
	vec3(-0.00759836,  1.00186,    0.00533002),
	vec3(0.00307257, -0.00509595, 1.08168)
);

// XYZ to ACEScg gamut
const mat3 XYZ_2_AP0_MAT = mat3
( 
	vec3(1.0498110175, 0.0000000000,-0.0000974845),
	vec3(-0.4959030231, 1.3733130458, 0.0982400361),
	vec3(0.0000000000, 0.0000000000, 0.9912520182)
);

// ACEScg to XYZ gamut
const mat3 AP0_2_XYZ_MAT = mat3
( 
	vec3(0.9525523959, 0.0000000000, 0.0000936786),
	vec3(0.3439664498, 0.7281660966,-0.0721325464),
	vec3(0.0000000000, 0.0000000000, 1.0088251844)
);

// XYZ to ACEStoning gamut
const mat3 XYZ_2_AP1_MAT = mat3
( 
	vec3(1.6410233797, -0.3248032942, -0.2364246952),
	vec3(-0.6636628587,  1.6153315917,  0.0167563477),
	vec3(0.0117218943, -0.0082844420,  0.9883948585)
);

// ACEStoning to XYZ gamut
const mat3 AP1_2_XYZ_MAT = mat3
( 
	vec3(0.6624541811, 0.1340042065, 0.1561876870),
	vec3(0.2722287168, 0.6740817658, 0.0536895174),
	vec3(-0.0055746495, 0.0040607335, 1.0103391003)
);

// ACEScg to ACEStoneing gamut
const mat3 AP0_2_AP1_MAT = mat3
( 
	vec3(1.4514393161, -0.2365107469, -0.2149285693),
	vec3(-0.0765537734,  1.1762296998, -0.0996759264),
	vec3(0.0083161484, -0.0060324498,  0.9977163014)
);

// ACEStoning to ACEScg gamut
const mat3 AP1_2_AP0_MAT = mat3
( 
	vec3(0.6954522414,  0.1406786965,  0.1638690622),
	vec3(0.0447945634,  0.8596711185,  0.0955343182),
	vec3(-0.0055258826,  0.0040252103,  1.0015006723)
);

const vec3 AP1_RGB2Y = vec3(0.2722287168, 0.6740817658, 0.0536895174);
const mat3 sRGB_2_AP0 = (sRGB_2_XYZ_MAT * D65_2_D60_CAT) * XYZ_2_AP0_MAT;

// L*a*b*/CIELAB
// CIELAB was developed in 1976 in an attempt to make a perceptually uniform color space.
// While it doesn't always do a great job of this (especially in the deep blues), it is still frequently used.
float XYZ_TO_LAB_F(float x) 
{
    //          (24/116)^3                         1/(3*(6/29)^2)     4/29
    return x > 0.00885645167 ? pow(x, 0.333333333) : 7.78703703704 * x + 0.13793103448;
}

const vec3 D65_WHITE = vec3(0.95045592705, 1.0, 1.08905775076);
//                          0.3457/0.3585  1.0  (1.0-0.3457-0.3585)/0.3585
const vec3 D50_WHITE = vec3(0.96429567643, 1.0, 0.82510460251);

vec3 XYZ_TO_LAB(vec3 xyz) 
{
	vec3 WHITE = D65_WHITE;
	
    vec3 xyz_scaled = xyz / WHITE;
    xyz_scaled = vec3(
        XYZ_TO_LAB_F(xyz_scaled.x),
        XYZ_TO_LAB_F(xyz_scaled.y),
        XYZ_TO_LAB_F(xyz_scaled.z)
    );
    return vec3(
        (116.0 * xyz_scaled.y) - 16.0,
        500.0 * (xyz_scaled.x - xyz_scaled.y),
        200.0 * (xyz_scaled.y - xyz_scaled.z)
    );
}

// Linear srgb to cie lab
vec3 sRGB2LAB(vec3 c)
{
	vec3 xyz = c * sRGB_2_XYZ_MAT;
	return XYZ_TO_LAB(xyz);
}

float log10(float x) 
{
	const float a = 1.0 / log(10.0);
	return log(x) * a;
}

vec3 log10(vec3 x) 
{
	const float a = 1.0 / log(10.0);
	return log(x) * a;
}

// Sigmoid function in the range 0 to 1 spanning -2 to +2.
float sigmoid_shaper(float x) 
{ 
	float t = max(1.0 - abs(0.5 * x), 0.0);
	float y = 1.0 + sign(x) * (1.0 - t * t);

	return 0.5 * y;
}

float rgb_2_saturation(vec3 rgb) 
{
	float minrgb = min(min(rgb.r, rgb.g), rgb.b);
	float maxrgb = max(max(rgb.r, rgb.g), rgb.b);
	return (max(maxrgb, 1e-10) - max(minrgb, 1e-10)) / max(maxrgb, 1e-2);
}

// Converts RGB to a luminance proxy, here called YC. YC is ~ Y + K * Chroma.
float rgb_2_yc(vec3 rgb, float ycRadiusWeight) 
{ 
	float chroma = sqrt(rgb.b * (rgb.b - rgb.g) + rgb.g * (rgb.g - rgb.r) + rgb.r * (rgb.r - rgb.b));

	return (rgb.b + rgb.g + rgb.r + ycRadiusWeight * chroma) / 3.0;
}

float glow_fwd(float ycIn, float glowGainIn, float glowMid) 
{
	float glowGainOut;

	if (ycIn <= 2.0 / 3.0 * glowMid) 
    {
		glowGainOut = glowGainIn;
	} 
    else if ( ycIn >= 2.0 * glowMid) 
    {
		glowGainOut = 0;
	} 
    else 
    {
		glowGainOut = glowGainIn * (glowMid / ycIn - 0.5);
	}

	return glowGainOut;
}

// Returns a geometric hue angle in degrees (0-360) based on RGB values.
float rgb_2_hue(vec3 rgb) 
{ 
	float hue;
	if (rgb[0] == rgb[1] && rgb[1] == rgb[2]) 
    { 
        // For neutral colors, hue is undefined and the function will return a quiet NaN value.
		hue = 0;
	} 
    else 
    {
        // flip due to opengl spec compared to hlsl
		hue = (180.0 / kPI) * atan(2.0 * rgb[0] - rgb[1] - rgb[2], sqrt(3.0) * (rgb[1] - rgb[2])); 
	}

	if (hue < 0.0)
		hue = hue + 360.0;

	return clamp(hue, 0.0, 360.0);
}

float center_hue(float hue, float centerH) 
{
	float hueCentered = hue - centerH;

	if (hueCentered < -180.0) 
    {
		hueCentered += 360.0;
	} 
    else if (hueCentered > 180.0) 
    {
		hueCentered -= 360.0;
	}

	return hueCentered;
}

// Transformations between CIE XYZ tristimulus values and CIE x,y
// chromaticity coordinates
vec3 XYZ_2_xyY( vec3 XYZ ) 
{
	float divisor = max(XYZ[0] + XYZ[1] + XYZ[2], 1e-10);

	vec3 xyY    = XYZ.xyy;
	     xyY.rg = XYZ.rg / divisor;

	return xyY;
}

vec3 xyY_2_XYZ(vec3 xyY) 
{
	vec3 XYZ   = vec3(0.0);
	     XYZ.r = xyY.r * xyY.b / max(xyY.g, 1e-10);
	     XYZ.g = xyY.b;
	     XYZ.b = (1.0 - xyY.r - xyY.g) * xyY.b / max(xyY.g, 1e-10);

	return XYZ;
}

/*******************************************************************************
 - ACES Real, slow, never actually use in production.
 ******************************************************************************/
float cubic_basis_shaper(float x, float w) 
{
	//return Square( smoothstep( 0, 1, 1 - abs( 2 * x/w ) ) );

	const mat4 M = mat4(
		vec4(-1.0 / 6.0,  3.0 / 6.0, -3.0 / 6.0,  1.0 / 6.0),
		vec4(3.0 / 6.0, -6.0 / 6.0,  3.0 / 6.0,  0.0 / 6.0),
		vec4(-3.0 / 6.0,  0.0 / 6.0,  3.0 / 6.0,  0.0 / 6.0),
		vec4(1.0 / 6.0,  4.0 / 6.0,  1.0 / 6.0,  0.0 / 6.0)
	);

	float knots[5] = float[5](-0.5 * w, -0.25 * w, 0, 0.25 * w, 0.5 * w);

	float y = 0;
	if ((x > knots[0]) && (x < knots[4])) 
    {
		float knot_coord = (x - knots[0]) * 4.0 / w;
		int j = int(knot_coord);
		float t = knot_coord - j;

		vec4 monomials = vec4(t * t * t, t * t, t, 1.0);

		// (if/else structure required for compatibility with CTL < v1.5.)
		if (j == 3) 
        {
			y = monomials[0] * M[0][0] + monomials[1] * M[1][0] +
				monomials[2] * M[2][0] + monomials[3] * M[3][0];
		} 
        else if (j == 2) 
        {
			y = monomials[0] * M[0][1] + monomials[1] * M[1][1] +
				monomials[2] * M[2][1] + monomials[3] * M[3][1];
		} 
        else if (j == 1) 
        {
			y = monomials[0] * M[0][2] + monomials[1] * M[1][2] +
				monomials[2] * M[2][2] + monomials[3] * M[3][2];
		} 
        else if (j == 0) 
        {
			y = monomials[0] * M[0][3] + monomials[1] * M[1][3] +
				monomials[2] * M[2][3] + monomials[3] * M[3][3];
		} 
        else 
        {
			y = 0.0;
		}
	}

	return y * 1.5;
}

struct SegmentedSplineParams_c5 
{
	float coefsLow[6];    // coefs for B-spline between minPoint and midPoint (units of log luminance)
	float coefsHigh[6];   // coefs for B-spline between midPoint and maxPoint (units of log luminance)
	vec2 minPoint; // {luminance, luminance} linear extension below this
	vec2 midPoint; // {luminance, luminance} 
	vec2 maxPoint; // {luminance, luminance} linear extension above this
	float slopeLow;       // log-log slope of low linear extension
	float slopeHigh;      // log-log slope of high linear extension
};

struct SegmentedSplineParams_c9 
{
	float coefsLow[10];    // coefs for B-spline between minPoint and midPoint (units of log luminance)
	float coefsHigh[10];   // coefs for B-spline between midPoint and maxPoint (units of log luminance)
	float slopeLow;       // log-log slope of low linear extension
	float slopeHigh;      // log-log slope of high linear extension
};

const mat3 M = mat3
(
	 0.5, -1.0,  0.5,
	-1.0,  1.0,  0.5,
	 0.5,  0.0,  0.0
);

float segmented_spline_c5_fwd(float x) 
{
	const SegmentedSplineParams_c5 C = SegmentedSplineParams_c5
    (
		float[6] ( -4.0000000000, -4.0000000000, -3.1573765773, -0.4852499958, 1.8477324706, 1.8477324706 ),
		float[6] ( -0.7185482425, 2.0810307172, 3.6681241237, 4.0000000000, 4.0000000000, 4.0000000000 ),
		vec2(0.18*exp2(-15.0), 0.0001),
		vec2(0.18,              4.8),
		vec2(0.18*exp2(18.0),  10000.),
		0.0,
		0.0
	);

	const int N_KNOTS_LOW = 4;
	const int N_KNOTS_HIGH = 4;

	// Check for negatives or zero before taking the log. If negative or zero,
	// set to ACESMIN.1
	float xCheck = x <= 0 ? exp2(-14.0) : x;

	float logx = log10( xCheck);
	float logy;

	if (logx <= log10(C.minPoint.x)) 
    {
		logy = logx * C.slopeLow + (log10(C.minPoint.y) - C.slopeLow * log10(C.minPoint.x));
	} 
    else if ((logx > log10(C.minPoint.x)) && (logx < log10(C.midPoint.x))) {

		float knot_coord = (N_KNOTS_LOW-1) * (logx-log10(C.minPoint.x))/(log10(C.midPoint.x)-log10(C.minPoint.x));
		int j = int(knot_coord);
		float t = knot_coord - float(j);

		vec3 cf = vec3( C.coefsLow[ j], C.coefsLow[ j + 1], C.coefsLow[ j + 2]);
    
		vec3 monomials = vec3(t * t, t, 1.0);
		logy = dot( monomials, M * cf);
	} 
    else if ((logx >= log10(C.midPoint.x)) && (logx < log10(C.maxPoint.x))) 
    {
		float knot_coord = (N_KNOTS_HIGH - 1) * (logx - log10(C.midPoint.x)) / (log10(C.maxPoint.x) - log10(C.midPoint.x));
		int j = int(knot_coord);
		float t = knot_coord - float(j);

		vec3 cf = vec3(C.coefsHigh[j], C.coefsHigh[j + 1], C.coefsHigh[j + 2]);
		vec3 monomials = vec3(t * t, t, 1.0);
		
		logy = dot(monomials, M * cf);
	} 
    else 
    {
		logy = logx * C.slopeHigh + (log10(C.maxPoint.y) - C.slopeHigh * log10(C.maxPoint.x));
	}

	return pow(10.0, logy);
}

float segmented_spline_c9_fwd( float x, const SegmentedSplineParams_c9 C, const mat3x2 toningPoints) 
{
	const int N_KNOTS_LOW = 8;
	const int N_KNOTS_HIGH = 8;

	// Check for negatives or zero before taking the log. If negative or zero,
	// set to OCESMIN.
	float xCheck = x <= 0 ? 1e-4 : x;

	vec2 minPoint = toningPoints[0];
	vec2 midPoint = toningPoints[1];
	vec2 maxPoint = toningPoints[2];

	float logx = log10(xCheck);
	float logy;

	if (logx <= log10(minPoint.x)) {
		logy = logx * C.slopeLow + (log10(minPoint.y) - C.slopeLow * log10(minPoint.x));
	} else if ((logx > log10(minPoint.x)) && (logx < log10(midPoint.x))) {
		float knot_coord = (N_KNOTS_LOW - 1) * (logx - log10(minPoint.x)) / (log10(midPoint.x) - log10(minPoint.x));
		int j = int(knot_coord);
		float t = knot_coord - float(j);

		vec3 cf = vec3(C.coefsLow[j], C.coefsLow[j + 1], C.coefsLow[j + 2]);
		vec3 monomials = vec3(t * t, t, 1.0);
		
		logy = dot(monomials, M * cf);
	} else if ((logx >= log10(midPoint.x)) && (logx < log10(maxPoint.x))) {
		float knot_coord = (N_KNOTS_HIGH - 1) * (logx - log10(midPoint.x)) / (log10(maxPoint.x) - log10(midPoint.x));
		int j = int(knot_coord);
		float t = knot_coord - float(j);

		vec3 cf = vec3(C.coefsHigh[j], C.coefsHigh[j + 1], C.coefsHigh[j + 2]);
		vec3 monomials = vec3(t * t, t, 1.0);
		
		logy = dot(monomials, M * cf);
	} else {
		logy = logx * C.slopeHigh + (log10(maxPoint.y) - C.slopeHigh * log10(maxPoint.x));
	}

	return pow(10.0, logy);
}

vec3 RRT(vec3 aces) 
{
	// "Glow" module constants. No idea what the actual fuck this even means.
	const float RRT_GLOW_GAIN = 0.05;
	const float RRT_GLOW_MID = 0.08;

	float saturation = rgb_2_saturation(aces);
	float ycIn = rgb_2_yc(aces, 1.75);
	float s = sigmoid_shaper((saturation - 0.4) / 0.2);
	float addedGlow = 1.0 + glow_fwd(ycIn, RRT_GLOW_GAIN * s, RRT_GLOW_MID);
	aces *= addedGlow;

	// --- Red modifier --- //
	const float RRT_RED_SCALE = 0.82;
	const float RRT_RED_PIVOT = 0.03;
	const float RRT_RED_HUE = 0;
	const float RRT_RED_WIDTH = 135;
	float hue = rgb_2_hue(aces);
	float centeredHue = center_hue(hue, RRT_RED_HUE);
	float hueWeight = cubic_basis_shaper(centeredHue, RRT_RED_WIDTH);

	aces.r += hueWeight * saturation * (RRT_RED_PIVOT - aces.r) * (1.0 - RRT_RED_SCALE);

	// --- ACES to RGB rendering space --- //
	aces = clamp(aces, 0, 65535.0);  // avoids saturated negative colors from becoming positive in the matrix
	vec3 rgbPre = aces * AP0_2_AP1_MAT;
	rgbPre = clamp(rgbPre, 0.0, 65535.0);

	// --- Global desaturation --- //
	const float RRT_SAT_FACTOR = 0.96;
	rgbPre = mix(vec3(dot(rgbPre, AP1_RGB2Y)), rgbPre, vec3(RRT_SAT_FACTOR));

	// --- Apply the tonescale independently in rendering-space RGB --- //
	vec3 rgbPost = vec3(0.0);
	rgbPost.r = segmented_spline_c5_fwd(rgbPre.r);
	rgbPost.g = segmented_spline_c5_fwd(rgbPre.g);
	rgbPost.b = segmented_spline_c5_fwd(rgbPre.b);

	// --- RGB rendering space to OCES --- //
	return rgbPost * AP1_2_AP0_MAT;
}

vec3 Y_2_linCV(vec3 Y, float Ymax, float Ymin)  
{
	return (Y - Ymin) / (Ymax - Ymin);
}

vec3 darkSurround_to_dimSurround(vec3 linearCV) 
{
    const float DIM_SURROUND_GAMMA = 0.9811;

	vec3 XYZ = linearCV * AP1_2_XYZ_MAT;

	vec3 xyY = XYZ_2_xyY(XYZ);
	xyY[2] = clamp(xyY[2], 0, 65535.0);
	xyY[2] = pow(xyY[2], DIM_SURROUND_GAMMA);
	XYZ = xyY_2_XYZ(xyY);

	return XYZ * XYZ_2_AP1_MAT;
}

vec3 ODT_sRGB_D65(vec3 oces) 
{
	// OCES to RGB rendering space
	vec3 rgbPre = oces * AP0_2_AP1_MAT;

	const SegmentedSplineParams_c9 ODT_48nits = SegmentedSplineParams_c9
    (
		float[10] ( -1.6989700043, -1.6989700043, -1.4779000000, -1.2291000000, -0.8648000000, -0.4480000000, 0.0051800000, 0.4511080334, 0.9113744414, 0.9113744414 ), // coefsLow[10]
		float[10] ( 0.5154386965, 0.8470437783, 1.1358000000, 1.3802000000, 1.5197000000, 1.5985000000, 1.6467000000, 1.6746091357, 1.6878733390, 1.6878733390 ), // coefsHigh[10]
		0.0, // slopeLow
		0.04 // slopeHigh
	);

	vec3 splines = vec3(0.0);
	splines.r = segmented_spline_c5_fwd(0.18 * exp2(-6.5)); // vec3(minPoint, midPoint, MaxPoint)
	splines.g = segmented_spline_c5_fwd(0.18);
	splines.b = segmented_spline_c5_fwd(0.18 * exp2(6.5));

	mat3x2 toningPoints = mat3x2(
		splines.x, 0.02,
		splines.y, 4.8,
		splines.z, 48.0
	);

	// Apply the tonescale independently in rendering-space RGB
	vec3 rgbPost = vec3(0.0);
	rgbPost.r = segmented_spline_c9_fwd(rgbPre.r, ODT_48nits, toningPoints);
	rgbPost.g = segmented_spline_c9_fwd(rgbPre.g, ODT_48nits, toningPoints);
	rgbPost.b = segmented_spline_c9_fwd(rgbPre.b, ODT_48nits, toningPoints);

	// Target white and black points for cinema system tonescale
	const float CINEMA_WHITE = 48.0;
	const float CINEMA_BLACK = 0.02; // CINEMA_WHITE / 2400.

	// Scale luminance to linear code value
	vec3 linearCV = Y_2_linCV(rgbPost, CINEMA_WHITE, CINEMA_BLACK);

	// Apply gamma adjustment to compensate for dim surround
	//linearCV = darkSurround_to_dimSurround(linearCV);

	// Apply desaturation to compensate for luminance difference
	const float ODT_SAT_FACTOR = 0.93;
	linearCV = mix(vec3(dot(linearCV, AP1_RGB2Y)), linearCV, vec3(ODT_SAT_FACTOR));

	// Convert to display primary encoding
	// Rendering space RGB to XYZ
	vec3 XYZ = linearCV * AP1_2_XYZ_MAT;

	// Apply CAT from ACES white point to assumed observer adapted white point
	XYZ = XYZ * D60_2_D65_CAT;

	// CIE XYZ to display primaries
	linearCV = XYZ * XYZ_2_sRGB_MAT;

	return clamp(linearCV, 0.0, 1.0);
}

vec3 ACESOutputTransformsAP1(vec3 ap1) 
{
	vec3 oces = RRT(ap1 * AP1_2_AP0_MAT);
	vec3 OutputReferredLinearsRGBColor =  ODT_sRGB_D65(oces);
	return OutputReferredLinearsRGBColor;
}

#endif