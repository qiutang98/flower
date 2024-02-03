#ifndef COMMON_SHADER_GLSL
#define COMMON_SHADER_GLSL

#extension GL_EXT_samplerless_texture_functions : enable

#include "common_header.h"
#include "common_sampler.glsl"

#ifndef DEBUG_LINE_ENABLE
#define DEBUG_LINE_ENABLE 1
#endif

// 128 meter per slice.
// 8192.0 * 2.0 meter full coverage most case.
#define kDistantSkyLitMax 8192.0 * 2.0
vec2 getSkySampleDistantLitUv(const float worldHeight)
{
    return clamp(vec2(worldHeight / kDistantSkyLitMax, 0.5), 0.0, 1.0);
}

const float kPI = 3.141592653589793;
const float kLog2 = log(2.0);
const vec3  kMax111110BitsFloat3 = vec3(kMax11BitsFloat, kMax11BitsFloat, kMax10BitsFloat);

#define kCloudShadowExp 5.0

#include "aces.glsl"

const mat3 sRGB_2_AP1_MAT = (sRGB_2_XYZ_MAT * D65_2_D60_CAT) * XYZ_2_AP1_MAT;
const mat3 AP1_2_sRGB_MAT = (AP1_2_XYZ_MAT  * D60_2_D65_CAT) * XYZ_2_sRGB_MAT;

/**
 * NOTE from filament engine: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 * 
 * Transforms a texture UV to make it suitable for a render target attachment.
 *
 * In Vulkan and Metal, texture coords are Y-down but in OpenGL they are Y-up. This wrapper function
 * accounts for these differences. When sampling from non-render targets (i.e. uploaded textures)
 * these differences do not matter because OpenGL has a second piece of backwardness, which is that
 * the first row of texels in glTexImage2D is interpreted as the bottom row.
 *
 * To protect users from these differences, we recommend that materials in the SURFACE domain
 * leverage this wrapper function when sampling from offscreen render targets.
 *
 */

#if 0
    // Importance things on screen uv which different from vulkan and opengl.
    vec2 clipToUV(vec2 uv) 
    {
        uv = uv * 0.5f + 0.5f;

    #if defined(VULKAN) || defined(METAL) || defined(DIRECTX)
        uv.y = 1.0 - uv.y; // Vulkan Metal And DirectX
    #endif
        return uv; // Open GL
    }

    vec2 uvToClip(vec2 uv) 
    {
    #if defined(VULKAN) || defined(METAL) || defined(DIRECTX)
        uv.y = 1.0 - uv.y; // Vulkan Metal And DirectX
    #endif

        uv = uv * 2.0f - 1.0f;
        return uv; // Open GL
    }
#endif

// NOTE: Engine's view space depth range is [-zFar, -zNear].
//       See linearizeDepth function and viewspaceDepth function.

//////////////////////////////////////////////////////////////////////////////////////////////////////

//------------------------------------------------------------------
// LMS
// https://en.wikipedia.org/wiki/LMS_color_space
//------------------------------------------------------------------

const mat3 sRGB_2_LMS_MAT = mat3
(
    vec3(17.8824,  43.5161,  4.1193),
    vec3(3.4557,   27.1554,  3.8671),
    vec3(0.02996,  0.18431,  1.4670)
);

const mat3 LMS_2_sRGB_MAT = mat3
(
    vec3(0.0809,  -0.1305,   0.1167),
    vec3(-0.0102,  0.0540,  -0.1136),
    vec3(-0.0003, -0.0041,   0.6935)
);

vec3 sRGB_2_LMS(vec3 RGB)
{
	return RGB * sRGB_2_LMS_MAT;
}

vec3 sRGB_2_AP1(vec3 RGB)
{
	return RGB * sRGB_2_AP1_MAT;
}

vec3 AP1_2_sRGB(vec3 AP1)
{
	return AP1 * AP1_2_sRGB_MAT;
}

vec3 LMS_2_sRGB(vec3 LMS)
{
	return LMS * LMS_2_sRGB_MAT;
}

vec3 convertColorSpace(vec3 srgbColor)
{
#if AP1_COLOR_SPACE
    return srgbColor * sRGB_2_AP1_MAT;
#else
    return srgbColor;
#endif
}

vec3 Ap1_2_LMS(vec3 Ap1)
{
    return (Ap1 * AP1_2_sRGB_MAT) * sRGB_2_LMS_MAT;
}

vec3 LMS_2_Ap1(vec3 LMS)
{
    return (LMS * LMS_2_sRGB_MAT) * sRGB_2_AP1_MAT;
}

// Project position to uv space.
vec3 projectPos(vec3 origin, in const mat4 inMatrix)
{
    vec4 projectPos = inMatrix * vec4(origin, 1.0);
    projectPos.xyz /= projectPos.w;

    projectPos.xy = 0.5 * projectPos.xy + 0.5;
    projectPos.y  = 1.0 - projectPos.y;

    return projectPos.xyz;
}

vec3 posTransform(vec3 position, in const mat4 matrix)
{
    return (matrix * vec4(position, 1.0)).xyz;
}

// Construct position like view space or world space.
vec3 constructPos(vec2 uv, float depthZ, in const mat4 invertMatrix)
{
    vec4 posClip  = vec4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, depthZ, 1.0f);
    vec4 posWorldRebuild = invertMatrix * posClip;
    return posWorldRebuild.xyz / posWorldRebuild.w;
}

// Construct world position from device z and it's sample uv.
vec3 getWorldPos(vec2 uv, float depthZ, in const PerFrameData view)
{
    return constructPos(uv, depthZ, view.camInvertViewProj);
}

// Construct view space position from device z and it's sample uv.
vec3 getViewPos(vec2 uv, float depthZ, in const PerFrameData view)
{
    return constructPos(uv, depthZ, view.camInvertProj);
}

// Vulkan linearize z.
// NOTE: viewspace z range is [-zFar, -zNear], linear z is viewspace z mul -1 result on vulkan.
// if no exist reverse z:
//       linearZ = zNear * zFar / (zFar +  deviceZ * (zNear - zFar));
//  when reverse z enable, then the function is:
//       linearZ = zNear * zFar / (zNear + deviceZ * (zFar - zNear));
float linearizeDepth(float z, float n, float f)
{
    return n * f / (z * (f - n) + n);
}

float linearizeDepth(float z, in const PerFrameData view)
{
    const float n = view.camInfo.z;
    const float f = view.camInfo.w;
    return linearizeDepth(z, n, f);
}

float linearizeDepth01(float z, in const PerFrameData view)
{
    const float n = view.camInfo.z;
    const float f = view.camInfo.w;
    return (linearizeDepth(z, n, f) - n) / (f - n);
}

float linearizeDepthPrev(float z, in const PerFrameData view)
{
    const float n = view.camInfoPrev.z;
    const float f = view.camInfoPrev.w;
    return linearizeDepth(z, n, f);
}

// Derived by glm::perspectiveRH_ZO, same with linearizeDepth function.
float viewspaceDepth(float z, float n, float f)
{
    return linearizeDepth(z, n, f) * -1.0f;
}

// Radical inverse based on http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
vec2 hammersley2d(uint i, uint N) 
{
    // Efficient VanDerCorpus calculation.
	uint bits = (i << 16u) | (i >> 16u);
	bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
	bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
	bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
	bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
	float rdi = float(bits) * 2.3283064365386963e-10;

    // Hammersley sequence.
	return vec2(float(i) /float(N), rdi);
}

// high frequency dither pattern appearing almost random without banding steps
// note: from "NEXT GENERATION POST PROCESSING IN CALL OF DUTY: ADVANCED WARFARE"
//      http://advances.realtimerendering.com/s2014/index.html
float interleavedGradientNoise(vec2 uv, float frameId)
{
	// magic values are found by experimentation
	uv += frameId * (vec2(47.0, 17.0) * 0.695f);

    const vec3 magic = vec3(0.06711056f, 0.00583715f, 52.9829189f);
    return fract(magic.z * fract(dot(uv, magic.xy)));
}

// 3D random number generator inspired by PCGs (permuted congruential generator)
// Using a **simple** Feistel cipher in place of the usual xor shift permutation step
// @param v = 3D integer coordinate
// @return three elements w/ 16 random bits each (0-0xffff).
// ~8 ALU operations for result.x    (7 mad, 1 >>)
// ~10 ALU operations for result.xy  (8 mad, 2 >>)
// ~12 ALU operations for result.xyz (9 mad, 3 >>)
uvec3 rand3DPCG16(ivec3 p)
{
	// taking a signed int then reinterpreting as unsigned gives good behavior for negatives
	uvec3 v = uvec3(p);

	// Linear congruential step. These LCG constants are from Numerical Recipies
	// For additional #'s, PCG would do multiple LCG steps and scramble each on output
	// So v here is the RNG state
	v = v * 1664525u + 1013904223u;

	// PCG uses xorshift for the final shuffle, but it is expensive (and cheap
	// versions of xorshift have visible artifacts). Instead, use simple MAD Feistel steps
	//
	// Feistel ciphers divide the state into separate parts (usually by bits)
	// then apply a series of permutation steps one part at a time. The permutations
	// use a reversible operation (usually ^) to part being updated with the result of
	// a permutation function on the other parts and the key.
	//
	// In this case, I'm using v.x, v.y and v.z as the parts, using + instead of ^ for
	// the combination function, and just multiplying the other two parts (no key) for 
	// the permutation function.
	//
	// That gives a simple mad per round.
	v.x += v.y*v.z;
	v.y += v.z*v.x;
	v.z += v.x*v.y;
	v.x += v.y*v.z;
	v.y += v.z*v.x;
	v.z += v.x*v.y;

	// only top 16 bits are well shuffled
	return v >> 16u;
}

// Quad schedule style, fake pixel shader dispatch style.
// Input-> [0, 63]
//
// Output:
//  00 01 08 09 10 11 18 19
//  02 03 0a 0b 12 13 1a 1b
//  04 05 0c 0d 14 15 1c 1d
//  06 07 0e 0f 16 17 1e 1f
//  20 21 28 29 30 31 38 39
//  22 23 2a 2b 32 33 3a 3b
//  24 25 2c 2d 34 35 3c 3d
//  26 27 2e 2f 36 37 3e 3f
uvec2 remap8x8(uint lane) // gl_LocalInvocationIndex in 8x8 threadgroup.
{
    return uvec2(
        (((lane >> 2) & 0x0007) & 0xFFFE) | lane & 0x0001,
        ((lane >> 1) & 0x0003) | (((lane >> 3) & 0x0007) & 0xFFFC)
    );
}

float mean(vec2 v) { return dot(v, vec2(1.0f / 2.0f)); }
float mean(vec3 v) { return dot(v, vec3(1.0f / 3.0f)); }
float mean(vec4 v) { return dot(v, vec4(1.0f / 4.0f)); }

float sum(vec2 v) { return v.x + v.y; }
float sum(vec3 v) { return v.x + v.y + v.z; }
float sum(vec4 v) { return v.x + v.y + v.z + v.w; }

// Max between three components
float max3(vec3 xyz) { return max(xyz.x, max(xyz.y, xyz.z)); }
float max4(vec4 xyzw) { return max(xyzw.x, max(xyzw.y, max(xyzw.z, xyzw.w))); }

float min3(vec3 xyz) { return min(xyz.x, min(xyz.y, xyz.z)); }
float min4(vec4 xyzw) { return min(xyzw.x, min(xyzw.y, min(xyzw.z, xyzw.w))); }

float saturate(float x) { return clamp(x, 0.0, 1.0); }
vec2  saturate(vec2  x) { return clamp(x, vec2(0.0), vec2(1.0)); }
vec3  saturate(vec3  x) { return clamp(x, vec3(0.0), vec3(1.0)); }
vec4  saturate(vec4  x) { return clamp(x, vec4(0.0), vec4(1.0)); }

// Saturated range, [0, 1] 
bool isSaturated(float x) { return x >= 0.0f && x <= 1.0f; }
bool isSaturated( vec2 x) { return isSaturated(x.x) && isSaturated(x.y); }
bool isSaturated( vec3 x) { return isSaturated(x.x) && isSaturated(x.y) && isSaturated(x.z);}
bool isSaturated( vec4 x) { return isSaturated(x.x) && isSaturated(x.y) && isSaturated(x.z) && isSaturated(x.w);}

// On range, [minV, maxV]
bool onRange(float x, float minV, float maxV) { return x >= minV && x <= maxV;}
bool onRange( vec2 x,  vec2 minV,  vec2 maxV) { return onRange(x.x, minV.x, maxV.x) && onRange(x.y, minV.y, maxV.y);}
bool onRange( vec3 x,  vec3 minV,  vec3 maxV) { return onRange(x.x, minV.x, maxV.x) && onRange(x.y, minV.y, maxV.y) && onRange(x.z, minV.z, maxV.z);}
bool onRange( vec4 x,  vec4 minV,  vec4 maxV) { return onRange(x.x, minV.x, maxV.x) && onRange(x.y, minV.y, maxV.y) && onRange(x.z, minV.z, maxV.z) && onRange(x.w, minV.w, maxV.w);}

// Rounds value to the nearest multiple of 8
uvec2 roundUp8(uvec2 value) 
{
    uvec2 roundDown = value & ~0x7;
    return (roundDown == value) ? value : value + 8;
}

float luminance(vec3 color)
{
    // human eye aware lumiance function.
    return dot(color, vec3(0.299, 0.587, 0.114));
}

float luminanceRec709(vec3 color)
{
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

// Build one TBN matrix from normal input.
// 
mat3 createTBN(vec3 N) 
{
    vec3 U;
    if (abs(N.z) > 0.0) 
    {
        float k = sqrt(N.y * N.y + N.z * N.z);
        U.x = 0.0; 
        U.y = -N.z / k; 
        U.z = N.y / k;
    }
    else 
    {
        float k = sqrt(N.x * N.x + N.y * N.y);
        U.x = N.y / k; 
        U.y = -N.x / k; 
        U.z = 0.0;
    }

    mat3 TBN = mat3(U, cross(N, U), N);
    return transpose(TBN);
}

struct ScreenSpaceRay 
{
    vec3 ssRayStart;
    vec3 ssRayEnd;
    vec3 ssViewRayEnd;
    vec3 uvRayStart;
    vec3 uvRay;
};

void initScreenSpaceRay(
	  out  ScreenSpaceRay ray
	, vec3 wsRayStart
	, vec3 wsRayDirection
	, float wsRayLength
	, in const PerFrameData viewData) 
{
    const mat4 worldToClip = viewData.camViewProj;
    const mat4 viewToClip = viewData.camProj;

    // ray end in world space
    vec3 wsRayEnd = wsRayStart + wsRayDirection * wsRayLength;

    // ray start/end in clip space
    vec4 csRayStart = worldToClip * vec4(wsRayStart, 1.0);
    vec4 csRayEnd = worldToClip * vec4(wsRayEnd, 1.0);
    vec4 csViewRayEnd = csRayStart + viewToClip * vec4(0.0, 0.0, wsRayLength, 0.0);

    // ray start/end in screen space
    ray.ssRayStart = csRayStart.xyz / csRayStart.w;
    ray.ssRayEnd = csRayEnd.xyz / csRayEnd.w;
    ray.ssViewRayEnd = csViewRayEnd.xyz / csViewRayEnd.w;

    // convert all to uv (texture) space
    vec3 uvRayEnd = vec3(ray.ssRayEnd.xy * 0.5 + 0.5, ray.ssRayEnd.z);
	uvRayEnd.y = 1.0f - uvRayEnd.y;

    ray.uvRayStart = vec3(ray.ssRayStart.xy * 0.5 + 0.5, ray.ssRayStart.z);
	ray.uvRayStart.y = 1.0f - ray.uvRayStart.y;

    ray.uvRay = uvRayEnd - ray.uvRayStart;
}

// OpenGL core profile specs, section 8.13.
// Get 3d sampling vector from uv, useful when do cubemap filter on compute shader. 
vec3 getSamplingPosition(uint faceId, vec2 st)
{
    vec2 uv = 2.0 * vec2(st.x, 1.0 - st.y) - vec2(1.0);
    vec3 ret;

         if(faceId == 0) ret = vec3(  1.0,  uv.y, -uv.x);
    else if(faceId == 1) ret = vec3( -1.0,  uv.y,  uv.x);
    else if(faceId == 2) ret = vec3( uv.x,  1.0,  -uv.y);
    else if(faceId == 3) ret = vec3( uv.x, -1.0,   uv.y);
    else if(faceId == 4) ret = vec3( uv.x,  uv.y,   1.0);
    else if(faceId == 5) ret = vec3(-uv.x,  uv.y,  -1.0);

    return ret;
}

vec3 getSamplingVector(uint faceId, vec2 st)
{
    return normalize(getSamplingPosition(faceId, st));
}

vec3 cubeSmooth(vec3 x)
{
    return x * x * (3.0 - 2.0 * x);
}

// Activision GTAO paper: https://www.activision.com/cdn/research/s2016_pbs_activision_occlusion.pptx
vec3 AoMultiBounce(float AO, vec3 baseColor)
{
    vec3 a =  2.0404 * baseColor - 0.3324;
    vec3 b = -4.7951 * baseColor + 0.6417;
    vec3 c =  2.7552 * baseColor + 0.6903;

    vec3 x  = vec3(AO);

    return max(x, ((x * a + b) * x + c) * x);
}

// Ray sphere intersection. 
// https://zhuanlan.zhihu.com/p/136763389
// https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection
// Returns distance from r0 to first intersecion with sphere, or -1.0 if no intersection.
float raySphereIntersectNearest(
      vec3  r0  // ray origin
    , vec3  rd  // normalized ray direction
    , vec3  s0  // sphere center
    , float sR) // sphere radius
{
	float a = dot(rd, rd);

	vec3 s02r0 = r0 - s0;
	float b = 2.0 * dot(rd, s02r0);

	float c = dot(s02r0, s02r0) - (sR * sR);
	float delta = b * b - 4.0 * a * c;

    // No intersection state.
	if (delta < 0.0 || a == 0.0)
	{
		return -1.0;
	}

	float sol0 = (-b - sqrt(delta)) / (2.0 * a);
	float sol1 = (-b + sqrt(delta)) / (2.0 * a);
	// sol1 > sol0


    // Intersection on negative direction, no suitable for ray.
	if (sol1 < 0.0) // When sol1 < 0.0, sol0 < 0.0 too.
	{
		return -1.0;
	}

    // Maybe exist one positive intersection.
	if (sol0 < 0.0)
	{
		return max(0.0, sol1);
	}

    // Two positive intersection, return nearest one.
	return max(0.0, min(sol0, sol1));
}

// When ensure r0 is inside of sphere.
// Only exist one positive result, use it.
float raySphereIntersectInside(
      vec3  r0  // ray origin
    , vec3  rd  // normalized ray direction
    , vec3  s0  // sphere center
    , float sR) // sphere radius
{
	float a = dot(rd, rd);

	vec3 s02r0 = r0 - s0;
	float b = 2.0 * dot(rd, s02r0);

	float c = dot(s02r0, s02r0) - (sR * sR);
	float delta = b * b - 4.0 * a * c;

	// float sol0 = (-b - sqrt(delta)) / (2.0 * a);
	float sol1 = (-b + sqrt(delta)) / (2.0 * a);

	// sol1 > sol0, so just return sol1
	return sol1;
}

// Ray intersection from outside of sphere.
// Return true if exist intersect. don't care about tangent case.
bool raySphereIntersectOutSide(
      vec3  r0  // ray origin
    , vec3  rd  // normalized ray direction
    , vec3  s0  // sphere center
    , float sR  // sphere radius
	, out vec2 t0t1) 
{
	float a = dot(rd, rd);

	vec3 s02r0 = r0 - s0;
	float b = 2.0 * dot(rd, s02r0);

	float c = dot(s02r0, s02r0) - (sR * sR);
	float delta = b * b - 4.0 * a * c;

    // No intersection state.
	if (delta < 0.0 || a == 0.0)
	{
		return false;
	}

	float sol0 = (-b - sqrt(delta)) / (2.0 * a);
	float sol1 = (-b + sqrt(delta)) / (2.0 * a);
	

    // Intersection on negative direction, no suitable for ray.
	if (sol1 <= 0.0 || sol0 <= 0.0)
	{
		return false;
	}

    // Two positive intersection, return nearest one.
	t0t1 = vec2(sol0, sol1); // sol1 > sol0
	return true; 
}

float getUniformPhase()
{
	return 1.0f / (4.0f * kPI);
}

// https://www.shadertoy.com/view/Mtc3Ds
// rayleigh phase function.
float rayleighPhase(float cosTheta)
{
	const float factor = 3.0f / (16.0f * kPI);
	return factor * (1.0f + cosTheta * cosTheta);
}

// Schlick approximation
float cornetteShanksMiePhaseFunction(float g, float cosTheta)
{
	float k = 3.0 / (8.0 * kPI) * (1.0 - g * g) / (2.0 + g * g);
	return k * (1.0 + cosTheta * cosTheta) / pow(1.0 + g * g - 2.0 * g * -cosTheta, 1.5);
}

// Crazy light intensity.
float henyeyGreenstein(float cosTheta, float g) 
{
	float gg = g * g;
	return (1. - gg) / pow(1. + gg - 2. * g * cosTheta, 1.5);
}

// See http://www.pbr-book.org/3ed-2018/Volume_Scattering/Phase_Functions.html
float hgPhase(float g, float cosTheta)
{
	float numer = 1.0f - g * g;
	float denom = 1.0f + g * g + 2.0f * g * cosTheta;
	return numer / (4.0f * kPI * denom * sqrt(denom));
}

float dualLobPhase(float g0, float g1, float w, float cosTheta)
{
	return mix(hgPhase(g0, cosTheta), hgPhase(g1, cosTheta), w);
}

float beersLaw(float density, float stepLength, float densityScale)
{
	return exp(-density * stepLength * densityScale);
}

// From https://www.shadertoy.com/view/4sjBDG
float numericalMieFit(float costh)
{
    // This function was optimized to minimize (delta*delta)/reference in order to capture
    // the low intensity behavior.
    float bestParams[10];
    bestParams[0]=9.805233e-06;
    bestParams[1]=-6.500000e+01;
    bestParams[2]=-5.500000e+01;
    bestParams[3]=8.194068e-01;
    bestParams[4]=1.388198e-01;
    bestParams[5]=-8.370334e+01;
    bestParams[6]=7.810083e+00;
    bestParams[7]=2.054747e-03;
    bestParams[8]=2.600563e-02;
    bestParams[9]=-4.552125e-12;
    
    float p1 = costh + bestParams[3];
    vec4 expValues = exp(vec4(bestParams[1] *costh+bestParams[2], bestParams[5] *p1*p1, bestParams[6] *costh, bestParams[9] *costh));
    vec4 expValWeight= vec4(bestParams[0], bestParams[4], bestParams[7], bestParams[8]);
    return dot(expValues, expValWeight);
}

// Relative error : ~3.4% over full
// Precise format : ~small float
// 2 ALU
float rsqrtFast(float x)
{
	int i = floatBitsToInt(x);
	i = 0x5f3759df - (i >> 1);
	return intBitsToFloat (i);
}

// max absolute error 9.0x10^-3
// Eberly's polynomial degree 1 - respect bounds
// 4 VGPR, 12 FR (8 FR, 1 QR), 1 scalar
// input [-1, 1] and output [0, PI]
float acosFast(float inX) 
{
    float x = abs(inX);
    float res = -0.156583f * x + (0.5 * kPI);
    res *= sqrt(1.0f - x);
    return (inX >= 0) ? res : kPI - res;
}

// Approximates acos(x) with a max absolute error of 9.0x10^-3.
// Input [0, 1]
float acosFastPositive(float x) 
{
    float p = -0.1565827f * x + 1.570796f;
    return p * sqrt(1.0 - x);
}

// Activision GTAO paper: https://www.activision.com/cdn/research/s2016_pbs_activision_occlusion.pptx
vec3 gtaoMultiBounce(float AO, vec3 baseColor)
{
    vec3 a =  2.0404 * baseColor - 0.3324;
    vec3 b = -4.7951 * baseColor + 0.6417;
    vec3 c =  2.7552 * baseColor + 0.6903;

    vec3 x  = vec3(AO);

    return max(x, ((x * a + b) * x + c) * x);
}

float whangHashNoise(uint u, uint v, uint s)
{
    // return fract(sin(float(s + (u*1080u + v)%10000u) * 78.233) * 43758.5453);
    uint seed = (u*1664525u + v) + s;
    
    seed  = (seed ^ 61u) ^(seed >> 16u);
    seed *= 9u;
    seed  = seed ^(seed >> 4u);
    seed *= uint(0x27d4eb2d);
    seed  = seed ^(seed >> 15u);
    
    float value = float(seed) / (4294967296.0);
    return value;
}

// Simple hash uint.
// from niagara stream. see https://www.youtube.com/watch?v=BR2my8OE1Sc
uint simpleHash(uint a)
{
   a = (a + 0x7ed55d16) + (a << 12);
   a = (a ^ 0xc761c23c) ^ (a >> 19);
   a = (a + 0x165667b1) + (a << 5);
   a = (a + 0xd3a2646c) ^ (a << 9);
   a = (a + 0xfd7046c5) + (a << 3);
   a = (a ^ 0xb55a4f09) ^ (a >> 16);
   return a;
}

// Simple hash color from uint value.
// from niagara stream. see https://www.youtube.com/watch?v=BR2my8OE1Sc
vec3 simpleHashColor(uint i)
{
    uint h = simpleHash(i);
    return vec3(float(h & 255), float((h >> 8) & 255), float((h >> 16) & 255)) / 255.0;
}

// return intersect t, negative meaning no intersection.
float linePlaneIntersect(vec3 lineStart, vec3 lineDirection, vec3 planeP, vec3 planeNormal)
{
    float ndl = dot(lineDirection, planeNormal);
    return ndl != 0.0f ? dot((planeP - lineStart), planeNormal) / ndl : -1.0f;
}

// world space box intersect.
// Axis is y up, negative meaning no intersection.
float boxLineIntersectWS(vec3 lineStart, vec3 lineDirection, vec3 bboxMin, vec3 bboxMax)
{
    const float kMaxDefault = 9e10f;
    const float kMaxDiff = 9e9f;

    float t0 = linePlaneIntersect(lineStart, lineDirection, bboxMin, vec3(-1.0f, 0.0f, 0.0f)); t0 = t0 < 0.0f ? kMaxDefault : t0;
    float t1 = linePlaneIntersect(lineStart, lineDirection, bboxMax, vec3( 1.0f, 0.0f, 0.0f)); t1 = t1 < 0.0f ? kMaxDefault : t1;
    float t2 = linePlaneIntersect(lineStart, lineDirection, bboxMax, vec3( 0.0f, 1.0f, 0.0f)); t2 = t2 < 0.0f ? kMaxDefault : t2;
    float t3 = linePlaneIntersect(lineStart, lineDirection, bboxMin, vec3(0.0f, -1.0f, 0.0f)); t3 = t3 < 0.0f ? kMaxDefault : t3;
    float t4 = linePlaneIntersect(lineStart, lineDirection, bboxMin, vec3(0.0f, 0.0f, -1.0f)); t4 = t4 < 0.0f ? kMaxDefault : t4;
    float t5 = linePlaneIntersect(lineStart, lineDirection, bboxMax, vec3( 0.0f, 0.0f, 1.0f)); t5 = t5 < 0.0f ? kMaxDefault : t5;

    float tMin = t0;
    tMin = min(tMin, t1);
    tMin = min(tMin, t2);
    tMin = min(tMin, t3);
    tMin = min(tMin, t4);
    tMin = min(tMin, t5);

    return tMin < kMaxDiff ? tMin : -1.0f;
}

float intersectDirPlaneOneSided(vec3 dir, vec3 normal, vec3 pt)
{
    float d = -dot(pt, normal);
    float t = d / max(1e-5f, -dot(dir, normal));
    return t;
}

// From Open Asset Import Library
// https://github.com/assimp/assimp/blob/master/include/assimp/matrix3x3.inl
mat3 rotFromToMatrix(vec3 from, vec3 to)
{
    float e = dot(from, to);
    float f = abs(e);

    if (f > 1.0f - 0.0003f)
        return mat3(
            1.f, 0.f, 0.f, 
            0.f, 1.f, 0.f, 
            0.f, 0.f, 1.f);

    vec3 v   = cross(from, to);
    float h    = 1.f / (1.f + e);      /* optimization by Gottfried Chen */
    float hvx  = h * v.x;
    float hvz  = h * v.z;
    float hvxy = hvx * v.y;
    float hvxz = hvx * v.z;
    float hvyz = hvz * v.y;

    mat3 mtx;
    mtx[0][0] = e + hvx * v.x;
    mtx[0][1] = hvxy - v.z;
    mtx[0][2] = hvxz + v.y;

    mtx[1][0] = hvxy + v.z;
    mtx[1][1] = e + h * v.y * v.y;
    mtx[1][2] = hvyz - v.x;

    mtx[2][0] = hvxz - v.y;
    mtx[2][1] = hvyz + v.x;
    mtx[2][2] = e + hvz * v.z;

    return transpose(mtx);
}

float atan2(vec2 v)
{
	return v.x == 0.0 ?
		(1.0 - step(abs(v.y), 0.0)) * sign(v.y) * kPI * 0.5 :
		atan(v.y / v.x) + step(v.x, 0.0) * sign(v.y) * kPI;
}



// Atmosphere shader functions.
struct AtmosphereParameters
{
	// Radius of the planet (center to ground)
	float bottomRadius;

	// Maximum considered atmosphere height (center to atmosphere top)
	float topRadius;

	// Rayleigh scattering exponential distribution scale in the atmosphere
	float rayleighDensityExpScale;

	// Rayleigh scattering coefficients
	vec3 rayleighScattering;

	// Mie scattering exponential distribution scale in the atmosphere
	float mieDensityExpScale;

	// Mie scattering coefficients
	vec3 mieScattering;

	// Mie extinction coefficients
	vec3 mieExtinction;

	// Mie absorption coefficients
	vec3 mieAbsorption;

	// Mie phase function excentricity
	float miePhaseG;

	// Another medium type in the atmosphere
	float absorptionDensity0LayerWidth;
	float absorptionDensity0ConstantTerm;
	float absorptionDensity0LinearTerm;
	float absorptionDensity1ConstantTerm;
	float absorptionDensity1LinearTerm;

	// This other medium only absorb light, e.g. useful to represent ozone in the earth atmosphere
	vec3 absorptionExtinction;

	// The albedo of the ground.
	vec3 groundAlbedo;

	float multipleScatteringFactor; 

	uint viewRayMarchMinSPP;
	uint viewRayMarchMaxSPP;

    float cloudAreaStartHeight; // km
    float cloudAreaThickness;
    mat4 cloudShadowViewProj;
    mat4 cloudShadowViewProjInverse;
};

// Build atmosphere parameters from frame data.
AtmosphereParameters getAtmosphereParameters(in const PerFrameData frameData)
{
	AtmosphereParameters parameters;

    const AtmosphereParametersInputs config = frameData.atmosphere;

	parameters.absorptionExtinction = config.absorptionColor * config.absorptionLength;

    // Copy parameters.
    parameters.groundAlbedo             = config.groundAlbedo;
	parameters.bottomRadius             = config.bottomRadius;
	parameters.topRadius                = config.topRadius;
    parameters.viewRayMarchMinSPP       = config.viewRayMarchMinSPP;
	parameters.viewRayMarchMaxSPP       = config.viewRayMarchMaxSPP;
	parameters.miePhaseG                = config.miePhaseFunctionG;
	parameters.multipleScatteringFactor = config.multipleScatteringFactor;

	// Traslation from Bruneton2017 parameterisation.
	parameters.rayleighDensityExpScale        = config.rayleighDensity[1].w;
	parameters.mieDensityExpScale             = config.mieDensity[1].w;
	parameters.absorptionDensity0LayerWidth   = config.absorptionDensity[0].x;
	parameters.absorptionDensity0ConstantTerm = config.absorptionDensity[1].x;
	parameters.absorptionDensity0LinearTerm   = config.absorptionDensity[0].w;
	parameters.absorptionDensity1ConstantTerm = config.absorptionDensity[2].y;
	parameters.absorptionDensity1LinearTerm   = config.absorptionDensity[2].x;

	parameters.rayleighScattering = config.rayleighScatteringColor * config.rayleighScatterLength;
    parameters.mieAbsorption      = config.mieAbsorption;
	parameters.mieScattering      = config.mieScatteringColor * config.mieScatteringLength;
	parameters.mieExtinction      = parameters.mieScattering + config.mieAbsColor * config.mieAbsLength;

    parameters.cloudAreaStartHeight       = frameData.cloud.cloudAreaStartHeight;
    parameters.cloudAreaThickness         = frameData.cloud.cloudAreaThickness;
    parameters.cloudShadowViewProj        = frameData.cloud.cloudSpaceViewProject;
    parameters.cloudShadowViewProjInverse = frameData.cloud.cloudSpaceViewProjectInverse;

	return parameters;
}

// https://github.com/sebh/UnrealEngineSkyAtmosphere
// Transmittance LUT function parameterisation from Bruneton 2017 https://github.com/ebruneton/precomputed_atmospheric_scattering
// Detail also in video https://www.youtube.com/watch?v=y-oBGzDCZKI at 08:35.
void lutTransmittanceParamsToUv(
    in const AtmosphereParameters atmosphere, 
    in float viewHeight, // [bottomRAdius, topRadius]
    in float viewZenithCosAngle, // [-1,1]
    out vec2 uv) // [0,1]
{
	float H = sqrt(max(0.0f, atmosphere.topRadius * atmosphere.topRadius - atmosphere.bottomRadius * atmosphere.bottomRadius));
	float rho = sqrt(max(0.0f, viewHeight * viewHeight - atmosphere.bottomRadius * atmosphere.bottomRadius));

	uv.y = rho / H;

	// Distance to atmosphere boundary
	float discriminant = viewHeight * viewHeight * (viewZenithCosAngle * viewZenithCosAngle - 1.0) + atmosphere.topRadius * atmosphere.topRadius;
	float d = max(0.0, (-viewHeight * viewZenithCosAngle + sqrt(discriminant))); 

	float dMin = atmosphere.topRadius - viewHeight;
	float dMax = rho + H;

	uv.x = (d - dMin) / (dMax - dMin);
}

void uvToLutTransmittanceParams(
    in const AtmosphereParameters atmosphere, 
    out float viewHeight, // [bottomRAdius, topRadius]
    out float viewZenithCosAngle, // [-1,1]
    in vec2 uv) // [0,1]
{ 
	float H = sqrt(atmosphere.topRadius * atmosphere.topRadius - atmosphere.bottomRadius * atmosphere.bottomRadius);
	float rho = H * uv.y;
	viewHeight = sqrt(rho * rho + atmosphere.bottomRadius * atmosphere.bottomRadius);

	float dMin = atmosphere.topRadius - viewHeight;
	float dMax = rho + H;

	// Distance to atmosphere boundary
	float d = dMin + uv.x * (dMax - dMin);

	viewZenithCosAngle = (d == 0.0) ? 1.0f : (H * H - rho * rho - d * d) / (2.0 * viewHeight * d);
	viewZenithCosAngle = clamp(viewZenithCosAngle, -1.0, 1.0);
}

float fromUnitToSubUvs(float u, float resolution) { return (u + 0.5f / resolution) * (resolution / (resolution + 1.0f)); }
float fromSubUvsToUnit(float u, float resolution) { return (u - 0.5f / resolution) * (resolution / (resolution - 1.0f)); }

void skyViewLutParamsToUv(
	in const AtmosphereParameters atmosphere, 
	in bool  bIntersectGround, 
	in float viewZenithCosAngle, 
	in float lightViewCosAngle, 
	in float viewHeight, 
    in vec2 lutSize,
	out vec2 uv)
{
	float vHorizon = sqrt(viewHeight * viewHeight - atmosphere.bottomRadius * atmosphere.bottomRadius);

	// Ground to horizon cos.
	float cosBeta = vHorizon / viewHeight;		

	float beta = acos(cosBeta);
	float zenithHorizonAngle = kPI - beta;

	if (!bIntersectGround)
	{
		float coord = acos(viewZenithCosAngle) / zenithHorizonAngle;
		coord = 1.0 - coord;
		coord = sqrt(coord); // Non-linear sky view lut.

		coord = 1.0 - coord;
		uv.y = coord * 0.5f;
	}
	else
	{
		float coord = (acos(viewZenithCosAngle) - zenithHorizonAngle) / beta;
		coord = sqrt(coord); // Non-linear sky view lut.

		uv.y = coord * 0.5f + 0.5f;
	}

	// UV x remap.
	{
		float coord = -lightViewCosAngle * 0.5f + 0.5f;
		coord = sqrt(coord);
		uv.x = coord;
	}

	// Constrain uvs to valid sub texel range (avoid zenith derivative issue making LUT usage visible)
	uv = vec2(fromUnitToSubUvs(uv.x, lutSize.x), fromUnitToSubUvs(uv.y, lutSize.y));
}

float aerialPerspectiveDepthToSlice(float depth) { return depth * (1.0f / kAtmosphereAirPerspectiveKmPerSlice); }
float aerialPerspectiveSliceToDepth(float slice) { return slice * kAtmosphereAirPerspectiveKmPerSlice; }

#define kSkyFogSafeHZBLevel 1

// Total 32x32x32 0.5 meaning total 16 km.
#define kAtmosphereDistantGridKmPerSlice 4.0
float distantGridDepthToSlice(float depth) { return depth * (1.0f / kAtmosphereDistantGridKmPerSlice); }
float distantGridSliceToDepth(float slice) { return slice * kAtmosphereDistantGridKmPerSlice; }

// Camera unit to atmosphere unit convert. meter -> kilometers.
vec3 convertToAtmosphereUnit(vec3 o, in const PerFrameData frame)
{
	return o / kAtmosphereUnitScale + vec3(0.0, kAtmosphereCameraOffset, 0.0);
}  

// atmosphere unit to camera unit convert. kilometers -> meter.
vec3 convertToCameraUnit(vec3 o, in const PerFrameData frame)
{
	return (o - vec3(0.0, kAtmosphereCameraOffset, 0.0)) * kAtmosphereUnitScale;
}  

// Participating media functions and struct.
// From https://github.com/sebh/UnrealEngineSkyAtmosphere
struct MediumSampleRGB
{
	vec3 scattering;
	vec3 absorption;
	vec3 extinction;

	vec3 scatteringMie;
	vec3 absorptionMie;
	vec3 extinctionMie;

	vec3 scatteringRay;
	vec3 absorptionRay;
	vec3 extinctionRay;

	vec3 scatteringOzo;
	vec3 absorptionOzo;
	vec3 extinctionOzo;

	vec3 albedo;
};

float getAlbedo(float scattering, float extinction)
{
	return scattering / max(0.001, extinction);
}

vec3 getAlbedo(vec3 scattering, vec3 extinction)
{
	return scattering / max(vec3(0.001), extinction);
}

float getViewHeight(vec3 worldPos, in const AtmosphereParameters atmosphere)
{
	// Current default set planet center is (0, 0, 0).
    // And we start from horizontal plane which treat as 0 plane on height.
	return length(worldPos) - atmosphere.bottomRadius;
}

bool moveToTopAtmosphere(inout vec3 worldPos, in const vec3 worldDir, in const float atmosphereTopRadius)
{
	float viewHeight = length(worldPos);
	if (viewHeight > atmosphereTopRadius)
	{
		float tTop = raySphereIntersectNearest(worldPos, worldDir, vec3(0.0), atmosphereTopRadius);
		if (tTop >= 0.0f)
		{
			vec3 upVector = worldPos / viewHeight;
			vec3 upOffset = upVector * -kAtmospherePlanetRadiusOffset;
			worldPos = worldPos + worldDir * tTop + upOffset;
		}
		else
		{
			// Ray is not intersecting the atmosphere
			return false;
		}
	}
	return true; // ok to start tracing
}

MediumSampleRGB sampleMediumRGB(in vec3 worldPos, in const AtmosphereParameters atmosphere)
{
	const float viewHeight = getViewHeight(worldPos, atmosphere);

    // Get mie and ray density.
	const float densityMie = exp(atmosphere.mieDensityExpScale * viewHeight);
	const float densityRay = exp(atmosphere.rayleighDensityExpScale * viewHeight);

    // Get ozone density.
	const float densityOzo = saturate(viewHeight < atmosphere.absorptionDensity0LayerWidth ?
		atmosphere.absorptionDensity0LinearTerm * viewHeight + atmosphere.absorptionDensity0ConstantTerm :
		atmosphere.absorptionDensity1LinearTerm * viewHeight + atmosphere.absorptionDensity1ConstantTerm);

    // Build medium sample.
	MediumSampleRGB s;

    // Mie term.
	s.scatteringMie = densityMie * atmosphere.mieScattering;
	s.absorptionMie = densityMie * atmosphere.mieAbsorption;
	s.extinctionMie = densityMie * atmosphere.mieExtinction;

    // Ray term.
	s.scatteringRay = densityRay * atmosphere.rayleighScattering;
	s.absorptionRay = vec3(0.0);
	s.extinctionRay = s.scatteringRay + s.absorptionRay;

    // Ozone term.
	s.scatteringOzo = vec3(0.0);
	s.absorptionOzo = densityOzo * atmosphere.absorptionExtinction;
	s.extinctionOzo = s.scatteringOzo + s.absorptionOzo;

    // Composite.
	s.scattering = s.scatteringMie + s.scatteringRay + s.scatteringOzo;
	s.absorption = s.absorptionMie + s.absorptionRay + s.absorptionOzo;
	s.extinction = s.extinctionMie + s.extinctionRay + s.extinctionOzo;
	s.albedo = getAlbedo(s.scattering, s.extinction);
	return s;
}

// End atmosphere shader functions.

// Optional load exposure.
float getExposure(in const PerFrameData inFrame, texture2D tex)
{
    if(inFrame.postprocessing.bAutoExposureEnable != 0)
    {
        return texelFetch(tex, ivec2(0, 0), 0).r;
    }
    else
    {
        return inFrame.postprocessing.autoExposureFixExposure;
    }
}

uvec2 jitterSequence(uint index, uvec2 dimension, uvec2 dispatchId)
{
	uvec2 offset = uvec2(vec2(0.754877669, 0.569840296) * index * dimension);
    uvec2 offsetId = dispatchId + offset;
    offsetId.x = offsetId.x % dimension.x;
    offsetId.y = offsetId.y % dimension.y;

	return offsetId;
}

float pack28bitUnsignedNorm(int v)
{
	return float(v + 0.5f) / 255.0f;
}

int unpackFrom8BitUnsignedNorm(float v)
{
	return int(v * 255.0f);
}

vec3 packWorldNormal(vec3 src)
{
	return (src + vec3(1.0)) * 0.5;
}

vec3 unpackWorldNormal(vec3 pack)
{
	return normalize(pack * 2.0 - vec3(1.0));
}

// Pack to 8bit unorm.
float packShadingModelId(EShadingModelType src)
{
	return pack28bitUnsignedNorm(src);
}

EShadingModelType unpackShadingModelId(float v)
{
	return unpackFrom8BitUnsignedNorm(v);
}

// 16 bit unorm.
float packObjectId(uint objectId)
{
    return float(objectId + 0.5f) / 65535.0f;
}

uint unpackFrom16bitObjectId(float v)
{
    return uint(v * 65535.0f);
}

// This function take a rgb color (best is to provide color in sRGB space)
// and return a YCoCg color in [0..1] space for 8bit (An offset is apply in the function)
// Ref: http://www.nvidia.com/object/real-time-ycocg-dxt-compression.html
#define YCOCG_CHROMA_BIAS (128.0 / 255.0)
vec3 RGBToYCoCg(vec3 rgb)
{
    vec3 YCoCg;
    YCoCg.x = dot(rgb, vec3(0.25, 0.5, 0.25));
    YCoCg.y = dot(rgb, vec3(0.5, 0.0, -0.5))    + YCOCG_CHROMA_BIAS;
    YCoCg.z = dot(rgb, vec3(-0.25, 0.5, -0.25)) + YCOCG_CHROMA_BIAS;

    return YCoCg;
}

vec3 YCoCgToRGB(vec3 YCoCg)
{
    float Y = YCoCg.x;
    float Co = YCoCg.y - YCOCG_CHROMA_BIAS;
    float Cg = YCoCg.z - YCOCG_CHROMA_BIAS;

    vec3 rgb;
    rgb.r = Y + Co - Cg;
    rgb.g = Y + Cg;
    rgb.b = Y - Co - Cg;

    return rgb;
}

vec4 loadBentNormalSSAO(vec4 sampleValue, vec4 fallback, in const PerFrameData frameData)
{
    return (frameData.postprocessing.ssao_enable != 0) ? sampleValue : fallback;
}

uint depthPackUnit(float depth)
{
    return floatBitsToUint(depth);
}

float uintDepthUnpack(uint uintDepth)
{
    return uintBitsToFloat(uintDepth);
}

float remap(float value, float orignalMin, float orignalMax, float newMin, float newMax)
{
    return newMin + (saturate((value - orignalMin) / (orignalMax - orignalMin)) * (newMax - newMin));
}

float screenSpaceContactShadow(
    texture2D inSceneDepth,
    sampler inPointClampEdgeSampler,
    in const PerFrameData inViewData,
    float noise01, 
    uint stepNum, 
    vec3 wsRayStart, 
    vec3 wsRayDirection, 
    float wsRayLength) 
{
    // cast a ray in the direction of the light
    float occlusion = 0.0;
    
    ScreenSpaceRay rayData;
    initScreenSpaceRay(rayData, wsRayStart, wsRayDirection, wsRayLength, inViewData);

    // step
    const uint kStepCount = stepNum;
    const float dt = 1.0 / float(kStepCount);

    // tolerance
    const float tolerance = abs(rayData.ssViewRayEnd.z - rayData.ssRayStart.z) * dt;

    // dither the ray with interleaved gradient noise
    const float dither = noise01 - 0.5;

    // normalized position on the ray (0 to 1)
    float t = dt * dither + dt;

    vec3 ray;
    for (uint i = 0u ; i < kStepCount ; i++, t += dt) 
    {
        ray = rayData.uvRayStart + rayData.uvRay * t;
        float z = texture(sampler2D(inSceneDepth, inPointClampEdgeSampler), ray.xy).r;
        float dz = z - ray.z;
        if (abs(tolerance - dz) < tolerance) 
        {
            occlusion = 1.0;
            break;
        }
    }

    // we fade out the contribution of contact shadows towards the edge of the screen
    // because we don't have depth data there
    vec2 fade = max(12.0 * abs(ray.xy - 0.5) - 5.0, 0.0);
    occlusion *= saturate(1.0 - dot(fade, fade));

    return occlusion;
}

// 4x4 bayer filter, use for cloud reconstruction.
int kBayerMatrix16[16] = int[16]
(
    0,  8,  2,  10, 
    12, 4,  14, 6, 
    3,  11, 1,  9, 
    15, 7,  13, 5
);
// ivec2 offset = ivec2(kBayerMatrix16[frameId] % 4, kBayerMatrix16[frameId] / 4);
float bayerDither(float grayscale, ivec2 pixelCoord)
{    
    int pixelIndex16 = (pixelCoord.x % 4) + (pixelCoord.y % 4) * 4;
    return grayscale > (float(kBayerMatrix16[pixelIndex16]) + 0.5) / 16.0 ? 1.0 : 0.0;
}

float bayer2(vec2 a) 
{
    a = floor(a);
    return fract(dot(a, vec2(0.5, a.y * 0.75)));
}

float bayer4(const vec2 a)   { return bayer2 (0.5   * a) * 0.25     + bayer2(a); }
float bayer8(const vec2 a)   { return bayer4 (0.5   * a) * 0.25     + bayer2(a); }
float bayer16(const vec2 a)  { return bayer4 (0.25  * a) * 0.0625   + bayer4(a); }
float bayer32(const vec2 a)  { return bayer8 (0.25  * a) * 0.0625   + bayer4(a); }
float bayer64(const vec2 a)  { return bayer8 (0.125 * a) * 0.015625 + bayer8(a); }
float bayer128(const vec2 a) { return bayer16(0.125 * a) * 0.015625 + bayer8(a); }

// https://gist.github.com/TheRealMJP/c83b8c0f46b63f3a88a5986f4fa982b1
vec4 sampleTextureCatmullRom(in texture2D tex, in sampler linearSampler, in vec2 uv, in vec2 texSize)
{
    // We're going to sample a a 4x4 grid of texels surrounding the target UV coordinate. We'll do this by rounding
    // down the sample location to get the exact center of our "starting" texel. The starting texel will be at
    // location [1, 1] in the grid, where [0, 0] is the top left corner.
    vec2 samplePos = uv * texSize;
    vec2 texPos1 = floor(samplePos - 0.5f) + 0.5f;

    // Compute the fractional offset from our starting texel to our original sample location, which we'll
    // feed into the Catmull-Rom spline function to get our filter weights.
    vec2 f = samplePos - texPos1;

    // Compute the Catmull-Rom weights using the fractional offset that we calculated earlier.
    // These equations are pre-expanded based on our knowledge of where the texels will be located,
    // which lets us avoid having to evaluate a piece-wise function.
    vec2 w0 = f * (-0.5f + f * (1.0f - 0.5f * f));
    vec2 w1 = 1.0f + f * f * (-2.5f + 1.5f * f);
    vec2 w2 = f * (0.5f + f * (2.0f - 1.5f * f));
    vec2 w3 = f * f * (-0.5f + 0.5f * f);

    // Work out weighting factors and sampling offsets that will let us use bilinear filtering to
    // simultaneously evaluate the middle 2 samples from the 4x4 grid.
    vec2 w12 = w1 + w2;
    vec2 offset12 = w2 / (w1 + w2);

    // Compute the final UV coordinates we'll use for sampling the texture
    vec2 texPos0 = texPos1 - 1;
    vec2 texPos3 = texPos1 + 2;
    vec2 texPos12 = texPos1 + offset12;

    texPos0 /= texSize;
    texPos3 /= texSize;
    texPos12 /= texSize;

    vec4 result = vec4(0.0f);
    result += texture(sampler2D(tex, linearSampler), vec2(texPos0.x,  texPos0.y)) * w0.x * w0.y;
    result += texture(sampler2D(tex, linearSampler), vec2(texPos12.x, texPos0.y)) * w12.x * w0.y;
    result += texture(sampler2D(tex, linearSampler), vec2(texPos3.x,  texPos0.y)) * w3.x * w0.y;

    result += texture(sampler2D(tex, linearSampler), vec2(texPos0.x,  texPos12.y)) * w0.x * w12.y;
    result += texture(sampler2D(tex, linearSampler), vec2(texPos12.x, texPos12.y)) * w12.x * w12.y;
    result += texture(sampler2D(tex, linearSampler), vec2(texPos3.x,  texPos12.y)) * w3.x * w12.y;

    result += texture(sampler2D(tex, linearSampler), vec2(texPos0.x,  texPos3.y)) * w0.x * w3.y;
    result += texture(sampler2D(tex, linearSampler), vec2(texPos12.x, texPos3.y)) * w12.x * w3.y;
    result += texture(sampler2D(tex, linearSampler), vec2(texPos3.x,  texPos3.y)) * w3.x * w3.y;

    return result;
}

// https://gist.github.com/Fewes/59d2c831672040452aa77da6eaab2234
vec4 textureTricubic(in texture3D tex, in sampler linearSampler, vec3 coord)
{

	// Shift the coordinate from [0,1] to [-0.5, texture_size-0.5]
    vec3 texture_size = vec3(textureSize(tex, 0));
	vec3 coord_grid = coord * texture_size - 0.5;
	vec3 index = floor(coord_grid);
	vec3 fraction = coord_grid - index;
	vec3 one_frac = 1.0 - fraction;

	vec3 w0 = 1.0/6.0 * one_frac*one_frac*one_frac;
	vec3 w1 = 2.0/3.0 - 0.5 * fraction*fraction*(2.0-fraction);
	vec3 w2 = 2.0/3.0 - 0.5 * one_frac*one_frac*(2.0-one_frac);
	vec3 w3 = 1.0/6.0 * fraction*fraction*fraction;

	vec3 g0 = w0 + w1;
	vec3 g1 = w2 + w3;
	vec3 mult = 1.0 / texture_size;
	vec3 h0 = mult * ((w1 / g0) - 0.5 + index); //h0 = w1/g0 - 1, move from [-0.5, texture_size-0.5] to [0,1]
	vec3 h1 = mult * ((w3 / g1) + 1.5 + index); //h1 = w3/g1 + 1, move from [-0.5, texture_size-0.5] to [0,1]

	// Fetch the eight linear interpolations
	// Weighting and fetching is interleaved for performance and stability reasons
	vec4 tex000 = texture(sampler3D(tex, linearSampler), h0);
	vec4 tex100 = texture(sampler3D(tex, linearSampler), vec3(h1.x, h0.y, h0.z));
	tex000 = mix(tex100, tex000, g0.x); // Weigh along the x-direction

	vec4 tex010 = texture(sampler3D(tex, linearSampler), vec3(h0.x, h1.y, h0.z));
	vec4 tex110 = texture(sampler3D(tex, linearSampler), vec3(h1.x, h1.y, h0.z));
	tex010 = mix(tex110, tex010, g0.x); // Weigh along the x-direction
	tex000 = mix(tex010, tex000, g0.y); // Weigh along the y-direction

	vec4 tex001 = texture(sampler3D(tex, linearSampler), vec3(h0.x, h0.y, h1.z));
	vec4 tex101 = texture(sampler3D(tex, linearSampler), vec3(h1.x, h0.y, h1.z));
	tex001 = mix(tex101, tex001, g0.x); // Weigh along the x-direction

	vec4 tex011 = texture(sampler3D(tex, linearSampler), vec3(h0.x, h1.y, h1.z));
	vec4 tex111 = texture(sampler3D(tex, linearSampler), vec3(h1));
	tex011 = mix(tex111, tex011, g0.x); // Weigh along the x-direction
	tex001 = mix(tex011, tex001, g0.y); // Weigh along the y-direction

	return mix(tex001, tex000, g0.z); // Weigh along the z-direction
}

// Terrain


float getTerrainLODSizeFromLOD(uint lod)
{
    return exp2(lod) * kTerrainLowestNodeDim;
}

#endif