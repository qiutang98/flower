#ifndef SHARED_FUNCTIONS_GLSL
#define SHARED_FUNCTIONS_GLSL

#extension GL_EXT_samplerless_texture_functions : require

#include "shared_struct.glsl"
#include "shared_aces.glsl"
#include "shared_shading_model.glsl"

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

// Project position to uv space.
vec3 projectPos(vec3 origin, in const mat4 inMatrix)
{
    vec4 projectPos = inMatrix * vec4(origin, 1.0);
    projectPos.xyz /= projectPos.w;

    projectPos.xy = 0.5 * projectPos.xy + 0.5;
    projectPos.y  = 1.0 - projectPos.y;

    return projectPos.xyz;
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


// Quad schedule style, from AMD FSR.
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

uint packToIdBuffer(uint sceneNodeId, uint bSelected)
{
    uint idBuffer = (sceneNodeId & 0xffffffff) << 1;
    idBuffer |= ((bSelected & 0xffffffff) & 0x01);
    return idBuffer; 
}

uint unpackToSceneNodeId(uint value)
{
    return (value & 0xffffffff) >> 1;
}

uint unpackToSceneNodeSelected(uint value)
{
    return (value & 0xffffffff) & 0x01;
}

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

// GT tonemapper, flexibility to hdr and ldr.
///////////////////////////////////////////////////////// 

float uchimuraTonemapper(float x, float P, float a, float m, float l, float c, float b) {
    // Uchimura 2017, "HDR theory and practice"
    // Math: https://www.desmos.com/calculator/gslcdxvipg
    // Source: https://www.slideshare.net/nikuque/hdr-theory-and-practicce-jp
    float l0 = ((P - m) * l) / a;
    float L0 = m - m / a;
    float L1 = m + (1.0 - m) / a;
    float S0 = m + l0;
    float S1 = m + a * l0;
    float C2 = (a * P) / (P - S1);
    float CP = -C2 / P;

    float w0 = 1.0 - smoothstep(0.0, m, x);
    float w2 = step(m + l0, x);
    float w1 = 1.0 - w0 - w2;

    float T = m * pow(x / m, c) + b;
    float S = P - (P - S1) * exp(CP * (x - S0));
    float L = m + a * (x - m);

    return T * w0 + L * w1 + S * w2;
}

//////////////////////////////////////////////

//--------------------------------------------------------------------------------------
// AMD Tonemapper
//--------------------------------------------------------------------------------------
// General tonemapping operator, build 'b' term.
float ColToneB(float hdrMax, float contrast, float shoulder, float midIn, float midOut) 
{
    return
        -((-pow(midIn, contrast) + (midOut*(pow(hdrMax, contrast * shoulder) * pow(midIn, contrast) -
            pow(hdrMax, contrast) * pow(midIn, contrast*shoulder) * midOut)) /
            (pow(hdrMax, contrast * shoulder) * midOut - pow(midIn, contrast * shoulder) * midOut)) /
            (pow(midIn, contrast * shoulder) * midOut));
}

// General tonemapping operator, build 'c' term.
float ColToneC(float hdrMax, float contrast, float shoulder, float midIn, float midOut) 
{
    return (pow(hdrMax, contrast*shoulder)*pow(midIn, contrast) - pow(hdrMax, contrast)*pow(midIn, contrast*shoulder)*midOut) /
           (pow(hdrMax, contrast*shoulder)*midOut - pow(midIn, contrast*shoulder)*midOut);
}

// General tonemapping operator, p := {contrast,shoulder,b,c}.
float ColTone(float x, vec4 p) 
{ 
    float z = pow(x, p.r); 
    return z / (pow(z, p.g)*p.b + p.a); 
}

vec3 AMDTonemapper(vec3 color)
{
    const float hdrMax = 25.0; // How much HDR range before clipping. HDR modes likely need this pushed up to say 25.0.
    const float contrast = 2.0; // Use as a baseline to tune the amount of contrast the tonemapper has.
    const float shoulder = 1.0; // Likely don't need to mess with this factor, unless matching existing tonemapper is not working well..
    const float midIn = 0.18; // most games will have a {0.0 to 1.0} range for LDR so midIn should be 0.18.
    const float midOut = 0.18; // Use for LDR. For HDR10 10:10:10:2 use maybe 0.18/25.0 to start. For scRGB, I forget what a good starting point is, need to re-calculate.

    float b = ColToneB(hdrMax, contrast, shoulder, midIn, midOut);
    float c = ColToneC(hdrMax, contrast, shoulder, midIn, midOut);

    #define EPS 1e-6f
    float peak = max(color.r, max(color.g, color.b));
    peak = max(EPS, peak);

    vec3 ratio = color / peak;
    peak = ColTone(peak, vec4(contrast, shoulder, b, c) );
    // then process ratio

    // probably want send these pre-computed (so send over saturation/crossSaturation as a constant)
    float crosstalk = 4.0; // controls amount of channel crosstalk
    float saturation = contrast; // full tonal range saturation control
    float crossSaturation = contrast*16.0; // crosstalk saturation

    float white = 1.0;

    // wrap crosstalk in transform
    ratio = pow(abs(ratio), vec3(saturation / crossSaturation));
    ratio = mix(ratio, vec3(white), vec3(pow(peak, crosstalk)));
    ratio = pow(abs(ratio), vec3(crossSaturation));

    // then apply ratio to peak
    color = peak * ratio;
    return color;
}

/////////////////////////////////////////////////////////////////////////////////////////

// Lottes 2016, "Advanced Techniques and Optimization of HDR Color Pipelines" - https://gpuopen.com/wp-content/uploads/2016/03/GdcVdrLottes.pdf
float TonemapLottes(float x, float contrast)
{
    const float a = 1.6 * contrast;
    const float d = 0.977;
    const float hdrMax = 8.0;
    const float midIn = 0.18;
    const float midOut = 0.267;

    // Can be precomputed
    const float b =
        (-pow(midIn, a) + pow(hdrMax, a) * midOut) /
        ((pow(hdrMax, a * d) - pow(midIn, a * d)) * midOut);
    const float c =
        (pow(hdrMax, a * d) * pow(midIn, a) - pow(hdrMax, a) * pow(midIn, a * d) * midOut) /
        ((pow(hdrMax, a * d) - pow(midIn, a * d)) * midOut);

    return pow(x, a) / (pow(x, a * d) * b + c);
}

vec3 TonemapLottesx3(vec3 c, float contrast) 
{
    return vec3(
        TonemapLottes(c.x, contrast), 
        TonemapLottes(c.y, contrast), 
        TonemapLottes(c.z, contrast)
    );
}

///////////////////////////////////////////////////////////////////

//== ACESFitted ===========================
//  Baking Lab
//  by MJP and David Neubelt
//  http://mynameismjp.wordpress.com/
//  All code licensed under the MIT license
//=========================================

// sRGB => XYZ => D65_2_D60 => AP1 => RRT_SAT
const mat3 ACESInputMat = mat3
(
    0.59719, 0.07600, 0.02840,
    0.35458, 0.90834, 0.13383,
    0.04823, 0.01566, 0.83777
);

// ODT_SAT => XYZ => D60_2_D65 => sRGB
const mat3 ACESOutputMat = mat3
(
    1.60475, -0.10208, -0.00327,
    -0.53108,  1.10813, -0.07276,
    -0.07367, -0.00605,  1.07602
);

// ACES filmic tone map approximation
// see https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
vec3 RRTAndODTFit(vec3 color)
{
    vec3 a = color * (color + 0.0245786) - 0.000090537;
    vec3 b = color * (0.983729 * color + 0.4329510) + 0.238081;
    return a / b;
}

// Tone mapping 
vec3 acesFit(vec3 color)
{
    color = ACESInputMat * color;

    // Apply RRT and ODT
    color = RRTAndODTFit(color);
    color = ACESOutputMat * color;

    // Clamp to [0, 1]
    color = clamp(color, 0.0, 1.0);
    return color;
}

vec3 acesFilmFit(vec3 x)
{
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return saturate((x*(a*x+b))/(x*(c*x+d)+e));
}

// Gamma curve encode to srgb.
vec3 encodeSRGB(vec3 linearRGB)
{
    // Most PC Monitor is 2.2 Gamma, this function is enough.
    return pow(linearRGB, vec3(1.0 / 2.2));

    // TV encode Rec709 encode.
    vec3 a = 12.92 * linearRGB;
    vec3 b = 1.055 * pow(linearRGB, vec3(1.0 / 2.4)) - 0.055;
    vec3 c = step(vec3(0.0031308), linearRGB);
    return mix(a, b, c);
}

// PQ curve encode to ST2084.
vec3 encodeST2084(vec3 linearRGB)
{
	const float m1 = 2610. / 4096. * .25;
	const float m2 = 2523. / 4096. *  128;
	const float c1 = 2392. / 4096. * 32 - 2413. / 4096. * 32 + 1;
	const float c2 = 2413. / 4096. * 32;
	const float c3 = 2392. / 4096. * 32;

    // Standard encode 10000 nit, same with standard DaVinci Resolve nit.
    float C = 10000.0;

	   vec3 L = linearRGB / C;
    // vec3 L = linearRGB;

	vec3 Lm = pow(L, vec3(m1));
	vec3 N1 = (c1 + c2 * Lm);
	vec3 N2 = (1.0 + c3 * Lm);
	vec3 N = N1 / N2;

	vec3 P = pow(N, vec3(m2));
	return P;
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
vec3 AoMultiBounce(float AO, vec3 baseColor)
{
    vec3 a =  2.0404 * baseColor - 0.3324;
    vec3 b = -4.7951 * baseColor + 0.6417;
    vec3 c =  2.7552 * baseColor + 0.6903;

    vec3 x  = vec3(AO);

    return max(x, ((x * a + b) * x + c) * x);
}

float getExposure(in const PerFrameData inFrame, texture2D tex)
{
    if(inFrame.bAutoExposure != 0)
    {
        return texelFetch(tex, ivec2(0, 0), 0).r;
    }
    else
    {
        return inFrame.fixExposure;
    }
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

// 4x4 bayer filter, use for cloud reconstruction.
int kBayerMatrix16[16] = int[16]
(
     0,  8,  2, 10, 
    12,  4, 14,  6, 
     3, 11,  1,  9, 
    15,  7, 13,  5
);
// ivec2 offset = ivec2(kBayerMatrix16[frameId] % 4, kBayerMatrix16[frameId] / 4);
float bayerDither(float grayscale, ivec2 pixelCoord)
{    
    int pixelIndex16 = (pixelCoord.x % 4) + (pixelCoord.y % 4) * 4;
    return grayscale > (float(kBayerMatrix16[pixelIndex16]) + 0.5) / 16.0 ? 1.0 : 0.0;
}

float bayer2(vec2 a) {
    a = floor(a);

    return fract(dot(a, vec2(0.5, a.y * 0.75)));
}

float bayer4(const vec2 a)   { return bayer2 (0.5   * a) * 0.25     + bayer2(a); }
float bayer8(const vec2 a)   { return bayer4 (0.5   * a) * 0.25     + bayer2(a); }
float bayer16(const vec2 a)  { return bayer4 (0.25  * a) * 0.0625   + bayer4(a); }
float bayer32(const vec2 a)  { return bayer8 (0.25  * a) * 0.0625   + bayer4(a); }
float bayer64(const vec2 a)  { return bayer8 (0.125 * a) * 0.015625 + bayer8(a); }
float bayer128(const vec2 a) { return bayer16(0.125 * a) * 0.015625 + bayer8(a); }

#define dither2(p)   (bayer2(  p) - 0.375      )
#define dither4(p)   (bayer4(  p) - 0.46875    )
#define dither8(p)   (bayer8(  p) - 0.4921875  )
#define dither16(p)  (bayer16( p) - 0.498046875)
#define dither32(p)  (bayer32( p) - 0.499511719)
#define dither64(p)  (bayer64( p) - 0.49987793 )
#define dither128(p) (bayer128(p) - 0.499969482)



vec3 reinhardInverse(in vec3 sdr)
{
    return sdr / max(1.0f - sdr, 1e-2f);
}

vec3 reinhard(in vec3 hdr)
{
    return hdr / (hdr + 1.0f);
}

float reinhardInverse(in float sdr)
{
    return sdr / max(1.0f - sdr, 1e-2f);
}

float reinhard(in float hdr)
{
    return hdr / (hdr + 1.0f);
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

#endif