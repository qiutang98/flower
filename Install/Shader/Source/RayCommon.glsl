#ifndef RAY_COMMON_GLSL
#define RAY_COMMON_GLSL

#include "Common.glsl"

struct Ray
{
	vec3 o;
	vec3 d;
};

Ray createRay(in vec3 p, in vec3 d)
{
	Ray r;
	r.o = p;
	r.d = d;
	return r;
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
	, ViewData viewData) 
{
    mat4 worldToClip = viewData.camViewProj;
    mat4 viewToClip = viewData.camProj;

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

#endif