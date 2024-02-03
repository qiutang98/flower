#ifndef SHARED_LIGHTING_GLSL
#define SHARED_LIGHTING_GLSL

#include "common_shader.glsl"

// Physical based lighting collections.
// http://blog.selfshadow.com/publications/s2012-shading-course/burley/s2012_pbs_disney_brdf_notes_v3.pdf
// https://github.com/GPUOpen-Effects/FidelityFX-SSSR
// http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf
// https://www.cs.virginia.edu/~jdl/bib/appearance/analytic%20models/schlick94b.pdf

// PBR material info to evaluate shade.
struct PBRMaterial
{
    EShadingModelType shadingModel;

    // perceptualRoughness is texture sample value.
    float perceptualRoughness; 

    // alphaRoughness = perceptualRoughness * perceptualRoughness;
    float alphaRoughness; 

    vec3 diffuseColor; 
    vec3 specularColor;  

    vec3 reflectance0;
    vec3 reflectance90;   

    vec3  baseColor;
    float curvature;
};

// Light info mix.
struct AngularInfo
{
    float NdotL;  
    float NdotV;  
    float NdotH;  
    float LdotH; 
    float VdotH; 
};

AngularInfo getAngularInfo(vec3 pointToLight, vec3 normal, vec3 view)
{
    AngularInfo result;

    // Standard one-letter names
    vec3 n = normalize(normal);           // Outward direction of surface point
    vec3 v = normalize(view);             // Direction from surface point to view
    vec3 l = normalize(pointToLight);     // Direction from surface point to light
    vec3 h = normalize(l + v);            // Direction of the vector between l and v

    result.NdotL = clamp(dot(n, l), 0.0, 1.0);
    result.NdotV = clamp(dot(n, v), 0.0, 1.0);
    result.NdotH = clamp(dot(n, h), 0.0, 1.0);
    result.LdotH = clamp(dot(l, h), 0.0, 1.0);
    result.VdotH = clamp(dot(v, h), 0.0, 1.0);

    return result;
}

// https://github.com/KhronosGroup/glTF/blob/master/extensions/2.0/Khronos/KHR_lights_punctual/README.md#range-property
float getRangeAttenuation(float range, float distance)
{
    if (range < 0.0)
    {
        // negative range means unlimited
        return 1.0;
    }
    return max(mix(1, 0, distance / range), 0);
    //return max(min(1.0 - pow(distance / range, 4.0), 1.0), 0.0) / pow(distance, 2.0);
}

// https://github.com/KhronosGroup/glTF/blob/master/extensions/2.0/Khronos/KHR_lights_punctual/README.md#inner-and-outer-cone-angles
float getSpotAttenuation(vec3 pointToLight, vec3 spotDirection, float outerConeCos, float innerConeCos)
{
    float actualCos = dot(normalize(spotDirection), normalize(-pointToLight));
    if (actualCos > outerConeCos)
    {
        if (actualCos < innerConeCos)
        {
            return smoothstep(outerConeCos, innerConeCos, actualCos);
        }
        return 1.0;
    }
    return 0.0;
}

// The following equation models the Fresnel reflectance term of the spec equation (aka F())
// Implementation of fresnel from [4], Equation 15
vec3 F_Schlick(vec3 f0, vec3 f90, float u) 
{
    return f0 + (f90 - f0) * pow(clamp(1.0 - u, 0.0, 1.0), 5.0);
}

vec3 F_SchlickFast(vec3 f0, float u) 
{
    float f = pow(1.0 - u, 5.0);
    return f + f0 * (1.0 - f);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
// Diffuse term start.

// Lambert lighting
// see https://seblagarde.wordpress.com/2012/01/08/pi-or-not-to-pi-in-game-lighting-equation/
vec3 Fd_LambertDiffuse(PBRMaterial materialInfo)
{
    return materialInfo.diffuseColor / kPI;
}

// NOTE: Burley diffuse is expensive, and only add slightly image quality improvement.
//       We default use lambert diffuse. 
// Burley 2012, "Physically-Based Shading at Disney"
vec3 Fd_BurleyDiffuse(PBRMaterial materialInfo, AngularInfo angularInfo) 
{
    float f90 = 0.5 + 2.0 * materialInfo.alphaRoughness * angularInfo.LdotH * angularInfo.LdotH;
    vec3 lightScatter = F_Schlick(vec3(1.0), vec3(f90), angularInfo.NdotL);
    vec3 viewScatter  = F_Schlick(vec3(1.0), vec3(f90), angularInfo.NdotV);
    return materialInfo.diffuseColor * lightScatter * viewScatter * (1.0 / kPI);
}

// Diffuse term end.
/////////////////////////////////////////////////////////////////////////////////////////////////////


vec3 specularReflection(PBRMaterial materialInfo, AngularInfo angularInfo)
{
    return F_Schlick(materialInfo.reflectance0, materialInfo.reflectance90, angularInfo.VdotH);
}

//////////////////////////////////////////////////////////////////////////////////////////
// Geometry visibility item start.

// Smith Joint GGX
// Note: Vis = G / (4 * NdotL * NdotV)
// see Eric Heitz. 2014. Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs. Journal of Computer Graphics Techniques, 3
// see Real-Time Rendering. Page 331 to 336.
// see https://google.github.io/filament/Filament.md.html#materialsystem/specularbrdf/geometricshadowing(specularg)
float V_SmithGGXCorrelated(float NoV, float NoL, float alphaRoughness) 
{
    float a2 = alphaRoughness * alphaRoughness;

    float GGXV = NoL * sqrt(NoV * NoV * (1.0 - a2) + a2);
    float GGXL = NoV * sqrt(NoL * NoL * (1.0 - a2) + a2);

    float GGX = GGXV + GGXL;
    return (GGX > 0.0) ? (0.5 / GGX) : 0.0;
}

// Fast approximate for GGX.
float V_SmithGGXCorrelatedFast(float NoV, float NoL, float alphaRoughness) 
{
    float a = alphaRoughness;
    float GGXV = NoL * (NoV * (1.0 - a) + a);
    float GGXL = NoV * (NoL * (1.0 - a) + a);
    return 0.5 / (GGXV + GGXL);
}

float visibilityOcclusion(PBRMaterial materialInfo, AngularInfo angularInfo)
{
    return V_SmithGGXCorrelated(angularInfo.NdotL, angularInfo.NdotV, materialInfo.alphaRoughness);
}

// Geometry visibility item end.
//////////////////////////////////////////////////////////////////////////////////////////


// The following equation(s) model the distribution of microfacet normals across the area being drawn (aka D())
// Implementation from "Average Irregularity Representation of a Roughened Surface for Ray Reflection" by T. S. Trowbridge, and K. P. Reitz
// Follows the distribution function recommended in the SIGGRAPH 2013 course notes from EPIC Games [1], Equation 3.
float D_GGX(float NoH, float alphaRoughness) 
{
    float a2 = alphaRoughness * alphaRoughness;
    float f = (NoH * a2 - NoH) * NoH + 1.0;
    return a2 / (kPI * f * f + 0.000001f);
}

float microfacetDistribution(PBRMaterial materialInfo, AngularInfo angularInfo)
{
    return D_GGX(angularInfo.NdotH, materialInfo.alphaRoughness);
}

struct ShadingResult
{
    vec3 diffuseTerm;
    vec3 specularTerm;
};

ShadingResult getDefaultShadingResult()
{
    ShadingResult result;

    result.diffuseTerm = vec3(0.0);
    result.specularTerm = vec3(0.0);

    return result;
}

ShadingResult getPointShade(vec3 pointToLight, PBRMaterial materialInfo, vec3 normal, vec3 view)
{
    ShadingResult result = getDefaultShadingResult();
    AngularInfo angularInfo = getAngularInfo(pointToLight, normal, view);

    if(materialInfo.shadingModel == EShadingModelType_DefaultLit)
    {
        // Skip unorientation to light pixels.
        if (angularInfo.NdotL > 0.0 || angularInfo.NdotV > 0.0)
        {
            // Calculate the shading terms for the microfacet specular shading model
            vec3  F   = specularReflection(materialInfo, angularInfo);
            float Vis = visibilityOcclusion(materialInfo, angularInfo);
            float D   = microfacetDistribution(materialInfo, angularInfo);

            // Calculation of analytical lighting contribution
            vec3 diffuseContrib = (1.0 - F) * Fd_LambertDiffuse(materialInfo);
            vec3 specContrib    = F * Vis * D;

            // Obtain final intensity as reflectance (BRDF) scaled by the energy of the light (cosine law)

            result.diffuseTerm = angularInfo.NdotL * diffuseContrib;
            result.specularTerm = angularInfo.NdotL * specContrib;
        }
    }

    return result;
}

// Directional light direct lighting evaluate.
ShadingResult evaluateSkyDirectLight(SkyLightInfo sky, PBRMaterial materialInfo, vec3 normal, vec3 view)
{
    // Directional lighting direction is light pos to camera pos normalize vector.
    // So need inverse here for point to light.
    vec3 pointToLight = normalize(-sky.direction);

    ShadingResult shade = getPointShade(pointToLight, materialInfo, normal, view);
    shade.diffuseTerm  *= sky.intensity * sky.color;
    shade.specularTerm *= sky.intensity * sky.color;

    return  shade;
}

float specularAOLagarde(float NoV, float visibility, float roughness) 
{
    // Lagarde and de Rousiers 2014, "Moving Frostbite to PBR"
    return saturate(pow(NoV + visibility, exp2(-16.0 * roughness - 1.0)) - 1.0 + visibility);
}

// Importance sampling

// https://www.mathematik.uni-marburg.de/~thormae/lectures/graphics1/code/ImportanceSampling/importance_sampling_notes.pdf
// Based on http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_slides.pdf
// https://bruop.github.io/ibl/
vec3 importanceSampleGGX(vec2 Xi, float alphaRoughness, vec3 normal) 
{
	// Maps a 2D point to a hemisphere with spread based on roughness.
	float alpha = alphaRoughness * alphaRoughness;

    // Sample in spherical coordinates.
    float phi = 2.0 * kPI * Xi.x;
	float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (alpha * alpha - 1.0) * Xi.y));
	float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    // Construct tangent space sample vector.
	vec3 H = vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

	// Tangent space
	vec3 up = abs(normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
	vec3 tangentX = normalize(cross(up, normal));
	vec3 tangentY = normalize(cross(normal, tangentX));

	// Convert to world Space
	return normalize(tangentX * H.x + tangentY * H.y + normal * H.z);
}

// http://jcgt.org/published/0007/04/01/paper.pdf by Eric Heitz
// Input Ve: view direction
// Input alpha_x, alpha_y: roughness parameters
// Input U1, U2: uniform random numbers
// Output Ne: normal sampled with PDF D_Ve(Ne) = G1(Ve) * max(0, dot(Ve, Ne)) * D(Ne) / Ve.z
vec3 importanceSampleGGXVNDF(vec3 Ve, float alpha_x, float alpha_y, float U1, float U2) 
{
    // Section 3.2: transforming the view direction to the hemisphere configuration
    vec3 Vh = normalize(vec3(alpha_x * Ve.x, alpha_y * Ve.y, Ve.z));

    // Section 4.1: orthonormal basis (with special case if cross product is zero)
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    vec3 T1 = lensq > 0 ? vec3(-Vh.y, Vh.x, 0) * inversesqrt(lensq) : vec3(1, 0, 0);
    vec3 T2 = cross(Vh, T1);

    // Section 4.2: parameterization of the projected area
    float r = sqrt(U1);
    float phi = 2.0 * kPI * U2;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;

    // Section 4.3: reprojection onto hemisphere
    vec3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;

    // Section 3.4: transforming the normal back to the ellipsoid configuration
    vec3 Ne = normalize(vec3(alpha_x * Nh.x, alpha_y * Nh.y, max(0.0, Nh.z)));
    return Ne;
}

// Importance sample use cosine weight.
vec3 importanceSampleCosine(vec2 xi, vec3 N)
{
    float phi = 2.f * kPI * xi.y;

    float cosTheta = sqrt(xi.x);
    float sinTheta = sqrt(1 - xi.x);
    vec3 sampleHemisphere = vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);

    //orient sample into world space
    vec3 up = abs(N.z) < 0.999 ? vec3(0.f, 0.f, 1.f) : vec3(1.f, 0.f, 0.f);
    vec3 tangent = normalize(cross(up, N));
    vec3 bitangent = cross(N, tangent);

    vec3 sampleWorld = vec3(0);
    sampleWorld += sampleHemisphere.x * tangent;
    sampleWorld += sampleHemisphere.y * bitangent;
    sampleWorld += sampleHemisphere.z * N;

    return sampleWorld;
}

vec3 sphericalToCartesian(float phi, float cosTheta)
{
    float sinPhi = sin(phi);
    float cosPhi = cos(phi);

    float sinTheta = sqrt(saturate(1.0 - cosTheta * cosTheta));
    return vec3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
}

vec3 sampleSphereUniform(float u1, float u2)
{
    float phi = 6.28318530718f * u2;
    float cosTheta = 1.0 - 2.0 * u1;

    return sphericalToCartesian(phi, cosTheta);
}

vec3 sampleHemisphereCosine(float u1, float u2, vec3 normal)
{
    vec3 pointOnSphere = sampleSphereUniform(u1, u2);
    return normalize(normal + pointOnSphere);
}

#endif