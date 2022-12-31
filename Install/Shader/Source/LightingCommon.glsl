#ifndef LIGHTING_COMMON_GLSL
#define LIGHTING_COMMON_GLSL

/*
** Physical based render code, develop by engineer: qiutanguu.
*/

#include "Common.glsl"
#include "FastMath.glsl"

// Physical based lighting collections.
// http://blog.selfshadow.com/publications/s2012-shading-course/burley/s2012_pbs_disney_brdf_notes_v3.pdf
// https://github.com/GPUOpen-Effects/FidelityFX-SSSR
// http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf
// https://www.cs.virginia.edu/~jdl/bib/appearance/analytic%20models/schlick94b.pdf

// PBR material info to evaluate shade.
struct PBRMaterial
{
    float perceptualRoughness; // perceptualRoughness is texture sample value.
    float alphaRoughness; // alphaRoughness = perceptualRoughness * perceptualRoughness;

    vec3 diffuseColor; 
    vec3 specularColor;  

    vec3 reflectance0;
    vec3 reflectance90;   

    float meshAo;
    float diffuseSSAO;
    vec3 bentNormalViewspace;  
    vec3 bentNormalWorldspace;
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


vec3 getPointShade(vec3 pointToLight, PBRMaterial materialInfo, vec3 normal, vec3 view)
{
    AngularInfo angularInfo = getAngularInfo(pointToLight, normal, view);

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
        return angularInfo.NdotL * (diffuseContrib + specContrib);
    }

    return vec3(0.0, 0.0, 0.0);
}

// Directional light direct lighting evaluate.
vec3 evaluateDirectionalLight(DirectionalLightInfo light, PBRMaterial materialInfo, vec3 normal, vec3 view)
{
    // Directional lighting direction is light pos to camera pos normalize vector.
    // So need inverse here for point to light.
    vec3 pointToLight = -light.direction;

    vec3 shade = getPointShade(pointToLight, materialInfo, normal, view);
    return light.intensity * light.color * shade;
}

// Spot light direct lighting evaluate.
vec3 evaluateSpotLight(LocalSpotLightInfo light, PBRMaterial materialInfo, vec3 normal, vec3 worldPos, vec3 view)
{
    vec3 pointToLight = light.position - worldPos;
    float d = length(pointToLight);

    float rangeAttenuation = getRangeAttenuation(light.range, d);
    float spotAttenuation  = getSpotAttenuation(pointToLight, -light.direction, light.outerConeCos, light.innerConeCos);

    vec3 shade = getPointShade(pointToLight, materialInfo, normal, view);
    return rangeAttenuation * spotAttenuation * light.intensity * light.color * shade;
}

float specularAOLagarde(float NoV, float visibility, float roughness) 
{
    // Lagarde and de Rousiers 2014, "Moving Frostbite to PBR"
    return saturate(pow(NoV + visibility, exp2(-16.0 * roughness - 1.0)) - 1.0 + visibility);
}


#endif