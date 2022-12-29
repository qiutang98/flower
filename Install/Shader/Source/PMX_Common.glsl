#ifndef PMX_COMMON_GLSL
#define PMX_COMMON_GLSL

#include "Common.glsl"
#include "LightingCommon.glsl"

struct UniformPMX
{
    mat4 modelMatrix;
    mat4 modelMatrixPrev;
    uint texID;
    uint spTexID;
    uint toonTexID;
    uint pmxObjectID;
};

struct AngularInfoPMX
{
    vec3 n;
    vec3 v;
    vec3 l;
    vec3 h;

    float NdotL;  
    float NdotV;  
    float NdotH;  
    float LdotH; 
    float VdotH; 
    float halfLambert;

    float NdotLSafe;  
    float NdotVSafe;    
    float NdotHSafe;    
    float LdotHSafe;   
    float VdotHSafe;  
};

AngularInfoPMX getAngularInfoPMX(vec3 pointToLight, vec3 normal, vec3 view)
{
    AngularInfoPMX result;

    // Standard one-letter names
    result.n = normalize(normal);           // Outward direction of surface point
    result.v = normalize(view);             // Direction from surface point to view
    result.l = normalize(pointToLight);     // Direction from surface point to light
    result.h = normalize(result.l + result.v);            // Direction of the vector between l and v

    result.NdotL = dot(result.n, result.l);
    result.NdotV = dot(result.n, result.v);
    result.NdotH = dot(result.n, result.h);
    result.LdotH = dot(result.l, result.h);
    result.VdotH = dot(result.v, result.h);

    result.halfLambert = result.NdotL * 0.5 + 0.5;

    result.NdotLSafe = clamp(result.NdotL, 0.0, 1.0);
    result.NdotVSafe = clamp(result.NdotV, 0.0, 1.0);
    result.NdotHSafe = clamp(result.NdotH, 0.0, 1.0);
    result.LdotHSafe = clamp(result.LdotH, 0.0, 1.0);
    result.VdotHSafe = clamp(result.VdotH, 0.0, 1.0);

    return result;
}

#endif