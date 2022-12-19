#version 460

/*
** Physical based render code, develop by engineer: qiutanguu.
*/

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable

// Cloud shadowmap evaluate from directional light direction.
#include "Cloud_Common.glsl"
#include "Noise.glsl"

// Cloud shadow map, use for ESM.
layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    ivec2 texSize = imageSize(imageCloudShadowDepth);
    ivec2 workPos = ivec2(gl_GlobalInvocationID.xy);

    if(workPos.x >= texSize.x || workPos.y >= texSize.y)
    {
        return;
    }
    AtmosphereParameters atmosphere = getAtmosphereParameters(frameData);

    const vec2 uv = (vec2(workPos) + vec2(0.5f)) / vec2(texSize);
    
    // Reproject to directional light space, then evaluate cloud transmittance and store it's depth.
    vec4 posClip  = vec4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, 1.0f, 1.0f); // We reverse z, from near plane.
    vec4 posWorldRebuild = atmosphere.cloudShadowViewProjInverse * posClip;

    // Start pos to marching.
    vec3 marchingStartWorldPos = posWorldRebuild.xyz / posWorldRebuild.w;

    vec4 posClipEnd  = vec4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, 0.0f, 1.0f); // We reverse z, from near plane.
    vec4 posWorldRebuildEnd = atmosphere.cloudShadowViewProjInverse * posClipEnd;

    vec3 marchingEndWorldPos = posWorldRebuildEnd.xyz / posWorldRebuildEnd.w;

    // from Sun to earth.
    vec3 marchingDirection = normalize(marchingEndWorldPos - marchingStartWorldPos);

    float earthRadius = atmosphere.bottomRadius;
    float radiusCloudStart = atmosphere.cloudAreaStartHeight;
    float radiusCloudEnd = radiusCloudStart + atmosphere.cloudAreaThickness;

    float viewHeight = length(marchingStartWorldPos);
    // Find intersect position so we can do some ray marching.
    float tMin;
    float tMax;
    {
        // Eye out of cloud area.
        vec2 t0t1 = vec2(0.0);
        const bool bIntersectionEnd = raySphereIntersectOutSide(marchingStartWorldPos, marchingDirection, vec3(0.0), radiusCloudEnd, t0t1);

        vec2 t2t3 = vec2(0.0);
        const bool bIntersectionStart = raySphereIntersectOutSide(marchingStartWorldPos, marchingDirection, vec3(0.0), radiusCloudStart, t2t3);
        if(bIntersectionStart)
        {
            tMin = t0t1.x;
            tMax = t2t3.x;
        }
        else
        {
            tMin = t0t1.x;
            tMax = t0t1.y;
        }
    }

    tMin = max(tMin, 0.0);
    tMax = max(tMax, 0.0);
    
    float transmittance = 1.0;
    float depth = tMax - tMin;
    float stepDepth = depth / 50;

    float t = tMin + stepDepth * 0.5f;
    while(transmittance > 0.001 && t < tMax)
    {
        vec3 samplePos = marchingStartWorldPos + marchingDirection * t;
        float normalizeHeight = (length(samplePos) - atmosphere.cloudAreaStartHeight) / atmosphere.cloudAreaThickness;

        float density = cloudMap(samplePos * 1000.0, normalizeHeight);
        if(density > 0.)
        {
            transmittance *= exp(-density * stepDepth * 1000.0); // To meter.
        }

        depth = mix(depth, t - tMin, pow(transmittance, 4.0));

        t += stepDepth;
    }

	imageStore(imageCloudShadowDepth, workPos, vec4(depth, 1.0, 1.0, 1.0));
}