#version 460

#extension GL_GOOGLE_include_directive : enable

// Capture 360 cube map with low resolution.
// Use for scene capture sphere fallback.

#include "UE4_AtmosphereCommon.glsl"


layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main()
{
    ivec3 cubeCoord = ivec3(gl_GlobalInvocationID);
    ivec2 cubeSize = imageSize(imageCubeEnv);

    if(cubeCoord.x >= cubeSize.x || cubeCoord.y >= cubeSize.y || cubeCoord.z >= 6)
    {
        return;
    }

    const vec2 pixPos = vec2(cubeCoord) + vec2(0.5f);
    const vec2 uv = pixPos / vec2(cubeSize);

    AtmosphereParameters atmosphere = getAtmosphereParameters();

    vec3 worldDir = getSamplingVector(cubeCoord.z, uv);
    const vec3 sunDirection = -normalize(frameData.directionalLight.direction);

    // Sample skyview lut and store in cubemap capture.
    vec3 worldPos = convertToAtmosphereUnit(viewData.camWorldPos.xyz) + vec3(0.0, atmosphere.bottomRadius, 0.0);
    float viewHeight = length(worldPos);

    // If camera out of atmosphere, no capture.
    const bool bCanUseSkyViewLut = viewHeight < atmosphere.topRadius;
    vec3 result = vec3(0.0);
    if (bCanUseSkyViewLut)
    {
        vec3 upVector = normalize(worldPos);
		float viewZenithCosAngle = dot(worldDir, upVector);

        // Assumes non parallel vectors
		vec3 sideVector = normalize(cross(upVector, worldDir));		

        // aligns toward the sun light but perpendicular to up vector
		vec3 forwardVector = normalize(cross(sideVector, upVector));	

		vec2 lightOnPlane = vec2(dot(sunDirection, forwardVector), dot(sunDirection, sideVector));
		lightOnPlane = normalize(lightOnPlane);
		float lightViewCosAngle = lightOnPlane.x;

		bool bIntersectGround = raySphereIntersectNearest(worldPos, worldDir, vec3(0.0), atmosphere.bottomRadius) >= 0.0f;

        vec2 sampleUv;
		skyViewLutParamsToUv(atmosphere, bIntersectGround, viewZenithCosAngle, lightViewCosAngle, viewHeight, sampleUv);

		result = texture(sampler2D(inSkyViewLut, linearClampEdgeSampler), sampleUv).rgb;
    }
    else
    {
        // Move to top atmosphere as the starting point for ray marching.
        // This is critical to be after the above to not disrupt above atmosphere tests and voxel selection.
        if (moveToTopAtmosphere(worldPos, worldDir, atmosphere.topRadius))
        {
            // Ray intersecting the atmosphere.	
            const bool bGround = false;
            const float sampleCountIni = 0.0;
            const bool bVariableSampleCount = true;
            const bool bMieRayPhase = true;
            const float tMaxMax = kDefaultMaxT;
            const float depthBufferValue = -1.0;
            result = integrateScatteredLuminance(
                pixPos, 
                worldPos, 
                worldDir, 
                sunDirection, 
                atmosphere, 
                bGround, 
                sampleCountIni, 
                depthBufferValue, 
                bMieRayPhase,
                tMaxMax,
                bVariableSampleCount
            ).scatteredLight;
        }
    }

    imageStore(imageCubeEnv, cubeCoord, vec4(prepareOut(result, atmosphere), 1.0));
    return;
}