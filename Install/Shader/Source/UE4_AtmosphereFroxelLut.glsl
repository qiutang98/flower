#version 460

/*
** Physical based render code, develop by engineer: qiutanguu.
*/

#extension GL_GOOGLE_include_directive : enable

#include "UE4_AtmosphereCommon.glsl"

// 32 x 32 x 32 Dimension. 

layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main()
{
    ivec3 lutSize = imageSize(imageFroxelScatter);
    ivec3 workPos = ivec3(gl_GlobalInvocationID.xyz);

    AtmosphereParameters atmosphere = getAtmosphereParameters();
    
    const vec2 pixPos = vec2(workPos.xy) + vec2(0.5f);
    const vec2 uv = pixPos / vec2(lutSize.xy);

    vec4 clipSpace = vec4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, 0.0, 1.0);
    vec4 viewPosH = viewData.camInvertProj * clipSpace;
    vec3 viewDir = viewPosH.xyz / viewPosH.w;

    vec3 worldDir = normalize((viewData.camInvertView * vec4(viewDir, 0.0)).xyz);
    
	vec3 camPos = convertToAtmosphereUnit(viewData.camWorldPos.xyz) + vec3(0, atmosphere.bottomRadius, 0);
	vec3 sunDir = -normalize(frameData.directionalLight.direction);
	vec3 sunLuminance = vec3(0.0);

    // [0, 1)
    float slice = ((float(workPos.z) + 0.5f) / float(lutSize.z));
	slice *= slice;	// Squared distribution
	slice *= float(lutSize.z);

    vec3 worldPos = camPos;
	float viewHeight;

    // Compute position from froxel information
	float tMax = aerialPerspectiveSliceToDepth(slice);
	vec3 newWorldPos = worldPos + tMax * worldDir;

	// If the voxel is under the ground, make sure to offset it out on the ground.
	viewHeight = length(newWorldPos);
	if (viewHeight <= (atmosphere.bottomRadius + kPlanetRadiusOffset))
	{
		// Apply a position offset to make sure no artefact are visible close to the earth boundaries for large voxel.
		newWorldPos = normalize(newWorldPos) * (atmosphere.bottomRadius + kPlanetRadiusOffset + 0.001f);
		worldDir = normalize(newWorldPos - camPos);
		tMax = length(newWorldPos - camPos);
	}
	float tMaxMax = tMax;

    // Move ray marching start up to top atmosphere.
	viewHeight = length(worldPos);
	if (viewHeight >= atmosphere.topRadius)
	{
		vec3 prevWorlPos = worldPos;
		if (!moveToTopAtmosphere(worldPos, worldDir, atmosphere.topRadius))
		{
			// Ray is not intersecting the atmosphere
            imageStore(imageFroxelScatter, workPos, vec4(0.0, 0.0, 0.0, 1.0));
			return;
		}

		float lengthToAtmosphere = length(prevWorlPos - worldPos);
		if (tMaxMax < lengthToAtmosphere)
		{
			// tMaxMax for this voxel is not within earth atmosphere
            imageStore(imageFroxelScatter, workPos, vec4(0.0, 0.0, 0.0, 1.0));
			return;
		}

		// Now world position has been moved to the atmosphere boundary: we need to reduce tMaxMax accordingly. 
		tMaxMax = max(0.0, tMaxMax - lengthToAtmosphere);
	}

    const bool bGround = false;
	const float sampleCountIni = max(1.0, float(workPos.z + 1.0) * 2.0f);
	const float depthBufferValue = -1.0;
	const bool bVariableSampleCount = false;
	const bool bMieRayPhase = true;

	SingleScatteringResult ss = integrateScatteredLuminance(
        pixPos, 
        worldPos, 
        worldDir, 
        sunDir, 
        atmosphere, 
        bGround, 
        sampleCountIni, 
        depthBufferValue, 
        bMieRayPhase, 
        tMaxMax,
        bVariableSampleCount
    );

	ss.scatteredLight = min(ss.scatteredLight, vec3(kMaxHalfFloat));

    imageStore(imageFroxelScatter, workPos, vec4(ss.scatteredLight, 1.0 - mean(ss.transmittance)));
}