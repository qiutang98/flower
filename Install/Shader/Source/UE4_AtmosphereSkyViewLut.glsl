#version 460

/*
** Physical based render code, develop by engineer: qiutanguu.
*/

#extension GL_GOOGLE_include_directive : enable

#include "UE4_AtmosphereCommon.glsl"

layout (local_size_x = 8, local_size_y = 8) in;
void main() 
{
    ivec2 lutSize = imageSize(imageSkyViewLut);
    ivec2 workPos = ivec2(gl_GlobalInvocationID.xy);

    if(workPos.x >= lutSize.x || workPos.y >= lutSize.y)
    {
        return;
    }

    const vec2 pixPos = vec2(workPos) + vec2(0.5f);
    const vec2 uv = pixPos / vec2(lutSize);

    AtmosphereParameters atmosphere = getAtmosphereParameters();

    // We are revert z.
    vec4 clipSpace = vec4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, 0.0, 1.0);
	vec4 viewPosH = viewData.camInvertProj * clipSpace;
    vec3 viewDir = viewPosH.xyz / viewPosH.w;

    vec3 worldDir = normalize((viewData.camInvertView * vec4(viewDir, 0.0)).xyz);
	vec3 worldPos = convertToAtmosphereUnit(viewData.camWorldPos.xyz) + vec3(0.0, atmosphere.bottomRadius, 0.0);

    float viewHeight = length(worldPos);
	float viewZenithCosAngle;
	float lightViewCosAngle;
	uvToSkyViewLutParams(atmosphere, viewZenithCosAngle, lightViewCosAngle, viewHeight, uv);

	vec3 sunDir;
	{
		vec3 upVector = worldPos / viewHeight;
		float sunZenithCosAngle = dot(upVector, -normalize(frameData.directionalLight.direction));
		sunDir = normalize(vec3(sqrt(1.0 - sunZenithCosAngle * sunZenithCosAngle), sunZenithCosAngle, 0.0));
	}

    // Use view height as world pos here.
    worldPos = vec3(0.0, viewHeight, 0.0);
	float viewZenithSinAngle = sqrt(1 - viewZenithCosAngle * viewZenithCosAngle);
	worldDir = vec3(
		viewZenithSinAngle * lightViewCosAngle,
        viewZenithCosAngle,
		viewZenithSinAngle * sqrt(1.0 - lightViewCosAngle * lightViewCosAngle)
    );

    // Move to top atmospehre
	if (!moveToTopAtmosphere(worldPos, worldDir, atmosphere.topRadius)) 
	{
		// Ray is not intersecting the atmosphere
        imageStore(imageSkyViewLut, workPos, vec4(0.0, 0.0, 0.0, 1.0f));
        return;
	}

    const bool bGround = false;
	const float sampleCountIni = 30;
	const float depthBufferValue = -1.0;
    const bool bMieRayPhase = true;
    const float tMaxMax = kDefaultMaxT;
	const bool bVariableSampleCount = true;

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
    imageStore(imageSkyViewLut, workPos, vec4(ss.scatteredLight, 1.0f));
}