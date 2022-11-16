#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_basic : require

#define NO_MULTISCATAPPROX_ENABLED
#include "UE4_AtmosphereCommon.glsl"


const uint kSqrtSampleCount = 8;
const uint kSampleCount = kSqrtSampleCount * kSqrtSampleCount;

shared vec3 sharedMultiScatAs1[kSampleCount];
shared vec3 sharedScatterLight[kSampleCount];

layout (local_size_x = 1, local_size_y = 1, local_size_z = kSampleCount) in;
void main() 
{
    ivec2 lutSize = imageSize(imageMultiScatterLut);
    ivec2 workPos = ivec2(gl_GlobalInvocationID.xy);

    const uint flattenId = gl_GlobalInvocationID.z;

    AtmosphereParameters atmosphere = getAtmosphereParameters();

    const vec2 pixPos = vec2(workPos) + vec2(0.5f);

    vec2 uv = pixPos / vec2(lutSize);
    uv = vec2(fromSubUvsToUnit(uv.x, lutSize.x), fromSubUvsToUnit(uv.y, lutSize.y));

    float cosSunZenithAngle = uv.x * 2.0 - 1.0;
	vec3 sunDir = vec3(0.0, cosSunZenithAngle, sqrt(saturate(1.0 - cosSunZenithAngle * cosSunZenithAngle)));

	// We adjust again viewHeight according to kPlanetRadiusOffset to be in a valid range.
	float viewHeight = atmosphere.bottomRadius + saturate(uv.y + kPlanetRadiusOffset) * (atmosphere.topRadius - atmosphere.bottomRadius - kPlanetRadiusOffset);

    // Config world pos and world direction.
	vec3 worldPos = vec3(0.0f, viewHeight, 0.0f);
	vec3 worldDir = vec3(0.0f, 1.0f, 0.0f);

	const bool bGround = true;
	const float sampleCountIni = 20; // A minimum set of step is required for accuracy unfortunately.
	const float depthBufferValue = -1.0;
	const bool bMieRayPhase = false;
    const float tMaxMax = kDefaultMaxT;
    const bool bVariableSampleCount = false;
	const float sphereSolidAngle = 4.0 * kPI;
	const float isotropicPhase = 1.0 / sphereSolidAngle;
    
    const float sqrtSample = float(kSqrtSampleCount);

	float i = 0.5 + float(flattenId / kSqrtSampleCount);
	float j = 0.5 + float(flattenId - float((flattenId / kSqrtSampleCount) * kSqrtSampleCount));

    vec3 multiScatAs1 = vec3(0.0);
    vec3 scatteredLight = vec3(0.0);
	{
		float randA = i / sqrtSample;
		float randB = j / sqrtSample;
		float theta = 2.0f * kPI * randA;

        // Uniform distribution https://mathworld.wolfram.com/SpherePointPicking.html
		float phi = acos(1.0f - 2.0f * randB);	

		float cosPhi = cos(phi);
		float sinPhi = sin(phi);
		float cosTheta = cos(theta);
		float sinTheta = sin(theta);

        // Get direction, y up.
		worldDir.x = cosTheta * sinPhi;
		worldDir.y = cosPhi;
		worldDir.z = sinTheta * sinPhi;

		SingleScatteringResult result = integrateScatteredLuminance(
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

        sharedMultiScatAs1[flattenId] = result.multiScatAs1 * sphereSolidAngle / (sqrtSample * sqrtSample);
        sharedScatterLight[flattenId] = result.scatteredLight * sphereSolidAngle / (sqrtSample * sqrtSample);
	}

    groupMemoryBarrier();
    barrier();

    // Reduce.
    uint loopIndex = kSampleCount / 2;
    while(loopIndex > 0)
    {
        if(flattenId < loopIndex)
        {
            sharedMultiScatAs1[flattenId] += sharedMultiScatAs1[flattenId + loopIndex];
            sharedScatterLight[flattenId] += sharedScatterLight[flattenId + loopIndex];
        }

        loopIndex /= 2;

        groupMemoryBarrier();
        barrier();
    }

    if (flattenId > 0)
    {
        return;
    }
		
    // MultiScatAs1 represents the amount of luminance scattered as if the integral of scattered luminance over the sphere would be 1.
    //  - 1st order of scattering: one can ray-march a straight path as usual over the sphere. That is InScatteredLuminance.
    //  - 2nd order of scattering: the inscattered luminance is InScatteredLuminance at each of samples of fist order integration. Assuming a uniform phase function that is represented by MultiScatAs1,
    //  - 3nd order of scattering: the inscattered luminance is (InScatteredLuminance * MultiScatAs1 * MultiScatAs1)
    //  - etc.
    // For a serie, sum_{n=0}^{n=+inf} = 1 + r + r^2 + r^3 + ... + r^n = 1 / (1.0 - r), see https://en.wikipedia.org/wiki/Geometric_series 

    const vec3 r = sharedMultiScatAs1[flattenId] * isotropicPhase;	// Equation 7 f_ms
    vec3 inScatteredLuminance = sharedScatterLight[flattenId] * isotropicPhase; // Equation 5 L_2ndOrder
    const vec3 sumOfAllMultiScatteringEventsContribution = 1.0f / (1.0 - r);
    vec3 L = inScatteredLuminance * sumOfAllMultiScatteringEventsContribution; // Equation 10 Psi_ms


    vec3 result = L * atmosphere.multipleScatteringFactor;
    result = min(result, vec3(kMaxHalfFloat));
    imageStore(imageMultiScatterLut, workPos, vec4(result, 1.0f));
}