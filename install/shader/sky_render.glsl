#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable

#define SHARED_SAMPLER_SET 1
#include "common_shader.glsl"

// Atmosphere transmittance lut compute, base on 
// https://github.com/sebh/UnrealEngineSkyAtmosphere
// https://ebruneton.github.io/precomputed_atmospheric_scattering/

layout (set = 0, binding = 0) uniform UniformFrameData { PerFrameData frameData; };

layout (set = 0, binding = 1, rgba16f) uniform image2D imageTransmittanceLut;
layout (set = 0, binding = 2) uniform texture2D inTransmittanceLut;

layout (set = 0, binding = 3, rgba16f) uniform image2D imageSkyViewLut;
layout (set = 0, binding = 4) uniform texture2D inSkyViewLut;

layout (set = 0, binding = 5, rgba16f) uniform image2D imageMultiScatterLut;
layout (set = 0, binding = 6) uniform texture2D inMultiScatterLut;

layout (set = 0, binding = 7) uniform texture2D inDepth;

layout (set = 0, binding = 8, rgba16f) uniform image3D imageFroxelScatter;
layout (set = 0, binding = 9) uniform texture3D inFroxelScatter;

layout (set = 0, binding = 10, rgba16f) uniform image2D imageHdrSceneColor;
layout (set = 0, binding = 11, rgba16f) writeonly uniform imageCube imageCubeEnv;

layout (set = 0, binding = 12, r16f) uniform image2D imageGbufferId;
layout (set = 0, binding = 13, rgba16f) uniform image2D imageCloudDistantLit; // 1x1


layout (set = 0, binding = 14, rgba16f) uniform image3D imageDistantGrid;


AtmosphereParameters getAtmosphereParameters()
{
	return getAtmosphereParameters(frameData);
}

vec3 convertToAtmosphereUnit(vec3 o)
{
	return convertToAtmosphereUnit(o, frameData);
}  

vec3 convertToCameraUnit(vec3 o)
{
	return convertToCameraUnit(o, frameData);
}  

float getShadow(in const AtmosphereParameters atmospehre, vec3 p)
{
	// Use shadow can get godray like effect here. 
	// But we can't use because air perspective's resolution too low.
	return 1.0f;
}

// Generate a sample (using importance sampling) along an infinitely long path with a given constant extinction.
// Zeta is a random number in [0,1]
float infiniteTransmittanceIS(float extinction, float zeta)
{
	return -log(1.0f - zeta) / extinction;
}

// Normalized PDF from a sample on an infinitely long path according to transmittance and extinction.
float infiniteTransmittancePDF(float extinction, float transmittance)
{
	return extinction * transmittance;
}

// Same as above but a sample is generated constrained within a range t,
// where transmittance = exp(-extinction*t) over that range.
float rangedTransmittanceIS(float extinction, float transmittance, float zeta)
{
	return -log(1.0f - zeta * (1.0f - transmittance)) / extinction;
}

// https://www.youtube.com/watch?v=y-oBGzDCZKI at 9:20
void uvToSkyViewLutParams(
	in  const AtmosphereParameters atmosphere, 
	out float viewZenithCosAngle, 
	out float lightViewCosAngle, 
	in  float viewHeight, 
	in  vec2  uv)
{
	// Constrain uvs to valid sub texel range (avoid zenith derivative issue making LUT usage visible)
	vec2 lutSize = vec2(imageSize(imageSkyViewLut));
	uv = vec2(fromSubUvsToUnit(uv.x, lutSize.x), fromSubUvsToUnit(uv.y, lutSize.y));
	
	float vHorizon = sqrt(viewHeight * viewHeight - atmosphere.bottomRadius * atmosphere.bottomRadius);

	// Ground to horizon cos
	float cosBeta = vHorizon / viewHeight;

	float beta = acos(cosBeta);
	float zenithHorizonAngle = kPI - beta;

	if (uv.y < 0.5f)
	{
		float coord = 2.0 * uv.y;
		coord = 1.0 - coord;
		coord *= coord; // Non-linear sunLightInfo view lut.

		coord = 1.0 - coord;
		viewZenithCosAngle = cos(zenithHorizonAngle * coord);
	}
	else
	{
		float coord = uv.y * 2.0 - 1.0;
		coord *= coord; // Non-linear sunLightInfo view lut.
		viewZenithCosAngle = cos(zenithHorizonAngle + beta * coord);
	}

	float coord = uv.x;
	coord *= coord;

	lightViewCosAngle = -(coord * 2.0 - 1.0);
}

vec3 getMultipleScattering(
	in const AtmosphereParameters atmosphere, 
	vec3  scattering, 
	vec3  extinction, 
	vec3  worldPos, 
	float viewZenithCosAngle)
{
	const float viewHeight = getViewHeight(worldPos, atmosphere);
	vec2 lutSize = vec2(textureSize(inMultiScatterLut, 0));

	vec2 uv = saturate(vec2(viewZenithCosAngle * 0.5f + 0.5f, viewHeight / (atmosphere.topRadius - atmosphere.bottomRadius)));
	uv = vec2(fromUnitToSubUvs(uv.x, lutSize.x), fromUnitToSubUvs(uv.y, lutSize.y));

	vec3 multiScatteredLuminance = texture(sampler2D(inMultiScatterLut, linearClampEdgeSampler), uv).rgb;
	return multiScatteredLuminance;
}

struct SingleScatteringResult
{
    vec3 scatteredLight; // Scattered light (luminance)
    vec3 opticalDepth;   // Optical depth (1/m)
    vec3 transmittance;  // Transmittance in [0,1] (unitless)

    vec3 multiScatAs1;
	vec3 newMultiScatStep0Out;
	vec3 newMultiScatStep1Out;
};

SingleScatteringResult buildSingleScatteringResultDefault()
{
	SingleScatteringResult result;

	result.scatteredLight = vec3(0.0);
	result.opticalDepth   = vec3(0.0);
	result.transmittance  = vec3(0.0);

	result.multiScatAs1         = vec3(0.0);
	result.newMultiScatStep0Out = vec3(0.0);
	result.newMultiScatStep0Out = vec3(0.0);

	return result;
}

const float kDefaultMaxT = 9000000.0f;

SingleScatteringResult integrateScatteredLuminance(
	in vec2  pixPos, 
	in vec3  worldPos, 
	in vec3  worldDir, 
	in vec3  sunDir, 
	in const AtmosphereParameters atmosphere,
	in bool  bGround, 
	in float sampleCountIni, 
	in float depthBufferValue, 
	in bool  bMieRayPhase, 
	in float tMaxMax,
	in bool  bVariableSampleCount)
{
	SingleScatteringResult result = buildSingleScatteringResultDefault();

	const vec3 kEarthOrigin = vec3(0.0);

	// Compute next intersection with atmosphere or ground 
	float tBottom = raySphereIntersectNearest(worldPos, worldDir, kEarthOrigin, atmosphere.bottomRadius);
	float tTop = raySphereIntersectNearest(worldPos, worldDir, kEarthOrigin, atmosphere.topRadius);

	// Evaluate valid intersect t.
	float tMax = 0.0f;
	if (tBottom < 0.0f)  // No intersect with bottom.
	{
		if (tTop < 0.0f) // No intersect with atmosphere.
		{
			// No intersection with earth nor atmosphere: stop right away
			return result;
		}
		else // Intersect with atmosphere.
		{
			tMax = tTop;
		}
	}
	else // Intersect with bottom.
	{
		if (tTop > 0.0f) // Also intersect with atmosphere.
		{
			// Use nearest one.
			tMax = min(tTop, tBottom);
		}
	}

	// Correct t select with background depth and shading model id. 
	// Theses code used when composite sunLightInfo on the scene color texture.
	if (depthBufferValue >= 0.0 && depthBufferValue <= 1.0)
	{
		// World space depth.
		vec2 sampleUv = (pixPos + vec2(0.5)) / vec2(textureSize(inDepth, 0));
		vec3 depthBufferWorldPos = getWorldPos(sampleUv, depthBufferValue, frameData);

		// Apply earth offset to go back to origin as top of earth mode. 
		float tDepth = length(depthBufferWorldPos * 0.001 + vec3(0.0, atmosphere.bottomRadius, 0.0) - worldPos); // Meter -> kilometers
		if (tDepth < tMax)
		{
			tMax = tDepth;
		}
	}
	tMax = min(tMax, tMaxMax);

	// Sample count 
	float sampleCount = sampleCountIni;
	float sampleCountFloor = sampleCountIni;
	float tMaxFloor = tMax;
	if (bVariableSampleCount)
	{
		sampleCount = mix(float(atmosphere.viewRayMarchMinSPP), float(atmosphere.viewRayMarchMaxSPP), saturate(tMax * 0.01));
		sampleCountFloor = floor(sampleCount);
		tMaxFloor = tMax * sampleCountFloor / sampleCount;	// rescale tMax to map to the last entire step segment.
	}

	float dt = tMax / sampleCount;

	// Phase functions
	const float uniformPhase = getUniformPhase();
	const vec3 wi = sunDir;
	const vec3 wo = worldDir;
	float cosTheta = dot(wi, wo);

	// mnegate cosTheta because due to WorldDir being a "in" direction. 
	float miePhaseValue = hgPhase(atmosphere.miePhaseG, -cosTheta);
	float rayleighPhaseValue = rayleighPhase(cosTheta);

	vec3 globalL = frameData.sunLightInfo.color * frameData.sunLightInfo.intensity;

	vec3 L = vec3(0.0);
	vec3 throughput = vec3(1.0);
	vec3 opticalDepth = vec3(0.0);
	float t = 0.0f;
	float tPrev = 0.0;
	const float sampleSegmentT = 0.3f;

	for (float s = 0.0; s < sampleCount; s += 1.0)
	{
		if (bVariableSampleCount)
		{
			// More expenssive but artefact free
			float t0 = (s) / sampleCountFloor;
			float t1 = (s + 1.0f) / sampleCountFloor;

			// Non linear distribution of sample within the range.
			t0 = t0 * t0;
			t1 = t1 * t1;

			// Make t0 and t1 world space distances.
			t0 = tMaxFloor * t0;

			if (t1 > 1.0)
			{
				// this reveal depth slices
				//	t1 = tMaxFloor;	
				t1 = tMax;
			}
			else
			{
				t1 = tMaxFloor * t1;
			}
			// With dithering required to hide some sampling artefact relying on TAA later? This may even allow volumetric shadow?
			// t = t0 + (t1 - t0) * (whangHashNoise(pixPos.x, pixPos.y, gFrameId * 1920 * 1080)); 
			t = t0 + (t1 - t0) * sampleSegmentT;
			dt = t1 - t0;
		}
		else
		{
			float newT = tMax * (s + sampleSegmentT) / sampleCount;
			dt = newT - t;
			t = newT;
		}
		

		// Get current step position.
		vec3 P = worldPos + t * worldDir;

		// Sample medium color.
		MediumSampleRGB medium = sampleMediumRGB(P, atmosphere);

		// Get optical depth
		const vec3 sampleOpticalDepth = medium.extinction * dt;

		// transmittance is exp(-opticalDepth).
		const vec3 sampleTransmittance = exp(-sampleOpticalDepth); 

		// Accumulate sample optical depth.
		opticalDepth += sampleOpticalDepth;

		// Get sample height.
		float pHeight = length(P);

		// Get normalize direction.
		const vec3 upVector = P / pHeight;

		// Get pre-compute transmittance.
		float sunZenithCosAngle = dot(sunDir, upVector);
		vec2 uv;
		lutTransmittanceParamsToUv(atmosphere, pHeight, sunZenithCosAngle, uv);
		vec3 transmittanceToSun = texture(sampler2D(inTransmittanceLut, linearClampEdgeSampler), uv).rgb;

		vec3 phaseTimesScattering;
		if (bMieRayPhase)
		{
			phaseTimesScattering = medium.scatteringMie * miePhaseValue + medium.scatteringRay * rayleighPhaseValue;
		}
		else
		{
			phaseTimesScattering = medium.scattering * uniformPhase;
		}

		// Earth shadow 
		float tEarth = raySphereIntersectNearest(P, sunDir, kEarthOrigin + kAtmospherePlanetRadiusOffset * upVector, atmosphere.bottomRadius);
		float earthShadow = tEarth >= 0.0f ? 0.0f : 1.0f;

		// Dual scattering for multi scattering 
		vec3 multiScatteredLuminance = vec3(0.0); 

#ifndef NO_MULTISCATAPPROX_ENABLED
		multiScatteredLuminance = getMultipleScattering(atmosphere, medium.scattering, medium.extinction, P, sunZenithCosAngle);
#endif
		float shadow = getShadow(atmosphere, P);

		vec3 S = globalL * (earthShadow * shadow * transmittanceToSun * phaseTimesScattering + (multiScatteredLuminance * medium.scattering));

		// When using the power serie to accumulate all sattering order, serie r must be <1 for a serie to converge.
		// Under extreme coefficient, multiScatAs1 can grow larger and thus result in broken visuals.
		// The way to fix that is to use a proper analytical integration as proposed in slide 28 of http://www.frostbite.com/2015/08/physically-based-unified-volumetric-rendering-in-frostbite/
		// However, it is possible to disable as it can also work using simple power serie sum unroll up to 5th order. The rest of the orders has a really low contribution.
		vec3 ms = medium.scattering * 1;
		vec3 msint = (ms - ms * sampleTransmittance) / medium.extinction;
		result.multiScatAs1 += throughput * msint;

		// Evaluate input to multi scattering
		{
			vec3 newMS;

			newMS = earthShadow * transmittanceToSun * medium.scattering * uniformPhase * 1;
			result.newMultiScatStep0Out += throughput * (newMS - newMS * sampleTransmittance) / medium.extinction;

			newMS = medium.scattering * uniformPhase * multiScatteredLuminance;
			result.newMultiScatStep1Out += throughput * (newMS - newMS * sampleTransmittance) / medium.extinction;
		}

		// See slide 28 at http://www.frostbite.com/2015/08/physically-based-unified-volumetric-rendering-in-frostbite/ 
		vec3 sint = (S - S * sampleTransmittance) / medium.extinction; // integrate along the current step segment 
		L += throughput * sint;	// accumulate and also take into account the transmittance from previous steps
		throughput *= sampleTransmittance;

		tPrev = t;
	}

	if (bGround && (tMax == tBottom) && (tBottom > 0.0))
	{
		// Account for bounced light off the earth
		vec3 P = worldPos + tBottom * worldDir;
		float pHeight = length(P);

		const vec3 upVector = P / pHeight;
		float sunZenithCosAngle = dot(sunDir, upVector);
		vec2 uv;
		lutTransmittanceParamsToUv(atmosphere, pHeight, sunZenithCosAngle, uv);
		vec3 transmittanceToSun = texture(sampler2D(inTransmittanceLut, linearClampEdgeSampler), uv).rgb;

		const float NdotL = saturate(dot(normalize(upVector), normalize(sunDir)));
		L += globalL * transmittanceToSun * throughput * NdotL * atmosphere.groundAlbedo / kPI;
	}

	result.scatteredLight = L;
	result.opticalDepth  = opticalDepth;
	result.transmittance = throughput;
	return result;
}

// Input world position and compute it's scatter lighting.
vec3 getPosScatterLight(
    in const AtmosphereParameters atmosphere,
    in const vec3 inWorldPos,
    in const vec2 uv,
    in const bool bGround,
	in const vec2 pixPos)
{
    float viewHeight = length(inWorldPos);
	float viewZenithCosAngle;
	float lightViewCosAngle;
	uvToSkyViewLutParams(atmosphere, viewZenithCosAngle, lightViewCosAngle, viewHeight, uv);

	vec3 sunDir;
	{
		vec3 upVector = inWorldPos / viewHeight;
		float sunZenithCosAngle = dot(upVector, -normalize(frameData.sunLightInfo.direction));
		sunDir = normalize(vec3(sqrt(1.0 - sunZenithCosAngle * sunZenithCosAngle), sunZenithCosAngle, 0.0));
	}

    // Use view height as world pos here.
    vec3 worldPos = vec3(0.0, viewHeight, 0.0);
	float viewZenithSinAngle = sqrt(1 - viewZenithCosAngle * viewZenithCosAngle);
	vec3 worldDir = vec3(
		viewZenithSinAngle * lightViewCosAngle,
        viewZenithCosAngle,
		viewZenithSinAngle * sqrt(1.0 - lightViewCosAngle * lightViewCosAngle)
    );

    // Move to top atmospehre
	if (!moveToTopAtmosphere(worldPos, worldDir, atmosphere.topRadius)) 
	{
		// Ray is not intersecting the atmosphere
        return vec3(0.0, 0.0, 0.0);
	}

	const float sampleCountIni      = 30;
	const float depthBufferValue    = -1.0;
    const bool bMieRayPhase         = true;
    const float tMaxMax             = kDefaultMaxT;
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

    return ss.scatteredLight;
}

#ifdef TRANSMITTANCE_LUT_PASS

#ifndef NO_MULTISCATAPPROX_ENABLED
#error "Transmittance lut pass must disable multi scatter approx."
#endif

layout (local_size_x = 8, local_size_y = 8) in;
void main() 
{
    ivec2 lutSize = imageSize(imageTransmittanceLut);
    ivec2 workPos = ivec2(gl_GlobalInvocationID.xy);

    AtmosphereParameters atmosphere = getAtmosphereParameters();

    const vec2 pixPos = vec2(workPos) + vec2(0.5f);
    const vec2 uv = pixPos / vec2(lutSize);

    float viewHeight;
	float viewZenithCosAngle;
	uvToLutTransmittanceParams(atmosphere, viewHeight, viewZenithCosAngle, uv);
    
    const vec3 worldPos = vec3(0.0f, viewHeight, 0.0f);
    const vec3 worldDir = vec3(0.0f, viewZenithCosAngle, sqrt(1.0 - viewZenithCosAngle * viewZenithCosAngle));
    const vec3 sunDir = -normalize(frameData.sunLightInfo.direction);
    const bool bGround = false;
    const float sampleCountIni = 40.0;
    const float depthBufferValue = -1.0;
    const bool bMieRayPhase = false;
    const float tMaxMax = kDefaultMaxT;
    const bool bVariableSampleCount = false;
    vec3 opticalDepth = integrateScatteredLuminance(
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
    ).opticalDepth;

	vec3 transmittance = exp(-opticalDepth);
    transmittance = min(transmittance, vec3(kMaxHalfFloat));
	imageStore(imageTransmittanceLut, workPos, vec4(transmittance, 1.0f));
}

#endif // TRANSMITTANCE_LUT_PASS

#ifdef MULTI_SCATTER_PASS

#extension GL_GOOGLE_include_directive : enable
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_basic : require

#ifndef NO_MULTISCATAPPROX_ENABLED
#error "Multi scatter lut pass must disable multi scatter approx."
#endif

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

	// We adjust again viewHeight according to kAtmospherePlanetRadiusOffset to be in a valid range.
	float viewHeight = atmosphere.bottomRadius + saturate(uv.y + kAtmospherePlanetRadiusOffset) * (atmosphere.topRadius - atmosphere.bottomRadius - kAtmospherePlanetRadiusOffset);

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

#endif // MULTI_SCATTER_PASS

#ifdef SKY_LUT_PASS

#ifdef NO_MULTISCATAPPROX_ENABLED
#error "Skylut pass need multi scatter approx."
#endif

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
	vec3 worldPos = convertToAtmosphereUnit(frameData.camWorldPos.xyz) + vec3(0.0, atmosphere.bottomRadius, 0.0);
	vec3 scatteredLight = getPosScatterLight(atmosphere, worldPos, uv, true, pixPos);
	imageStore(imageSkyViewLut, workPos, vec4(scatteredLight, 1.0f));
}

#endif // SKY_LUT_PASS

#ifdef AIR_PERSPECTIVE_PASS

#ifdef NO_MULTISCATAPPROX_ENABLED
#error "Air perspective pass need multi scatter approx."
#endif

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
    vec4 viewPosH = frameData.camInvertProj * clipSpace;
    vec3 viewDir = viewPosH.xyz / viewPosH.w;

    vec3 worldDir = normalize((frameData.camInvertView * vec4(viewDir, 0.0)).xyz);
    
	vec3 camPos = convertToAtmosphereUnit(frameData.camWorldPos.xyz) + vec3(0, atmosphere.bottomRadius, 0);
	vec3 sunDir = -normalize(frameData.sunLightInfo.direction);
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
	if (viewHeight <= (atmosphere.bottomRadius + kAtmospherePlanetRadiusOffset))
	{
		// Apply a position offset to make sure no artefact are visible close to the earth boundaries for large voxel.
		newWorldPos = normalize(newWorldPos) * (atmosphere.bottomRadius + kAtmospherePlanetRadiusOffset + 0.001f);
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

#endif // AIR_PERSPECTIVE_PASS

#ifdef COMPOSITE_SKY_PASS

#ifdef NO_MULTISCATAPPROX_ENABLED
#error "Sky composite pass need multi scatter approx."
#endif

layout (local_size_x = 8, local_size_y = 8) in;
void main() 
{
    ivec2 colorSize = imageSize(imageHdrSceneColor);

    uvec2 groupThreadId = remap8x8(gl_LocalInvocationIndex);
    uvec2 dispatchId = groupThreadId + gl_WorkGroupID.xy * 8;

    ivec2 workPos = ivec2(dispatchId);

    if(workPos.x >= colorSize.x || workPos.y >= colorSize.y)
    {
        return;
    }

    const vec2 pixPos = vec2(workPos) + vec2(0.5f);
    const vec2 uv = pixPos / vec2(colorSize);

    vec4 srcColor = imageLoad(imageHdrSceneColor, workPos);

    AtmosphereParameters atmosphere = getAtmosphereParameters();

    // We are revert z.
    vec4 clipSpace = vec4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, 0.0, 1.0);
	vec4 viewPosH = frameData.camInvertProj * clipSpace;
    vec3 viewDir = viewPosH.xyz / viewPosH.w;
    vec3 worldDir = normalize((frameData.camInvertView * vec4(viewDir, 0.0)).xyz);
	vec3 worldPos = convertToAtmosphereUnit(frameData.camWorldPos.xyz) + vec3(0.0, atmosphere.bottomRadius, 0.0);
    float viewHeight = length(worldPos);

	vec3 L = vec3(0);
	float depthBufferValue = texture(sampler2D(inDepth, pointClampEdgeSampler), uv).r;

    const bool bSky = (depthBufferValue == 0.0f);

    const vec3 sunDirection = -normalize(frameData.sunLightInfo.direction);

    const bool bUnderAtmosphere =  viewHeight < atmosphere.topRadius;

    vec3 upVector = normalize(worldPos);

    // Back ground and under atmosphere pixel, sample sunLightInfo view lut.
    if (bUnderAtmosphere && bSky)
	{
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
		skyViewLutParamsToUv(atmosphere, bIntersectGround, viewZenithCosAngle, lightViewCosAngle, viewHeight, vec2(textureSize(inSkyViewLut, 0)), sampleUv);

		vec3 luminance = texture(sampler2D(inSkyViewLut, linearClampEdgeSampler), sampleUv).rgb;

		imageStore(imageHdrSceneColor, workPos, vec4(luminance, srcColor.a));
		imageStore(imageGbufferId, workPos, vec4(packObjectId(kSkyObjectId)));

        return;
	}

    float opacity = 0.0;
    if(bUnderAtmosphere) // Composite air perspective.
    {
        // Exist pre-compute data, sample it.

        // Build world position.
        clipSpace.z = depthBufferValue;
        vec4 depthBufferWorldPos = frameData.camInvertViewProj * clipSpace;
        depthBufferWorldPos.xyz /= depthBufferWorldPos.w;

        float tDepth = length((depthBufferWorldPos.xyz * 0.001) - (worldPos + vec3(0.0, -atmosphere.bottomRadius, 0.0))); // meter -> kilometers.
        float slice = aerialPerspectiveDepthToSlice(tDepth);

        float weight = 1.0;
        if (slice < 0.5)
        {
            // We multiply by weight to fade to 0 at depth 0. That works for luminance and opacity.
            weight = saturate(slice * 2.0);
            slice = 0.5;
        }
        ivec3 sliceLutSize = textureSize(inFroxelScatter, 0);
        float w = sqrt(slice / float(sliceLutSize.z));	// squared distribution

        const vec4 airPerspective = weight * texture(sampler3D(inFroxelScatter, linearClampEdgeSampler), vec3(uv, w));
        L.rgb += airPerspective.rgb;
        opacity = airPerspective.a;
    }
    else if(bSky)
    {
        // No precompute data can use. compute new data.

        // Move to top atmosphere as the starting point for ray marching.
        // This is critical to be after the above to not disrupt above atmosphere tests and voxel selection.
        if (!moveToTopAtmosphere(worldPos, worldDir, atmosphere.topRadius))
        {
            // Ray is not intersecting the atmosphere, return.	
            return;
        }

        const bool bGround = false;
        const float sampleCountIni = 0.0;
        const bool bVariableSampleCount = true;
        const bool bMieRayPhase = true;
        const float tMaxMax = kDefaultMaxT;
        depthBufferValue = -1.0;
        SingleScatteringResult ss = integrateScatteredLuminance(
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
        );

        L += ss.scatteredLight;
        vec3 throughput = ss.transmittance;

        const float transmittance = mean(throughput);
        opacity = 1.0 - transmittance;
    }

    vec3 outColor = L.rgb + (1.0 - opacity) * srcColor.xyz;


    imageStore(imageHdrSceneColor, workPos, vec4(outColor, srcColor.w));
}

#endif // COMPOSITE_SKY_PASS

#ifdef SKY_CAPTURE_PASS

#ifdef NO_MULTISCATAPPROX_ENABLED
#error "Sky capture pass need multi scatter approx."
#endif

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
    const vec3 sunDirection = -normalize(frameData.sunLightInfo.direction);

    // Sample skyview lut and store in cubemap capture.
    vec3 worldPos = convertToAtmosphereUnit(frameData.camWorldPos.xyz) + vec3(0.0, atmosphere.bottomRadius, 0.0);
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
		skyViewLutParamsToUv(atmosphere, bIntersectGround, viewZenithCosAngle, lightViewCosAngle, viewHeight, vec2(textureSize(inSkyViewLut, 0)), sampleUv);

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

    imageStore(imageCubeEnv, cubeCoord, vec4(result, 1.0));
    return;
}

#endif

#ifdef SKY_DISTANCE_LIT_CLOUD_PASS

#ifdef NO_MULTISCATAPPROX_ENABLED
#error "Distant lit need multi scatter approx."
#endif

// NOTE: Non visibility term in this distant lut.
shared vec3 shareSkyLit[64];

// Average distant lit sample for cloud SH approx.
layout(local_size_x = 64) in;
void main()
{
	AtmosphereParameters atmosphere = getAtmosphereParameters();

	// Total 8192 km.
	const float kSkyDistantOffset = kDistantSkyLitMax * (gl_WorkGroupID.z + 0.5) / float(imageSize(imageCloudDistantLit).x) * 0.001 + atmosphere.bottomRadius;

    // Get sample pos in km.
	uint localId   = gl_LocalInvocationIndex;
    uvec2 dispatchId = remap8x8(gl_LocalInvocationIndex);

    ivec2 workPos = ivec2(dispatchId);
	vec3 worldDir;
	{
		float randA = (workPos.x + 0.5) / 8.0;
		float randB = (workPos.y + 0.5) / 8.0;
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
	}

    const vec3 sunDirection = -normalize(frameData.sunLightInfo.direction);
	
	// Ray intersecting the atmosphere.	
	const bool bGround              = false;
	const float sampleCountIni      = 10.0;
	const bool bVariableSampleCount = false;
	const bool bMieRayPhase         = false;
	const float tMaxMax             = kDefaultMaxT;
	const float depthBufferValue    = -1.0;

	shareSkyLit[localId] = integrateScatteredLuminance(
		vec2(0.0), 
		vec3(0.0, kSkyDistantOffset, 0.0), 
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

    groupMemoryBarrier();
    barrier();

    uint loopIndex = 32;
    while(loopIndex > 0)
    {
        if(localId < loopIndex)
        {
            shareSkyLit[localId] += shareSkyLit[localId + loopIndex];
        }

        loopIndex /= 2;

        groupMemoryBarrier();
        barrier();
    }

    if (localId < 1)
    {
		vec3 avgColor = shareSkyLit[localId] / 64.0;
		imageStore(imageCloudDistantLit, ivec2(gl_WorkGroupID.z, 0), vec4(avgColor, 1.0));
    }
}

#endif

#ifdef SKY_DISTANCE_GRID_LIT_PASS

#ifdef NO_MULTISCATAPPROX_ENABLED
#error "Distant grid need multi scatter approx."
#endif

// We sample cloud shadow as an approximate of visibility.

shared vec3 shareSkyLit[64];
shared vec3 shareTransmittance[64];

// Average distant lit sample for cloud SH approx.
layout(local_size_x = 64) in;
void main()
{
    ivec3 lutSize = imageSize(imageDistantGrid);
    ivec3 workPos = ivec3(gl_WorkGroupID.xyz);

	AtmosphereParameters atmosphere = getAtmosphereParameters();

	const vec2 pixPos = vec2(workPos.xy) + vec2(0.5f);
    const vec2 uv = pixPos / vec2(lutSize.xy);

    vec4 clipSpace = vec4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, 0.0, 1.0);
    vec4 viewPosH = frameData.camInvertProj * clipSpace;
    vec3 viewDir = viewPosH.xyz / viewPosH.w;

    vec3 worldDir = normalize((frameData.camInvertView * vec4(viewDir, 0.0)).xyz);

	vec3 camPos = convertToAtmosphereUnit(frameData.camWorldPos.xyz) + vec3(0, atmosphere.bottomRadius, 0);
	vec3 sunDir = -normalize(frameData.sunLightInfo.direction);

	float slice = ((float(workPos.z) + 0.5f) / float(lutSize.z));
	slice *= slice;	// Squared distribution
	slice *= float(lutSize.z);

	vec3 worldPos = camPos;
	float viewHeight;

    // Compute position from froxel information
	float tMax = distantGridSliceToDepth(slice);
	vec3 newWorldPos = worldPos + tMax * worldDir;

	// If the voxel is under the ground, make sure to offset it out on the ground.
	viewHeight = length(newWorldPos);
	if (viewHeight <= (atmosphere.bottomRadius + kAtmospherePlanetRadiusOffset))
	{
		// Apply a position offset to make sure no artefact are visible close to the earth boundaries for large voxel.
		newWorldPos = normalize(newWorldPos) * (atmosphere.bottomRadius + kAtmospherePlanetRadiusOffset + 0.001f);
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
            imageStore(imageDistantGrid, workPos, vec4(0.0, 0.0, 0.0, 1.0));
			return;
		}

		float lengthToAtmosphere = length(prevWorlPos - worldPos);
		if (tMaxMax < lengthToAtmosphere)
		{
			// tMaxMax for this voxel is not within earth atmosphere
            imageStore(imageDistantGrid, workPos, vec4(0.0, 0.0, 0.0, 1.0));
			return;
		}

		// Now world position has been moved to the atmosphere boundary: we need to reduce tMaxMax accordingly. 
		tMaxMax = max(0.0, tMaxMax - lengthToAtmosphere);
	}

	const bool bGround = false;
	const float sampleCountIni = max(1.0, float(workPos.z + 1.0) * 2.0f);
	const float depthBufferValue = -1.0;
	const bool bVariableSampleCount = false;
	const bool bMieRayPhase = false;


	uint localId   = gl_LocalInvocationIndex;
	{
		uvec2 dispatchId = remap8x8(gl_LocalInvocationIndex);
		float randA = (dispatchId.x + 0.5) / 8.0;
		float randB = (dispatchId.y + 0.5) / 8.0;
		float theta = 2.0f * kPI * randA;

		float phi = acos(1.0f - 2.0f * randB);	

		float cosPhi = cos(phi);
		float sinPhi = sin(phi);
		float cosTheta = cos(theta);
		float sinTheta = sin(theta);

		worldDir.x = cosTheta * sinPhi;
		worldDir.y = cosPhi;
		worldDir.z = sinTheta * sinPhi;
	}

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

	shareSkyLit[localId] = ss.scatteredLight;
	shareTransmittance[localId] = ss.transmittance;

    groupMemoryBarrier();
    barrier();

    uint loopIndex = 32;
    while(loopIndex > 0)
    {
        if(localId < loopIndex)
        {
            shareSkyLit[localId] += shareSkyLit[localId + loopIndex];
			shareTransmittance[localId] += shareTransmittance[localId + loopIndex];
        }

        loopIndex /= 2;

        groupMemoryBarrier();
        barrier();
    }

    if (localId < 1)
    {
		vec3 avgColor = shareSkyLit[localId] / 64.0;
		vec3 avgTransmittance = shareTransmittance[localId] / 64.0;

		imageStore(imageDistantGrid, workPos, vec4(avgColor, 1.0 - mean(avgTransmittance)));
    }
}

#endif