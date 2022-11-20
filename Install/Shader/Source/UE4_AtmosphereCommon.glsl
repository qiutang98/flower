#ifndef ATMOSPHERE_COMMON_GLSL
#define ATMOSPHERE_COMMON_GLSL

#extension GL_EXT_samplerless_texture_functions : enable
#extension GL_GOOGLE_include_directive : require

#ifndef SAMPLE_SHADOW
#define SAMPLE_SHADOW 0
#endif

// Atmosphere transmittance lut compute, base on 
// https://github.com/sebh/UnrealEngineSkyAtmosphere
// https://ebruneton.github.io/precomputed_atmospheric_scattering/

// define NO_MULTISCATAPPROX_ENABLED

// All units in kilometers
#include "Common.glsl"
layout (set = 0, binding = 0, rgba16f) uniform image2D imageTransmittanceLut;
layout (set = 0, binding = 1) uniform texture2D inTransmittanceLut;

layout (set = 0, binding = 2, rgba16f) uniform image2D imageSkyViewLut;
layout (set = 0, binding = 3) uniform texture2D inSkyViewLut;

layout (set = 0, binding = 4, rgba16f) uniform image2D imageMultiScatterLut;
layout (set = 0, binding = 5) uniform texture2D inMultiScatterLut;

layout (set = 0, binding = 6) uniform texture2D inDepth;

layout (set = 0, binding = 7, rgba16f) uniform image3D imageFroxelScatter;
layout (set = 0, binding = 8) uniform texture3D inFroxelScatter;

layout (set = 0, binding = 9, rgba16f) uniform image2D imageHdrSceneColor;
layout (set = 0, binding = 10) uniform texture2D inGBufferA;

layout (set = 0, binding = 11) uniform texture2D inSDSMShadowDepth;
layout (set = 0, binding = 12) buffer SSBOCascadeInfoBuffer{ CascadeInfo cascadeInfos[]; };

layout (set = 0, binding = 13, rgba16f) writeonly uniform imageCube imageCubeEnv;

// Other common set.
layout (set = 1, binding = 0) uniform UniformView  { ViewData  viewData;  };
layout (set = 2, binding = 0) uniform UniformFrame { FrameData frameData; };

// Common sampler set.
#define COMMON_SAMPLER_SET 3
#include "CommonSamplerSet.glsl"

// Helper header.
#include "RayCommon.glsl"
#include "Sample.glsl"
#include "Phase.glsl"
#include "Deband16.glsl"

// NOTE: When offset height is big, this value should be more bigger.
// TODO: Compute this by camera height.
const float kPlanetRadiusOffset = 0.001f; // Offset 1 m.
const float kAirPerspectiveKmPerSlice = 4.0f; // total 32 * 4 = 128 km.

float aerialPerspectiveDepthToSlice(float depth) { return depth * (1.0f / kAirPerspectiveKmPerSlice); }
float aerialPerspectiveSliceToDepth(float slice) { return slice * kAirPerspectiveKmPerSlice; }

// All units in kilometers
AtmosphereParameters getAtmosphereParameters()
{
	return getAtmosphereParameters(frameData);
}

vec3 convertToAtmosphereUnit(vec3 o)
{
	return convertToAtmosphereUnit(o, viewData);
}  

vec3 convertToCameraUnit(vec3 o)
{
	return convertToCameraUnit(o, viewData);
}  

vec3 prepareOut(vec3 inColor, in const AtmosphereParameters atmosphere, vec2 workPos)
{
	vec3 c = inColor * atmosphere.atmospherePreExposure * frameData.directionalLight.color * frameData.directionalLight.intensity; 

	// Maybe add blue noise jitter is better.
	// c = quantise(c, workPos, frameData);

	return c;
}

// Get shadow from sdsm.
float getShadow(in const AtmosphereParameters atmospehre, vec3 p)
{
#if SAMPLE_SHADOW
	if(frameData.bSdsmDraw <= 0)
	{
		return 1.0f;
	}

	// get position relative to camera.
	vec3 pRelativeToCam = p + vec3(0.0, -atmospehre.bottomRadius, 0.0);
	pRelativeToCam = convertToCameraUnit(pRelativeToCam);

	const DirectionalLightInfo light = frameData.directionalLight;

	// First find active cascade.
    vec3 shadowNdcPos;
	const float perCascadeOffsetUV = 1.0f / light.cascadeCount;

	// Use final cascade.
	uint cascadeId = light.cascadeCount - 1;

	// Project to shadow space.
	const vec4 shadowProjPos =  cascadeInfos[cascadeId].viewProj * vec4(pRelativeToCam, 1.0f);

	// Perspective divide to get ndc position.
	shadowNdcPos = shadowProjPos.xyz / shadowProjPos.w;
	
	// Compute shadow coord, all should range in [0, 1].
	vec3 shadowCoord = vec3(shadowNdcPos.xy * 0.5f + 0.5f, shadowNdcPos.z);

	// vulkan screen uv need revert in y directional.
	shadowCoord.y = 1.0f - shadowCoord.y;

	// Check current cascade is valid in range.
	if(onRange(shadowCoord.xyz, vec3(light.cascadeBorderAdopt), vec3(1.0f - light.cascadeBorderAdopt)))
	{
		// Compute shadow position.
		vec3 shadowPosOnAltas = shadowCoord.xyz;

		// Also add altas bias.
		shadowPosOnAltas.x = (shadowPosOnAltas.x + float(cascadeId)) * perCascadeOffsetUV;

		float sampleDepth = texture(sampler2D(inSDSMShadowDepth, pointClampEdgeSampler), shadowPosOnAltas.xy).r;
		return shadowPosOnAltas.z > sampleDepth ? 1.0f : 0.0f;
	}

	return 1.0f;
#else
	return 1.0f;
#endif
}

// Participating media functions and struct.
// From https://github.com/sebh/UnrealEngineSkyAtmosphere
struct MediumSampleRGB
{
	vec3 scattering;
	vec3 absorption;
	vec3 extinction;

	vec3 scatteringMie;
	vec3 absorptionMie;
	vec3 extinctionMie;

	vec3 scatteringRay;
	vec3 absorptionRay;
	vec3 extinctionRay;

	vec3 scatteringOzo;
	vec3 absorptionOzo;
	vec3 extinctionOzo;

	vec3 albedo;
};

float getAlbedo(float scattering, float extinction)
{
	return scattering / max(0.001, extinction);
}

vec3 getAlbedo(vec3 scattering, vec3 extinction)
{
	return scattering / max(vec3(0.001), extinction);
}

float getViewHeight(vec3 worldPos, in const AtmosphereParameters atmosphere)
{
	// Current default set planet center is (0, 0, 0).
    // And we start from horizontal plane which treat as 0 plane on height.
	return length(worldPos) - atmosphere.bottomRadius;
}

bool moveToTopAtmosphere(inout vec3 worldPos, in const vec3 worldDir, in const float atmosphereTopRadius)
{
	float viewHeight = length(worldPos);
	if (viewHeight > atmosphereTopRadius)
	{
		float tTop = raySphereIntersectNearest(worldPos, worldDir, vec3(0.0), atmosphereTopRadius);
		if (tTop >= 0.0f)
		{
			vec3 upVector = worldPos / viewHeight;
			vec3 upOffset = upVector * -kPlanetRadiusOffset;
			worldPos = worldPos + worldDir * tTop + upOffset;
		}
		else
		{
			// Ray is not intersecting the atmosphere
			return false;
		}
	}
	return true; // ok to start tracing
}

MediumSampleRGB sampleMediumRGB(in vec3 worldPos, in const AtmosphereParameters atmosphere)
{
	const float viewHeight = getViewHeight(worldPos, atmosphere);

    // Get mie and ray density.
	const float densityMie = exp(atmosphere.mieDensityExpScale * viewHeight);
	const float densityRay = exp(atmosphere.rayleighDensityExpScale * viewHeight);

    // Get ozone density.
	const float densityOzo = saturate(viewHeight < atmosphere.absorptionDensity0LayerWidth ?
		atmosphere.absorptionDensity0LinearTerm * viewHeight + atmosphere.absorptionDensity0ConstantTerm :
		atmosphere.absorptionDensity1LinearTerm * viewHeight + atmosphere.absorptionDensity1ConstantTerm);

    // Build medium sample.
	MediumSampleRGB s;

    // Mie term.
	s.scatteringMie = densityMie * atmosphere.mieScattering;
	s.absorptionMie = densityMie * atmosphere.mieAbsorption;
	s.extinctionMie = densityMie * atmosphere.mieExtinction;

    // Ray term.
	s.scatteringRay = densityRay * atmosphere.rayleighScattering;
	s.absorptionRay = vec3(0.0);
	s.extinctionRay = s.scatteringRay + s.absorptionRay;

    // Ozone term.
	s.scatteringOzo = vec3(0.0);
	s.absorptionOzo = densityOzo * atmosphere.absorptionExtinction;
	s.extinctionOzo = s.scatteringOzo + s.absorptionOzo;

    // Composite.
	s.scattering = s.scatteringMie + s.scatteringRay + s.scatteringOzo;
	s.absorption = s.absorptionMie + s.absorptionRay + s.absorptionOzo;
	s.extinction = s.extinctionMie + s.extinctionRay + s.extinctionOzo;
	s.albedo = getAlbedo(s.scattering, s.extinction);
	return s;
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

float fromUnitToSubUvs(float u, float resolution) { return (u + 0.5f / resolution) * (resolution / (resolution + 1.0f)); }
float fromSubUvsToUnit(float u, float resolution) { return (u - 0.5f / resolution) * (resolution / (resolution - 1.0f)); }

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
		coord *= coord; // Non-linear sky view lut.

		coord = 1.0 - coord;
		viewZenithCosAngle = cos(zenithHorizonAngle * coord);
	}
	else
	{
		float coord = uv.y * 2.0 - 1.0;
		coord *= coord; // Non-linear sky view lut.
		viewZenithCosAngle = cos(zenithHorizonAngle + beta * coord);
	}

	float coord = uv.x;
	coord *= coord;

	lightViewCosAngle = -(coord * 2.0 - 1.0);
}

void skyViewLutParamsToUv(
	in const AtmosphereParameters atmosphere, 
	in bool  bIntersectGround, 
	in float viewZenithCosAngle, 
	in float lightViewCosAngle, 
	in float viewHeight, 
	out vec2 uv)
{
	float vHorizon = sqrt(viewHeight * viewHeight - atmosphere.bottomRadius * atmosphere.bottomRadius);

	// Ground to horizon cos.
	float cosBeta = vHorizon / viewHeight;		

	float beta = acos(cosBeta);
	float zenithHorizonAngle = kPI - beta;

	if (!bIntersectGround)
	{
		float coord = acos(viewZenithCosAngle) / zenithHorizonAngle;
		coord = 1.0 - coord;
		coord = sqrt(coord); // Non-linear sky view lut.

		coord = 1.0 - coord;
		uv.y = coord * 0.5f;
	}
	else
	{
		float coord = (acos(viewZenithCosAngle) - zenithHorizonAngle) / beta;
		coord = sqrt(coord); // Non-linear sky view lut.

		uv.y = coord * 0.5f + 0.5f;
	}

	// UV x remap.
	{
		float coord = -lightViewCosAngle * 0.5f + 0.5f;
		coord = sqrt(coord);
		uv.x = coord;
	}

	// Constrain uvs to valid sub texel range (avoid zenith derivative issue making LUT usage visible)
	vec2 lutSize = vec2(textureSize(inSkyViewLut, 0));
	uv = vec2(fromUnitToSubUvs(uv.x, lutSize.x), fromUnitToSubUvs(uv.y, lutSize.y));
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
	// Theses code used when composite sky on the scene color texture.
	if (depthBufferValue >= 0.0 && depthBufferValue <= 1.0)
	{
		// World space depth.
		vec2 sampleUv = (pixPos + vec2(0.5)) / vec2(textureSize(inDepth, 0));
		vec3 depthBufferWorldPos = getWorldPos(sampleUv, depthBufferValue, viewData);

		// Apply earth offset to go back to origin as top of earth mode. 
		float tDepth = length(convertToAtmosphereUnit(depthBufferWorldPos) + vec3(0.0, atmosphere.bottomRadius, 0.0) - worldPos);
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

	// TODO: Color change.
	vec3 globalL = vec3(1.0f); // frameData.directionalLight.color * atmosphere.atmospherePreExposure * frameData.directionalLight.intensity;

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

			// TODO: try this.
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
		float tEarth = raySphereIntersectNearest(P, sunDir, kEarthOrigin + kPlanetRadiusOffset * upVector, atmosphere.bottomRadius);
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
		// Look unused. TODO: Delete it
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

#endif