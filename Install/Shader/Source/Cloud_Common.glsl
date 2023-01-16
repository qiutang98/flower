#ifndef VOLUMETRIC_CLOUD_COMMON_GLSL
#define VOLUMETRIC_CLOUD_COMMON_GLSL

/*
** Physical based render code, develop by engineer: qiutanguu.
*/

// My personal volumetric cloud implement.
// Reference implement from https://www.slideshare.net/guerrillagames/the-realtime-volumetric-cloudscapes-of-horizon-zero-dawn.

#include "Common.glsl"
#include "Bayer.glsl"

layout (set = 0, binding = 0, rgba16f) uniform image2D imageHdrSceneColor;
layout (set = 0, binding = 1) uniform texture2D inHdrSceneColor;

layout (set = 0, binding = 2, rgba16f) uniform image2D imageCloudRenderTexture; // quater resolution.
layout (set = 0, binding = 3) uniform texture2D inCloudRenderTexture; // quater resolution.

layout (set = 0, binding = 4) uniform texture2D inDepth;
layout (set = 0, binding = 5) uniform texture2D inGBufferA;

layout (set = 0, binding = 6) uniform texture3D inBasicNoise;
layout (set = 0, binding = 7) uniform texture3D inDetailNoise;

layout (set = 0, binding = 8) uniform texture2D inWeatherTexture; // .r .g is cloud type, .b is wetness



layout (set = 0, binding = 9) uniform texture2D inCurlNoise;

layout (set = 0, binding = 10) uniform texture2D inTransmittanceLut;
layout (set = 0, binding = 11) uniform texture3D inFroxelScatter;

layout (set = 0, binding = 12, r32f) uniform image2D imageCloudShadowDepth;
layout (set = 0, binding = 13) uniform texture2D inCloudShadowDepth;

layout (set = 0, binding = 14, rgba16f) uniform image2D imageCloudReconstructionTexture;  // full resolution.
layout (set = 0, binding = 15) uniform texture2D inCloudReconstructionTexture;  // full resolution.

layout (set = 0, binding = 16, r32f) uniform image2D imageCloudDepthTexture;  // quater resolution.
layout (set = 0, binding = 17) uniform texture2D inCloudDepthTexture;  // quater resolution.

layout (set = 0, binding = 18, r32f) uniform image2D imageCloudDepthReconstructionTexture;  // full resolution.
layout (set = 0, binding = 19) uniform texture2D inCloudDepthReconstructionTexture;  // full resolution.

layout (set = 0, binding = 20) uniform texture2D inCloudReconstructionTextureHistory;
layout (set = 0, binding = 21) uniform texture2D inCloudDepthReconstructionTextureHistory;


layout (set = 0, binding = 22, r8) uniform image2D imageGbufferTranslucentMask;  // full resolution.

// Other common set.
layout (set = 1, binding = 0) uniform UniformView  { ViewData  viewData;  };
layout (set = 2, binding = 0) uniform UniformFrame { FrameData frameData; };

// Common sampler set.
#define COMMON_SAMPLER_SET 3
#include "CommonSamplerSet.glsl"

#define BLUE_NOISE_TEXTURE_SET 4
#define BLUE_NOISE_BUFFER_SET 5
#include "BlueNoiseCommon.glsl"

// Helper header.
#include "RayCommon.glsl"
#include "Sample.glsl"
#include "Phase.glsl"

// Config.



// 4 positions of a black, white, white, black gradient
const vec4 kCloudGradientSmall  = vec4(0.02f, 0.10f, 0.12f, 0.28f);
const vec4 kCloudGradientMedium = vec4(0.02f, 0.10f, 0.39f, 0.59f);
const vec4 kCloudGradientLarge  = vec4(0.02f, 0.07f, 0.88f, 1.00f);

const float kWeatherDensityAmount = 0.5;

const float kWeatherUVScale  = 0.009;

const float kCoverageAmount  = 0.8f;
const float kCoverageMinimum = 0.15f;

const float kTypeAmount  = 0.68f;
const float kTypeMinimum = 0.1f;

const float kAnvilAmount = 0.5f;
const float kAnvilOverhangHeight = 3.0f;

const float kSkewAlongWindDirection = 0.9f;

const float kTotalNoiseScale = 0.2;
const float kDetailScale     = 2.0;

const float kCurlScale = 0.5f;
const float kCurlNoiseModifier = 0.5;

const float kDetailNoiseModifier = 0.25f;
const float kDetailNoiseHeightFraction = 10.0f;


const vec3 kWindDirection = vec3(1.0, 0.0, 1.0);
const float kWindSpeed = 0.05;
const vec2 kCoverageDirection = vec2(-1.0, -1.0) * 0.01;
const float kAnimationMultiplier = 2.0;
//

// Utility function for cloud modeling.

float remap(float value, float orignalMin, float orignalMax, float newMin, float newMax)
{
    return newMin + (saturate((value - orignalMin) / (orignalMax - orignalMin)) * (newMax - newMin));
}

float getCloudGradient(vec4 gradient, float normalizeHeight)
{
	return smoothstep(gradient.x, gradient.y, normalizeHeight) - smoothstep(gradient.z, gradient.w, normalizeHeight);
}

float getCloudDensityHeightGradient(float normalizeHeight, vec3 weatherData)
{
	float cloudType = weatherData.g;
    
	float smallType   = 1.0f - saturate(cloudType * 2.0f);
	float mediumType  = 1.0f - abs(cloudType - 0.5f) * 2.0f;
	float largeType   = saturate(cloudType - 0.5f)   * 2.0f;

	vec4 cloudGradient =
		(kCloudGradientSmall  * smallType ) +
		(kCloudGradientMedium * mediumType) +
		(kCloudGradientLarge  * largeType );
	
	return getCloudGradient(cloudGradient, normalizeHeight);
}

vec3 sampleCloudWeatherMap(vec3 posKm, float normalizeHeight, vec2 coverageWindOffset)
{
	vec4 weatherData = texture(sampler2D(inWeatherTexture, linearRepeatSampler), (posKm.xz + coverageWindOffset) * kWeatherUVScale);
	
    // Apply effects for coverage
	weatherData.r = remap(weatherData.r * kCoverageAmount, 0.0, 1.0, kCoverageMinimum, 1.0); // Coverage 
	weatherData.g = remap(weatherData.g * kTypeAmount,     0.0, 1.0, kTypeMinimum,     1.0); // Type
    
	// Apply anvil clouds to coverage
	weatherData.r = pow(weatherData.r, max(remap(pow(1.0 - normalizeHeight, kAnvilOverhangHeight), 0.7, 0.8, 1.0, kAnvilAmount + 1.0), 0.0));
	
	return weatherData.rgb;
}

vec2 getWeatherOffset()
{
    return kAnimationMultiplier * kCoverageDirection * frameData.appTime.x;
}

float weatherDensity(vec3 weatherData)
{
	const float wetness = saturate(weatherData.b);
	return mix(1.0, 1.0 - kWeatherDensityAmount, wetness);
}

float cloudMap(vec3 posKm, float normalizeHeight, vec3 weatherData)
{
	vec3 posSampleKm = posKm + kWindSpeed * kWindDirection * frameData.appTime.x * kAnimationMultiplier;
	posSampleKm += normalizeHeight * kWindDirection * kSkewAlongWindDirection; // km
        
	float cloudSample = texture(sampler3D(inBasicNoise, linearRepeatSampler), posSampleKm * kTotalNoiseScale).r;

	// Apply height gradients
	float densityHeightGradient = getCloudDensityHeightGradient(normalizeHeight, weatherData);
	cloudSample *= densityHeightGradient;

	float cloudCoverage = weatherData.r;

	// Apply Coverage to sample
	cloudSample = cloudCoverage * remap(cloudSample, 1.0 - cloudCoverage, 1.0, 0.0, 1.0);
	
    // Erode with detail noise if cloud sample > 0
	if (cloudSample > 0.0)
	{
        // Apply curl noise to erode with tiny details.
        vec3 cv = texture(sampler2D(inCurlNoise, linearRepeatSampler), posKm.xz * kCurlScale * kTotalNoiseScale).rgb;

		vec3 curlNoise = 2.0 * cv - 1.0;
		posSampleKm += vec3(curlNoise.r, curlNoise.b, curlNoise.g) * (1.0 - normalizeHeight) * kCurlNoiseModifier;

        // Sample high frequency fbm.
		float highFrequencyFBM = texture(sampler3D(inDetailNoise, linearRepeatSampler), posSampleKm * kDetailScale * kTotalNoiseScale).r;
    
        // Dilate detail noise based on height
		float highFrequenceNoiseModifier = mix(1.0 - highFrequencyFBM, highFrequencyFBM, saturate(normalizeHeight * kDetailNoiseHeightFraction));
        
        // Erode with base of clouds
		cloudSample = remap(cloudSample, highFrequenceNoiseModifier * kDetailNoiseModifier, 1.0, 0.0, 1.0);
	}
	
	return max(cloudSample, 0.0);
}


///////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////
// Cloud shape.

// Cloud shape end.
////////////////////////////////////////////////////////////////////////////////////////


// Shadow light
#define kShadowLightStepNum 5
#define kShadowStepLength 3.0

const vec3 kExtinctionCoefficient = vec3(0.71f, 0.86f, 1.0f) * 0.02;

const float kMultiScatteringScattering   = 1.0;
const float kMultiScatteringExtinction   = 0.1;
const float kMultiScatteringEccentricity = 0.2;

const vec3  kAlbedo = vec3(1.0, 1.0, 1.0);
const float kBeerPowder = 20.0f;
const float kBeerPowderPower = 0.5f;


const uint kGroundContributionSampleCount = 2;

// Octaves for approximate multiple-scattering.
#define kMsCount 2

float multiPhase(float VoL)
{
	float forwardG  =  0.8;
	float backwardG = -0.2;

	float phases = 0.0;

	float c = 1.0;
    for (int i = 0; i < 4; i++)
    {
        phases += mix(hgPhase(backwardG * c, VoL), hgPhase(forwardG * c, VoL), 0.5) * c;
        c *= 0.5;
    }

	return phases;
}

struct ParticipatingMedia
{
	vec3 scatteringCoefficients[kMsCount];
	vec3 extinctionCoefficients[kMsCount];
	vec3 transmittanceToLight[kMsCount];
};

struct ParticipatingMediaPhase
{
	float phase[kMsCount];
};

ParticipatingMediaPhase getParticipatingMediaPhase(float basePhase, float baseMsPhaseFactor)
{
	ParticipatingMediaPhase participatingMediaPhase;
	participatingMediaPhase.phase[0] = basePhase;

	const float uniformPhase = getUniformPhase();
	float MsPhaseFactor = baseMsPhaseFactor;
	
	for (int ms = 1; ms < kMsCount; ms++)
	{
		participatingMediaPhase.phase[ms] = mix(uniformPhase, participatingMediaPhase.phase[0], MsPhaseFactor);
		MsPhaseFactor *= MsPhaseFactor;
	}

	return participatingMediaPhase;
}

ParticipatingMedia getParticipatingMedia(
    vec3 baseAlbedo, 
    vec3 baseExtinctionCoefficients, 
    float baseMsScatteringFactor, 
    float baseMsExtinctionFactor, 
    vec3 initialTransmittanceToLight)
{
	const vec3 scatteringCoefficients = baseAlbedo * baseExtinctionCoefficients;

	ParticipatingMedia participatingMedia;
	participatingMedia.scatteringCoefficients[0] = scatteringCoefficients;
	participatingMedia.extinctionCoefficients[0] = baseExtinctionCoefficients;
	participatingMedia.transmittanceToLight[0] = initialTransmittanceToLight;

	float MsScatteringFactor = baseMsScatteringFactor;
	float MsExtinctionFactor = baseMsExtinctionFactor;

	for (int ms = 1; ms < kMsCount; ++ms)
	{
		participatingMedia.scatteringCoefficients[ms] = participatingMedia.scatteringCoefficients[ms - 1] * MsScatteringFactor;
		participatingMedia.extinctionCoefficients[ms] = participatingMedia.extinctionCoefficients[ms - 1] * MsExtinctionFactor;
		MsScatteringFactor *= MsScatteringFactor;
		MsExtinctionFactor *= MsExtinctionFactor;

		participatingMedia.transmittanceToLight[ms] = initialTransmittanceToLight;
	}

	return participatingMedia;
}

void getVolumetricShadow(
    inout ParticipatingMedia participatingMedia, 
    in const AtmosphereParameters atmosphere,
    vec3 posKm,
    vec3 sunDirection)
{
	int ms = 0;
	vec3 extinctionAccumulation[kMsCount];

	for (ms = 0; ms < kMsCount; ms++)
	{
		extinctionAccumulation[ms] = vec3(0.0f);
	}

    const int sampleCount = kShadowLightStepNum;
    const float sampleSegmentT = 0.5f;

    // Collect total density along light ray.
    float intensitySum = 0.0;
	for(int j = 0; j < sampleCount; j++)
    {
		float t0 = float(j) / float(sampleCount);
		float t1 = float(j + 1.0) / float(sampleCount);

		// Non linear distribution of sample within the range.
		t0 = t0 * t0;
		t1 = t1 * t1;

        float delta = t1 - t0; // 5 samples: 0.04, 0.12, 0.2, 0.28, 0.36
		float t = t0 + delta * sampleSegmentT; // 5 samples: 0.02, 0.1, 0.26, 0.5, 0.82

        float shadowSampleT = kShadowStepLength * t; // km
        vec3 samplePosKm = posKm + sunDirection * shadowSampleT; // km

        float sampleHeightKm = length(samplePosKm);
        float sampleDt = sampleHeightKm - atmosphere.cloudAreaStartHeight;

        float normalizeHeight = sampleDt / atmosphere.cloudAreaThickness;

        vec3 weatherData = sampleCloudWeatherMap(samplePosKm, normalizeHeight, getWeatherOffset());
        float cloudIntensity = cloudMap(samplePosKm, normalizeHeight, weatherData);

        vec3 shadowExtinction = kExtinctionCoefficient * cloudIntensity;

        // Accumulate extinctionCoefficients.
		ParticipatingMedia shadowParticipatingMedia = getParticipatingMedia(
            vec3(0.0f),                 // baseAlbedo
            shadowExtinction,           // baseExtinctionCoefficients
            kMultiScatteringScattering, // baseMsScatteringFactor
            kMultiScatteringExtinction, // baseMsExtinctionFactor
            vec3(0.0f)                  // initialTransmittanceToLight
        );
        
		for (ms = 0; ms < kMsCount; ms++)
		{
			extinctionAccumulation[ms] += shadowParticipatingMedia.extinctionCoefficients[ms] * delta;
		}
    }

    for (ms = 0; ms < kMsCount; ms++)
	{
		participatingMedia.transmittanceToLight[ms] *= exp(-extinctionAccumulation[ms] * kShadowStepLength * 1000.0); // to meter.
	}
}

vec3 getVolumetricGroundContribution(
    in AtmosphereParameters atmosphere, 
    vec3 posKm, 
    vec3 sunDirection, 
    vec3 sunIlluminance, 
    vec3 atmosphereTransmittanceToLight,
    float posNormalizeHeight)
{
    const vec3 groundScatterDirection = vec3(0.0, -1.0, 0.0); // Y down.
    const vec3 planetSurfaceNormal    = vec3(0.0,  1.0, 0.0); // Ambient contribution from the clouds is only done on a plane above the planet

    const vec3 groundBrdfNdotL = saturate(dot(sunDirection, planetSurfaceNormal)) * (atmosphere.groundAlbedo / kPI); // Lambert BRDF diffuse shading
	const float uniformPhase = getUniformPhase();
	const float groundHemisphereLuminanceIsotropic = (2.0f * kPI) * uniformPhase; // Assumes the ground is uniform luminance to the cloud and solid angle is bottom hemisphere 2PI
	const vec3 groundToCloudTransfertIsoScatter = groundBrdfNdotL * groundHemisphereLuminanceIsotropic;

	float cloudSampleHeightToBottom = posNormalizeHeight * atmosphere.cloudAreaThickness; // Distance from altitude to bottom of clouds
	
    
	vec3 opticalDepth = vec3(0.0); // km.
	
	const float contributionStepLength = min(4.0, cloudSampleHeightToBottom); // km
	
    // Ground Contribution tracing loop, same idea as volumetric shadow
	const uint sampleCount = kGroundContributionSampleCount;
	const float sampleSegmentT = 0.5f;
	for (uint s = 0; s < sampleCount; s ++)
	{
		// More expensive but artefact free
		float t0 = float(s) / float(sampleCount);
		float t1 = float(s + 1.0) / float(sampleCount);

		// Non linear distribution of sample within the range.
		t0 = t0 * t0;
		t1 = t1 * t1;

		float delta = t1 - t0; // 5 samples: 0.04, 0.12, 0.2, 0.28, 0.36		
		float t = t0 + (t1 - t0) * sampleSegmentT; // 5 samples: 0.02, 0.1, 0.26, 0.5, 0.82

		float contributionSampleT = contributionStepLength * t; // km
		vec3 samplePosKm = posKm + groundScatterDirection * contributionSampleT; // Km

        float sampleHeightKm = length(samplePosKm);
        float sampleDt = sampleHeightKm - atmosphere.cloudAreaStartHeight;

        float normalizeHeight = sampleDt / atmosphere.cloudAreaThickness;

        vec3 weatherData = sampleCloudWeatherMap(samplePosKm, normalizeHeight, getWeatherOffset());
        float cloudIntensity = cloudMap(samplePosKm, normalizeHeight, weatherData);

		vec3 contributionExtinction = kExtinctionCoefficient * cloudIntensity; // km

		opticalDepth += contributionExtinction * delta;
	}
	
	const vec3 scatteredLuminance = atmosphereTransmittanceToLight * sunIlluminance * groundToCloudTransfertIsoScatter;
	return scatteredLuminance * exp(-opticalDepth * contributionStepLength * 1000.0); // to meter.
}

#endif