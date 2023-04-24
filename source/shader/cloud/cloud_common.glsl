#ifndef VOLUMETRIC_CLOUD_COMMON_GLSL
#define VOLUMETRIC_CLOUD_COMMON_GLSL

// My personal volumetric cloud implement.
// Reference implement from https://www.slideshare.net/guerrillagames/the-realtime-volumetric-cloudscapes-of-horizon-zero-dawn.

#include "../common/shared_functions.glsl"

layout (set = 0, binding = 0, rgba16f) uniform image2D imageHdrSceneColor;
layout (set = 0, binding = 1) uniform texture2D inHdrSceneColor;

layout (set = 0, binding = 2, rgba16f) uniform image2D imageCloudRenderTexture; // quater resolution.
layout (set = 0, binding = 3) uniform texture2D inCloudRenderTexture; // quater resolution.

layout (set = 0, binding = 4) uniform texture2D inDepth;
layout (set = 0, binding = 5) uniform texture2D inGBufferA;

layout (set = 0, binding = 6) uniform texture3D inBasicNoise;
layout (set = 0, binding = 7) uniform texture3D inWorleyNoise;

layout (set = 0, binding = 8) uniform texture2D inWeatherTexture;
layout (set = 0, binding = 9) uniform texture2D inCloudCurlNoise;

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


layout (set = 0, binding = 22, rgba16f) uniform imageCube imageCubeEnv; // Cube capture.

layout (set = 0, binding = 23) uniform texture2D inCloudGradientLut;

layout (set = 0, binding = 24) uniform texture2D inCloudSkyViewLutBottom;
layout (set = 0, binding = 25) uniform texture2D inCloudSkyViewLutTop;
layout (set = 0, binding = 26) uniform texture2D inSkyViewLut;

layout (set = 0, binding = 27) uniform texture2D inSDSMShadowDepth;
layout (set = 0, binding = 28) buffer SSBOCascadeInfoBuffer{ CascadeInfo cascadeInfos[]; };

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

///////////////////////////////////////////////////////////////////////////////////////
//////////////// Paramters ///////////////////////////

// Min max sample count define.
#define kSampleCountMin 2
#define kSampleCountMax 96

// Max sample count per distance. 16 tap/km
#define kCloudDistanceToSampleMaxCount (1.0 / 16.0)

#define ENABLE_GROUND_CONTRIBUTION 0
#define kGroundContributionSampleCount 2

#define kVolumetricLightSteps 6

#define kMsCount 2

struct ParticipatingMedia
{
	float extinctionCoefficients[kMsCount];
    float transmittanceToLight[kMsCount];
};

/////////////////////////////////////////////////////
////////////////////////////////////////////////////
// Cloud shape.

float remap(float value, float orignalMin, float orignalMax, float newMin, float newMax)
{
    return newMin + (saturate((value - orignalMin) / (orignalMax - orignalMin)) * (newMax - newMin));
}

// TODO: change shape function in the future. This should be artist bias.
float cloudMap(vec3 posMeter, float normalizeHeight)  // Meter
{
    // If normalize height out of range, pre-return.
    // May evaluate error value on shadow light.
    if(normalizeHeight < 1e-4f || normalizeHeight > 0.9999f)
    {
        return 0.0f;
    }

    const float kCoverage = frameData.earthAtmosphere.cloudCoverage;
    const float kDensity  = frameData.earthAtmosphere.cloudDensity;

    const vec3 windDirection = frameData.earthAtmosphere.cloudDirection;
    const float cloudSpeed = frameData.earthAtmosphere.cloudSpeed;

    posMeter += windDirection * normalizeHeight * 500.0f;
    vec3 posKm = posMeter * 0.001; 

    vec3 windOffset = (windDirection + vec3(0.0, 0.1, 0.0)) * frameData.appTime.x * cloudSpeed;

    vec2 sampleUv = posKm.xz * frameData.earthAtmosphere.cloudWeatherUVScale;
    vec4 weatherValue = texture(sampler2D(inWeatherTexture, linearRepeatSampler), sampleUv);

    float coverage = saturate(kCoverage * weatherValue.x);
	float gradienShape = remap(normalizeHeight, 0.00, 0.10, 0.1, 1.0) * remap(normalizeHeight, 0.10, 0.80, 1.0, 0.2);

    float basicNoise = texture(sampler3D(inBasicNoise, linearRepeatSampler), (posKm + windOffset) * vec3(frameData.earthAtmosphere.cloudBasicNoiseScale)).r;
    float basicCloudNoise = gradienShape * basicNoise;

	float basicCloudWithCoverage = coverage * remap(basicCloudNoise, 1.0 - coverage, 1, 0, 1);

    vec3 sampleDetailNoise = posKm - windOffset * 0.15 + vec3(basicNoise.x, 0.0, basicCloudNoise) * normalizeHeight;
    float detailNoiseComposite = texture(sampler3D(inWorleyNoise, linearRepeatSampler), sampleDetailNoise * frameData.earthAtmosphere.cloudDetailNoiseScale).r;
	float detailNoiseMixByHeight = 0.2 * mix(detailNoiseComposite, 1 - detailNoiseComposite, saturate(normalizeHeight * 10.0));
    
    float densityShape = saturate(0.01 + normalizeHeight * 1.15) * kDensity *
        remap(normalizeHeight, 0.0, 0.1, 0.0, 1.0) * 
        remap(normalizeHeight, 0.8, 1.0, 1.0, 0.0);

    float cloudDensity = remap(basicCloudWithCoverage, detailNoiseMixByHeight, 1.0, 0.0, 1.0);
	return cloudDensity * densityShape;
 
}

// Cloud shape end.
////////////////////////////////////////////////////////////////////////////////////////

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

float powder(float opticalDepth)
{
	return 1.0 - exp2(-opticalDepth * 2.0);
}

vec3 lookupSkylight(vec3 worldDir, vec3 worldPos, float viewHeight, vec3 upVector, ivec2 workPos, in const AtmosphereParameters atmosphere, texture2D lutImage)
{
    const vec3 sunDirection = -normalize(frameData.directionalLight.direction);

    float viewZenithCosAngle = dot(worldDir, upVector);
    // Assumes non parallel vectors
	vec3 sideVector = normalize(cross(upVector, worldDir));		

    // aligns toward the sun light but perpendicular to up vector
	vec3 forwardVector = normalize(cross(sideVector, upVector));	

    vec2 lightOnPlane = vec2(dot(sunDirection, forwardVector), dot(sunDirection, sideVector));
	lightOnPlane = normalize(lightOnPlane);
	float lightViewCosAngle = lightOnPlane.x;

    vec2 sampleUv;
    vec3 luminance;

    skyViewLutParamsToUv(atmosphere, false, viewZenithCosAngle, lightViewCosAngle, viewHeight, vec2(textureSize(lutImage, 0)), sampleUv);
	luminance = texture(sampler2D(lutImage, linearClampEdgeSampler), sampleUv).rgb;

    return skyPrepareOut(luminance, atmosphere, frameData, vec2(workPos));
}

ParticipatingMedia volumetricShadow(vec3 posKm, float cosTheta, vec3 sunDirection, in const AtmosphereParameters atmosphere)
{
    ParticipatingMedia participatingMedia;

	int ms = 0;

	float extinctionAccumulation[kMsCount];
    float extinctionCoefficients[kMsCount];

	for (ms = 0; ms < kMsCount; ms++)
	{
		extinctionAccumulation[ms] = 0.0f;
        extinctionCoefficients[ms] = 0.0f;
	}

    const float kStepLMul = frameData.earthAtmosphere.cloudLightStepMul;
    const uint kStepLight = frameData.earthAtmosphere.cloudLightStepNum;
    float stepL = frameData.earthAtmosphere.cloudLightBasicStep; // km
    
    float d = stepL * 0.5;

	// Collect total density along light ray.
	for(uint j = 0; j < kStepLight; j++)
    {
        vec3 samplePosKm = posKm + sunDirection * d; // km

        float sampleHeightKm = length(samplePosKm);
        float sampleDt = sampleHeightKm - atmosphere.cloudAreaStartHeight;

        float normalizeHeight = sampleDt / atmosphere.cloudAreaThickness;
        vec3 samplePosMeter = samplePosKm * 1000.0f;

        extinctionCoefficients[0] = cloudMap(samplePosMeter, normalizeHeight);
        extinctionAccumulation[0] += extinctionCoefficients[0] * stepL;

        float MsExtinctionFactor = frameData.earthAtmosphere.cloudMultiScatterExtinction;
        for (ms = 1; ms < kMsCount; ms++)
		{
            extinctionCoefficients[ms] = extinctionCoefficients[ms - 1] * MsExtinctionFactor;
            MsExtinctionFactor *= MsExtinctionFactor;

			extinctionAccumulation[ms] += extinctionCoefficients[ms] * stepL;
		}

        d += stepL;
        stepL *= kStepLMul;
	}

    for (ms = 0; ms < kMsCount; ms++)
	{
		participatingMedia.transmittanceToLight[ms] = exp(-extinctionAccumulation[ms] * 1000.0); // to meter.
	}

    return participatingMedia;
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

        vec3 samplePosMeter = samplePosKm * 1000.0f;
        float stepCloudDensity = cloudMap(samplePosMeter, normalizeHeight);

		opticalDepth += stepCloudDensity * delta;
	}
	
	const vec3 scatteredLuminance = atmosphereTransmittanceToLight * sunIlluminance * groundToCloudTransfertIsoScatter;
	return scatteredLuminance * exp(-opticalDepth * contributionStepLength * 1000.0); // to meter.
}

vec4 cloudColorCompute(in const AtmosphereParameters atmosphere, vec2 uv, float blueNoise, inout float cloudZ, ivec2 workPos, vec3 worldDir)
{
    // Get camera in atmosphere unit position, it will treat as ray start position.
    vec3 worldPos = convertToAtmosphereUnit(viewData.camWorldPos.xyz, viewData) + vec3(0.0, atmosphere.bottomRadius, 0.0);

    float earthRadius = atmosphere.bottomRadius;
    float radiusCloudStart = atmosphere.cloudAreaStartHeight;
    float radiusCloudEnd = radiusCloudStart + atmosphere.cloudAreaThickness;

    // Unit is atmosphere unit. km.
    float viewHeight = length(worldPos);

    // Find intersect position so we can do some ray marching.
    float tMin;
    float tMax;
    if(viewHeight < radiusCloudStart)
    {
        float tEarth = raySphereIntersectNearest(worldPos, worldDir, vec3(0.0), earthRadius);
        if(tEarth > 0.0)
        {
            // Intersect with earth, pre-return.
            return vec4(0.0, 0.0, 0.0, 1.0);
        }

        tMin = raySphereIntersectInside(worldPos, worldDir, vec3(0.0), radiusCloudStart);
        tMax = raySphereIntersectInside(worldPos, worldDir, vec3(0.0), radiusCloudEnd);
    }
    else if(viewHeight > radiusCloudEnd)
    {
        // Eye out of cloud area.

        vec2 t0t1 = vec2(0.0);
        const bool bIntersectionEnd = raySphereIntersectOutSide(worldPos, worldDir, vec3(0.0), radiusCloudEnd, t0t1);
        if(!bIntersectionEnd)
        {
            // No intersection.
            return vec4(0.0, 0.0, 0.0, 1.0);
        }

        vec2 t2t3 = vec2(0.0);
        const bool bIntersectionStart = raySphereIntersectOutSide(worldPos, worldDir, vec3(0.0), radiusCloudStart, t2t3);
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
    else
    {
        // Eye inside cloud area.
        float tStart = raySphereIntersectNearest(worldPos, worldDir, vec3(0.0), radiusCloudStart);
        if(tStart > 0.0)
        {
            tMax = tStart;
        }
        else
        {
            tMax = raySphereIntersectInside(worldPos, worldDir, vec3(0.0), radiusCloudEnd);
        }

        tMin = 0.0f; // From camera.
    }

    tMin = max(tMin, 0.0);
    tMax = max(tMax, 0.0);

    // Pre-return if too far.
    if(tMax <= tMin || tMin > frameData.earthAtmosphere.cloudTracingStartMaxDistance)
    {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }

    // Clamp marching distance by setting.    
    const float marchingDistance = min(frameData.earthAtmosphere.cloudMaxTraceingDistance, tMax - tMin);
	tMax = tMin + marchingDistance;

    const uint stepCountUnit =  uint(max(kSampleCountMin, kSampleCountMax * saturate((tMax - tMin) * kCloudDistanceToSampleMaxCount)));
    const float stepCount = float(stepCountUnit);
    const float stepT = (tMax - tMin) / stepCount; // Per step lenght.

    float sampleT = tMin + 0.001 * stepT; // Slightly delta avoid self intersect.

    // Jitter by blue noise.
    sampleT += stepT * blueNoise; 
    
    vec3 sunColor = frameData.directionalLight.color * frameData.directionalLight.intensity;
    vec3 sunDirection = -normalize(frameData.directionalLight.direction);

    float VoL = dot(worldDir, sunDirection);

    // Combine backward and forward scattering to have details in all directions.
    const float cosTheta = -VoL;
    float phase = dualLobPhase(frameData.earthAtmosphere.cloudPhaseForward, frameData.earthAtmosphere.cloudPhaseBackward, frameData.earthAtmosphere.cloudPhaseMixFactor, cosTheta);

    ParticipatingMediaPhase participatingMediaPhase = getParticipatingMediaPhase(phase, frameData.earthAtmosphere.cloudPhaseMixFactor);

    float transmittance  = 1.0;
    vec3 scatteredLight = vec3(0.0, 0.0, 0.0);

    // Average ray hit pos to evaluate air perspective and height fog.
    vec3 rayHitPos = vec3(0.0);
    float rayHitPosWeight = 0.0;

    // Skylight lookup between top and bottom sky. mix by height.
    // 
    vec3 cloudTopSkyLight     = lookupSkylight(worldDir, vec3(0.0, atmosphere.cloudAreaStartHeight + atmosphere.cloudAreaThickness, 0.0), atmosphere.cloudAreaStartHeight + atmosphere.cloudAreaThickness, vec3(0.0, 1.0, 0.0), workPos, atmosphere, inCloudSkyViewLutTop);
    vec3 cloudBottomSkyLight  = lookupSkylight(worldDir, vec3(0.0, atmosphere.cloudAreaStartHeight, 0.0), atmosphere.cloudAreaStartHeight, vec3(0.0, 1.0, 0.0), workPos, atmosphere, inCloudSkyViewLutBottom);

    // Cloud background sky color.
    vec3 skyBackgroundColor = lookupSkylight(worldDir, worldPos, viewHeight, normalize(worldPos), workPos, atmosphere, inSkyViewLut);

    for(uint i = 0; i < stepCountUnit; i ++)
    {
        // World space sample pos, in km unit.
        vec3 samplePos = sampleT * worldDir + worldPos;

        float sampleHeight = length(samplePos);

        // Get sample normalize height [0.0, 1.0]
        float normalizeHeight = (sampleHeight - atmosphere.cloudAreaStartHeight)  / atmosphere.cloudAreaThickness;

        // Convert to meter.
        vec3 samplePosMeter = samplePos * 1000.0f;
        float stepCloudDensity = cloudMap(samplePosMeter, normalizeHeight);

        // Add ray march pos, so we can do some average fading or atmosphere sample effect.
        rayHitPos += samplePos * transmittance;
        rayHitPosWeight += transmittance;

        if(stepCloudDensity > 0.) 
        {
            float opticalDepth = stepCloudDensity * stepT * 1000.0; // to meter unit.

            // Second evaluate transmittance due to participating media
            vec3 atmosphereTransmittance;
            {
                const vec3 upVector = samplePos / sampleHeight;
                float viewZenithCosAngle = dot(sunDirection, upVector);
                vec2 sampleUv;
                lutTransmittanceParamsToUv(atmosphere, viewHeight, viewZenithCosAngle, sampleUv);
                atmosphereTransmittance = texture(sampler2D(inTransmittanceLut, linearClampEdgeSampler), sampleUv).rgb;
            }

            // beer's lambert.
        #if 1
            float stepTransmittance = exp(-opticalDepth); 
        #else
            // Siggraph 2017's new step transmittance formula.
            float stepTransmittance = max(exp(-opticalDepth), exp(-opticalDepth * 0.25) * 0.7); 
        #endif

            // Compute powder term.
            float powderEffectTerm;
            {
            #if   0
                powderEffectTerm = powder(opticalDepth);
            #elif 0
                // Siggraph 2017's new powder formula.
                float depthProbability = 0.05 + pow(opticalDepth, remap(normalizeHeight, 0.3, 0.85, 0.5, 2.0));
                float verticalProbability = pow(remap(normalizeHeight, 0.07, 0.14, 0.1, 1.0), 0.8);
                powderEffectTerm = depthProbability * verticalProbability; //powder(opticalDepth * 2.0);
            #else
                // Unreal engine 5's implement powder formula.
                powderEffectTerm = pow(saturate(opticalDepth * frameData.earthAtmosphere.cloudPowderScale), frameData.earthAtmosphere.cloudPowderPow);
            #endif
            }

        #if 0
            vec3 ambientLight = skyBackgroundColor;
        #else
            vec3 ambientLight = mix(cloudBottomSkyLight, cloudTopSkyLight, normalizeHeight);
        #endif

        if(frameData.earthAtmosphere.cloudEnableGroundContribution != 0)
        {
            ambientLight += getVolumetricGroundContribution(
                atmosphere, 
                samplePos, 
                sunDirection, 
                sunColor, 
                atmosphereTransmittance,
                normalizeHeight
            );
        }

            // Amount of sunlight that reaches the sample point through the cloud 
            // is the combination of ambient light and attenuated direct light.
            vec3 sunlightTerm = frameData.earthAtmosphere.cloudShadingSunLightScale * sunColor; 

            ParticipatingMedia participatingMedia = volumetricShadow(samplePos, cosTheta, sunDirection, atmosphere);

            float sigmaS = stepCloudDensity;
            float sigmaE = sigmaS + 1e-4f;

            vec3 scatteringCoefficients[kMsCount];
            float extinctionCoefficients[kMsCount];

            vec3 albedo = frameData.earthAtmosphere.cloudAlbedo * vec3(powderEffectTerm);

            scatteringCoefficients[0] = sigmaS * albedo;
            extinctionCoefficients[0] = sigmaE;

            float MsExtinctionFactor = frameData.earthAtmosphere.cloudMultiScatterExtinction;
            float MsScatterFactor    = frameData.earthAtmosphere.cloudMultiScatterScatter;
            int ms;
            for (ms = 1; ms < kMsCount; ms++)
            {
                extinctionCoefficients[ms] = extinctionCoefficients[ms - 1] * MsExtinctionFactor;
                scatteringCoefficients[ms] = scatteringCoefficients[ms - 1] * MsScatterFactor;
                
                MsExtinctionFactor *= MsExtinctionFactor;
                MsScatterFactor    *= MsScatterFactor;
            }

            for (ms = kMsCount - 1; ms >= 0; ms--) // Should terminate at 0
            {
                float sunVisibilityTerm = participatingMedia.transmittanceToLight[ms];

                vec3 sunSkyLuminance = sunVisibilityTerm * sunlightTerm * participatingMediaPhase.phase[ms];
                sunSkyLuminance += (ms == 0 ? ambientLight : vec3(0.0, 0.0, 0.0));

                vec3 sactterLitStep = sunSkyLuminance * scatteringCoefficients[ms];

                // See slide 28 at http://www.frostbite.com/2015/08/physically-based-unified-volumetric-rendering-in-frostbite/
                scatteredLight += atmosphereTransmittance * transmittance * (sactterLitStep - sactterLitStep * stepTransmittance) / max(1e-4f, extinctionCoefficients[ms]);
            
                if(ms == 0)
                {
                    // Beer's law.
                    transmittance *= stepTransmittance;
                }
            }
        }

        if(transmittance <= 0.001)
        {
            break;
        }

        sampleT += stepT;
    }



    // Apply cloud transmittance.
    vec3 finalColor = skyBackgroundColor;

    // Apply some additional effect.
    if(transmittance <= 0.99999)
    {
        // Get average hit pos.
        rayHitPos /= rayHitPosWeight;

        vec3 rayHitInRender = convertToCameraUnit(rayHitPos - vec3(0.0, atmosphere.bottomRadius, 0.0), viewData);
        vec4 rayInH = viewData.camViewProj * vec4(rayHitInRender, 1.0);
        cloudZ = rayInH.z / rayInH.w;

        rayHitPos -= worldPos;
        float rayHitHeight = length(rayHitPos);

        // Fade effect apply.
        float fading = exp(-rayHitHeight * frameData.earthAtmosphere.cloudFogFade);
        float cloudTransmittanceFaded = mix(1.0, transmittance, fading);

        // Apply air perspective.
        float slice = aerialPerspectiveDepthToSlice(rayHitHeight);
        float weight = 1.0;
        if (slice < 0.5)
        {
            // We multiply by weight to fade to 0 at depth 0. That works for luminance and opacity.
            weight = saturate(slice * 2.0);
            slice = 0.5;
        }
        ivec3 sliceLutSize = textureSize(inFroxelScatter, 0);
        float w = sqrt(slice / float(sliceLutSize.z));	// squared distribution

        vec4 airPerspective = weight * texture(sampler3D(inFroxelScatter, linearClampEdgeSampler), vec3(uv, w));

        // Dual mix alpha.
        cloudTransmittanceFaded = mix(1.0, cloudTransmittanceFaded, 1.0 - airPerspective.a);

        finalColor *= cloudTransmittanceFaded;

        // Apply air perspective color.
        finalColor += airPerspective.rgb * (1.0 - cloudTransmittanceFaded);

        // Apply scatter color.
        finalColor += (1.0 - cloudTransmittanceFaded) * scatteredLight;

        // Update transmittance.
        transmittance = cloudTransmittanceFaded;
    }

    // Dual mix transmittance.
    return vec4(finalColor, transmittance);
}

#endif