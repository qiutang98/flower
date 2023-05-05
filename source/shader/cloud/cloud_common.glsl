#ifndef VOLUMETRIC_CLOUD_COMMON_GLSL
#define VOLUMETRIC_CLOUD_COMMON_GLSL

// My personal volumetric cloud implement.
// Reference implement from https://www.slideshare.net/guerrillagames/the-realtime-volumetric-cloudscapes-of-horizon-zero-dawn.

#include "../common/shared_functions.glsl"
#include "../common/shared_atmosphere.glsl"

        const float MsScattFactor = 1.0;
        const float MsExtinFactor = 0.1;
        const float MsPhaseFactor = 0.1;

layout (set = 0, binding = 0, rgba16f) uniform image2D imageHdrSceneColor;
layout (set = 0, binding = 1) uniform texture2D inHdrSceneColor;
layout (set = 0, binding = 2, rgba16f) uniform image2D imageCloudRenderTexture; // quater resolution.
layout (set = 0, binding = 3) uniform texture2D inCloudRenderTexture; // quater resolution.
layout (set = 0, binding = 4) uniform texture2D inDepth;
layout (set = 0, binding = 5) uniform texture2D inGBufferA;
layout (set = 0, binding = 6) uniform texture3D inBasicNoise;
layout (set = 0, binding = 7) uniform texture3D inDetailNoise;
layout (set = 0, binding = 8) uniform texture2D inWeatherTexture;
layout (set = 0, binding = 9) uniform texture2D inCloudCurlNoise;
layout (set = 0, binding = 10) uniform texture2D inTransmittanceLut;
layout (set = 0, binding = 11) uniform texture3D inFroxelScatter;
layout (set = 0, binding = 12, rgba16f) uniform image2D imageCloudReconstructionTexture;  // full resolution.
layout (set = 0, binding = 13) uniform texture2D inCloudReconstructionTexture;  // full resolution.
layout (set = 0, binding = 14, r32f) uniform image2D imageCloudDepthTexture;  // quater resolution.
layout (set = 0, binding = 15) uniform texture2D inCloudDepthTexture;  // quater resolution.
layout (set = 0, binding = 16, r32f) uniform image2D imageCloudDepthReconstructionTexture;  // full resolution.
layout (set = 0, binding = 17) uniform texture2D inCloudDepthReconstructionTexture;  // full resolution.
layout (set = 0, binding = 18) uniform texture2D inCloudReconstructionTextureHistory;
layout (set = 0, binding = 19) uniform texture2D inCloudDepthReconstructionTextureHistory;
layout (set = 0, binding = 20) uniform texture2D inSkyViewLut;
layout (set = 0, binding = 21) uniform UniformFrameData { PerFrameData frameData; };
layout (set = 0, binding = 22, rgba16f) uniform image2D imageCloudFogRenderTexture; // quater resolution.
layout (set = 0, binding = 23) uniform texture2D inCloudFogRenderTexture; // quater resolution.
layout (set = 0, binding = 24, rgba16f) uniform image2D imageCloudFogReconstructionTexture;  // full resolution.
layout (set = 0, binding = 25) uniform texture2D inCloudFogReconstructionTexture;  // full resolution.
layout (set = 0, binding = 26) uniform texture2D inCloudFogReconstructionTextureHistory;

#define SHARED_SAMPLER_SET 1
#include "../common/shared_sampler.glsl"

#define BLUE_NOISE_BUFFER_SET 2
#include "../common/shared_bluenoise.glsl"

float getDensity(vec3 worldPosition)
{
    const float fogStartHeight = 0.0;
	const float falloff = 0.001;
    const float constFog = 0.0;

    const float fogFinalScale = 0.00001 * frameData.sky.atmosphereConfig.cloudFogFade;
    return (constFog + exp(-(worldPosition.y - fogStartHeight) * falloff)) * fogFinalScale;
}

///////////////////////////////////////////////////////////////////////////////////////
//////////////// Paramters ///////////////////////////


// NOTE: 0 is fbm cloud.
//       1 is model based cloud.
// 1 is cute shape and easy get beautiful image.
// 0 is radom shape hard to control shape, but can get a spectacular result in some times.
#define CLOUD_SHAPE 0
#define kGroundContributionSampleCount 5

// Min max sample count define.
#define kMsCount 2

struct ParticipatingMediaContext
{
	vec3 ScatteringCoefficients[kMsCount];
	vec3 ExtinctionCoefficients[kMsCount];
	vec3 TransmittanceToLight[kMsCount];
};

vec3 getExtinction(float density)
{
    return vec3(0.719, 0.859, 1.0) * 0.05 * density;
}

ParticipatingMediaContext setupParticipatingMediaContext(
    vec3 BaseAlbedo, 
    vec3 BaseExtinctionCoefficients, 
    float MsSFactor, 
    float MsEFactor, 
    vec3 InitialTransmittanceToLight)
{
	const vec3 ScatteringCoefficients = BaseAlbedo * BaseExtinctionCoefficients;

	ParticipatingMediaContext PMC;
	PMC.ScatteringCoefficients[0] = ScatteringCoefficients;
	PMC.ExtinctionCoefficients[0] = BaseExtinctionCoefficients;
	PMC.TransmittanceToLight[0] = InitialTransmittanceToLight;

	for (int ms = 1; ms < kMsCount; ++ms)
	{
		PMC.ScatteringCoefficients[ms] = PMC.ScatteringCoefficients[ms - 1] * MsSFactor;
		PMC.ExtinctionCoefficients[ms] = PMC.ExtinctionCoefficients[ms - 1] * MsEFactor;
		MsSFactor *= MsSFactor;
		MsEFactor *= MsEFactor;
		PMC.TransmittanceToLight[ms] = InitialTransmittanceToLight;
	}

	return PMC;
}

/////////////////////////////////////////////////////
////////////////////////////////////////////////////
// Cloud shape.

float remap(float value, float orignalMin, float orignalMax, float newMin, float newMax)
{
    return newMin + (saturate((value - orignalMin) / (orignalMax - orignalMin)) * (newMax - newMin));
}

float cloudMap(vec3 posMeter, float normalizeHeight, int ocat, bool bFixDensity)  // Meter
{
    const float kCoverage = frameData.sky.atmosphereConfig.cloudCoverage;
    const float kDensity  = frameData.sky.atmosphereConfig.cloudDensity;
    const vec3  kWindDirection = frameData.sky.atmosphereConfig.cloudDirection;
    const float kCloudSpeed = frameData.sky.atmosphereConfig.cloudSpeed;

    vec2 samppleUv = (posMeter + vec3(5000.0f, 0.0f, 5000.0f)).xz * 0.00035 * 0.035;

    vec4 weatherValue = texture(sampler2D(inWeatherTexture, linearRepeatSampler), samppleUv);
    
	float gradienShape = texture(sampler2D(inCloudCurlNoise, linearClampEdgeSampler), vec2(pow(weatherValue.y, 0.25), 1.0 - normalizeHeight)).r;
    float basicDensity =  gradienShape * weatherValue.x;

    float basicNoise = texture(sampler3D(inBasicNoise, linearRepeatSampler), posMeter * 0.001 * vec3(frameData.sky.atmosphereConfig.cloudBasicNoiseScale)).r;
    basicNoise = basicDensity * saturate(pow(basicNoise, 5.0));


    float detailNoiseComposite = texture(sampler3D(inDetailNoise, linearRepeatSampler), posMeter * 0.001 * frameData.sky.atmosphereConfig.cloudDetailNoiseScale).r;
	float detailNoiseMixByHeight = 0.1 * mix(detailNoiseComposite, 1 - detailNoiseComposite, saturate(normalizeHeight * 10.0));

    return basicNoise;
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
	return pow(opticalDepth * 20.0, 0.5) * frameData.sky.atmosphereConfig.cloudPowderScale;
}

vec3 lookupSkylight(vec3 worldDir, vec3 worldPos, float viewHeight, vec3 upVector, ivec2 workPos, in const AtmosphereParameters atmosphere, texture2D lutImage)
{
    const vec3 sunDirection = -normalize(frameData.sky.direction);

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

    return luminance;
}

void volumetricShadow(inout ParticipatingMediaContext PMC, vec3 posKm, vec3 sunDirection, in const AtmosphereParameters atmosphere, int fixNum)
{
	int ms = 0;

	vec3 ExtinctionAcc[kMsCount];
	for (ms = 0; ms < kMsCount; ms++)
	{
		ExtinctionAcc[ms] = vec3(0.0f);
	}

    const float kStepLMul = frameData.sky.atmosphereConfig.cloudLightStepMul;
    const uint kStepLight = fixNum > 0 ? fixNum : frameData.sky.atmosphereConfig.cloudLightStepNum;
    float stepL = frameData.sky.atmosphereConfig.cloudLightBasicStep; // km
    
    float d = stepL * 0.5;

	// Collect total density along light ray.
	for(uint j = 0; j < kStepLight; j++)
    {
        vec3 samplePosKm = posKm + sunDirection * d; // km

        float sampleHeightKm = length(samplePosKm);
        float sampleDt = sampleHeightKm - atmosphere.cloudAreaStartHeight;

        float normalizeHeight = sampleDt / atmosphere.cloudAreaThickness;
        vec3 samplePosMeter = samplePosKm * 1000.0f;

        float stepDensity = cloudMap(samplePosMeter, normalizeHeight,  frameData.sky.atmosphereConfig.cloudSunLitMapOctave, false);
        vec3 ShadowExtinctionCoefficients = getExtinction(stepDensity);
        ParticipatingMediaContext ShadowPMC = setupParticipatingMediaContext(vec3(0.0f), ShadowExtinctionCoefficients, MsScattFactor, MsExtinFactor, vec3(0.0f));

        for (ms = 0; ms < kMsCount; ++ms)
        {
            ExtinctionAcc[ms] += ShadowPMC.ExtinctionCoefficients[ms] * stepL;
        }

        d += stepL;
        stepL *= kStepLMul;
	}

    //

    for (ms = 0; ms < kMsCount; ms++)
	{
		PMC.TransmittanceToLight[ms] *= exp(-ExtinctionAcc[ms] * 1000.0); // to meter.
	}
}

float powderEffectNew(float depth, float height, float VoL)
{
    float r = VoL * 0.5 + 0.5;
    r = r * r;
    height = height * (1.0 - r) + r;
    return depth * height;
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
        float stepCloudDensity = cloudMap(samplePosMeter, normalizeHeight, 2, true);

		opticalDepth += stepCloudDensity * delta;
	}
	
	const vec3 scatteredLuminance = atmosphereTransmittanceToLight * sunIlluminance * groundToCloudTransfertIsoScatter;
	return scatteredLuminance * exp(-opticalDepth * contributionStepLength * 1000.0); // to meter.
}



vec4 cloudColorCompute(
    in const AtmosphereParameters atmosphere, 
    vec2 uv, 
    float blueNoise, 
    inout float cloudZ, 
    ivec2 workPos, 
    vec3 worldDir, 
    bool bFog,
    inout vec4 lightingFog,
    float fogNoise)
{
    // Get camera in atmosphere unit position, it will treat as ray start position.
    vec3 worldPos = convertToAtmosphereUnit(frameData.camWorldPos.xyz, frameData) + vec3(0.0, atmosphere.bottomRadius, 0.0);

    lightingFog.w = -1.0f;

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
    if(tMax <= tMin || tMin > frameData.sky.atmosphereConfig.cloudTracingStartMaxDistance)
    {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }

    // Clamp marching distance by setting.    
    const float marchingDistance = min(frameData.sky.atmosphereConfig.cloudMaxTraceingDistance, tMax - tMin);
	tMax = tMin + marchingDistance;

    const uint stepCountUnit = frameData.sky.atmosphereConfig.cloudMarchingStepNum;
    const float stepCount = float(stepCountUnit);
    const float stepT = (tMax - tMin) / stepCount; // Per step lenght.

    float sampleT = tMin + 0.001 * stepT; // slightly delta avoid self intersect.

    // Jitter by blue noise.
    sampleT += stepT * blueNoise; 
    
    vec3 sunColor = frameData.sky.color * frameData.sky.intensity;
    vec3 sunDirection = -normalize(frameData.sky.direction);

    float VoL = dot(worldDir, sunDirection);

    // Combine backward and forward scattering to have details in all directions.
    float phase = 
            dualLobPhase(frameData.sky.atmosphereConfig.cloudPhaseForward, frameData.sky.atmosphereConfig.cloudPhaseBackward, frameData.sky.atmosphereConfig.cloudPhaseMixFactor, -VoL);


    ParticipatingMediaPhase participatingMediaPhase = getParticipatingMediaPhase(phase, MsPhaseFactor);

    vec3 TransmittanceToView  = vec3(1.0);
    vec3 scatteredLight = vec3(0.0, 0.0, 0.0);

    // Average ray hit pos to evaluate air perspective and height fog.
    vec3 rayHitPos = vec3(0.0);
    float rayHitPosWeight = 0.0;

    // Cloud background sky color.
    vec3 skyBackgroundColor = lookupSkylight(worldDir, worldPos, viewHeight, normalize(worldPos), workPos, atmosphere, inSkyViewLut);

    // Second evaluate transmittance due to participating media
    vec3 atmosphereTransmittance0;
    {
        vec3 samplePos = sampleT * worldDir + worldPos;
        float sampleHeight = length(samplePos);

        const vec3 upVector = samplePos / sampleHeight;
        float viewZenithCosAngle = dot(sunDirection, upVector);
        vec2 sampleUv;
        lutTransmittanceParamsToUv(atmosphere, viewHeight, viewZenithCosAngle, sampleUv);
        atmosphereTransmittance0 = texture(sampler2D(inTransmittanceLut, linearClampEdgeSampler), sampleUv).rgb;
    }
    vec3 atmosphereTransmittance1;
    {
        vec3 samplePos = tMax * worldDir + worldPos;
        float sampleHeight = length(samplePos);

        const vec3 upVector = samplePos / sampleHeight;
        float viewZenithCosAngle = dot(sunDirection, upVector);
        vec2 sampleUv;
        lutTransmittanceParamsToUv(atmosphere, viewHeight, viewZenithCosAngle, sampleUv);
        atmosphereTransmittance1 = texture(sampler2D(inTransmittanceLut, linearClampEdgeSampler), sampleUv).rgb;
    }

    for(uint i = 0; i < stepCountUnit; i ++)
    {
        // World space sample pos, in km unit.
        vec3 samplePos = sampleT * worldDir + worldPos;

        float sampleHeight = length(samplePos);

        vec3 atmosphereTransmittance = mix(atmosphereTransmittance0, atmosphereTransmittance1, saturate(sampleT / marchingDistance));

        // Get sample normalize height [0.0, 1.0]
        float normalizeHeight = (sampleHeight - atmosphere.cloudAreaStartHeight)  / atmosphere.cloudAreaThickness;

        // Convert to meter.
        vec3 samplePosMeter = samplePos * 1000.0f;
        float stepCloudDensity = cloudMap(samplePosMeter, normalizeHeight,  frameData.sky.atmosphereConfig.cloudSunLitMapOctave, false);



        vec3 albedo = pow(saturate(stepCloudDensity * frameData.sky.atmosphereConfig.cloudAlbedo * 20.0), vec3(0.5));
        vec3 extinctionCoefficients = vec3(0.719, 0.859, 1.0) * 0.05 * stepCloudDensity;

        ParticipatingMediaContext PMC = setupParticipatingMediaContext(
            albedo, extinctionCoefficients, MsScattFactor, MsExtinFactor, atmosphereTransmittance);

        if(max3(PMC.ScatteringCoefficients[0]) > 0.0f) 
        {
            volumetricShadow(PMC, samplePos, sunDirection, atmosphere, -1);
        }

        if(max3(PMC.ExtinctionCoefficients[0]) > 0.0f) 
        {
            // Add ray march pos, so we can do some average fading or atmosphere sample effect.
            float tpW = min(min(TransmittanceToView.x,TransmittanceToView.y), TransmittanceToView.z);
            rayHitPos += samplePos * tpW;
            rayHitPosWeight += tpW;
        }

        for (int ms = kMsCount - 1; ms >= 0; ms--) // Should terminate at 0
        {
            const vec3 ScatteringCoefficients = PMC.ScatteringCoefficients[ms];
			const vec3 ExtinctionCoefficients = PMC.ExtinctionCoefficients[ms];

            const vec3 TransmittanceToLight = PMC.TransmittanceToLight[ms];
			vec3 SunSkyLuminance = TransmittanceToLight * sunColor * participatingMediaPhase.phase[ms];

            const vec3 ScatteredLuminance = SunSkyLuminance * ScatteringCoefficients + vec3(0.0); // Emissive

            const vec3 SafeExtinctionThreshold = vec3(0.000001f);

            const vec3 SafeExtinctionCoefficients = max(SafeExtinctionThreshold, ExtinctionCoefficients);
			const vec3 SafePathSegmentTransmittance = exp(-SafeExtinctionCoefficients * stepT * 1000.0);

            vec3 LuminanceIntegral = (ScatteredLuminance - ScatteredLuminance * SafePathSegmentTransmittance) / SafeExtinctionCoefficients;
			scatteredLight += TransmittanceToView * LuminanceIntegral;

            if (ms == 0)
            {
                TransmittanceToView *= SafePathSegmentTransmittance;
            }
        }

        if(max3(TransmittanceToView) <= 0.001)
        {
            break;
        }
        sampleT += stepT;
    }

    float transmittance = mean(TransmittanceToView);

    // Apply some additional effect.
    {
        // Get average hit pos.
        rayHitPos /= rayHitPosWeight;

        vec3 rayHitInRender = convertToCameraUnit(rayHitPos - vec3(0.0, atmosphere.bottomRadius, 0.0), frameData);
        vec4 rayInH = frameData.camViewProj * vec4(rayHitInRender, 1.0);
        cloudZ = rayInH.z / rayInH.w;

        rayHitPos -= worldPos;
        float rayHitHeight = length(rayHitPos);

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
        scatteredLight = scatteredLight * (1.0 - airPerspective.a) + airPerspective.rgb * (1.0 - transmittance);


        // Height fog apply.
        {
            float worldDistance = distance(rayHitInRender, frameData.camWorldPos.xyz);

            float fogAmount = 1.0 - exp( -worldDistance * 0.001f * 0.001);
            vec3  fogColor  = vec3(0.5,0.6,0.7);

            // scatteredLight = scatteredLight * fogAmount + fogColor * (1.0 - transmittance);
        }

    }
    // Dual mix transmittance.
    return vec4(scatteredLight, transmittance);
}

#endif