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

// Max sample count per distance. 15 tap/km
#define kCloudDistanceToSampleMaxCount (1.0 / 15.0)

#define kShadowLightStepNum 6
#define kShadowLightStepBasicLen 0.167
#define kShadowLightStepMul 1.1

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

    float basicNoise = texture(sampler3D(inBasicNoise, linearRepeatSampler), (posKm + windOffset) * vec3(0.1)).r;
    float basicCloudNoise = gradienShape * basicNoise;

	float basicCloudWithCoverage = coverage * remap(basicCloudNoise, 1.0 - coverage, 1, 0, 1);

    vec3 sampleDetailNoise = posKm - windOffset * 0.15 + vec3(basicNoise.x, 0.0, basicCloudNoise) * normalizeHeight;
    float detailNoiseComposite = texture(sampler3D(inWorleyNoise, linearRepeatSampler), sampleDetailNoise * 0.2).r;
	float detailNoiseMixByHeight = 0.2 * mix(detailNoiseComposite, 1 - detailNoiseComposite, saturate(normalizeHeight * 10.0));
    
    float densityShape = saturate(0.01 + normalizeHeight * 1.15) * kDensity *
        remap(normalizeHeight, 0.0, 0.1, 0.0, 1.0) * 
        remap(normalizeHeight, 0.8, 1.0, 1.0, 0.0);

    float cloudDensity = remap(basicCloudWithCoverage, detailNoiseMixByHeight, 1.0, 0.0, 1.0);
	return cloudDensity * densityShape;
 
}

// Cloud shape end.
////////////////////////////////////////////////////////////////////////////////////////

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

float volumetricShadow(vec3 posKm, float cosTheta, vec3 sunDirection, in const AtmosphereParameters atmosphere)
{
    const float kStepLMul = kShadowLightStepMul;
    const int kStepLight = kShadowLightStepNum;
    float stepL = kShadowLightStepBasicLen; // km
    
    float d = stepL * 0.5;

	// Collect total density along light ray.
    float intensitySum = 0.0;
	for(int j = 0; j < kStepLight; j++)
    {
        vec3 samplePosKm = posKm + sunDirection * d; // km

        float sampleHeightKm = length(samplePosKm);
        float sampleDt = sampleHeightKm - atmosphere.cloudAreaStartHeight;

        // Start pos always inside cloud area, if out of bounds, it will never inside bounds again.
        if(sampleDt > atmosphere.cloudAreaThickness || sampleDt < 0)
        {
            break;
        }

        float normalizeHeight = sampleDt / atmosphere.cloudAreaThickness;
        vec3 samplePosMeter = samplePosKm * 1000.0f;

        float intensity = cloudMap(samplePosMeter, normalizeHeight);
        intensitySum += intensity * stepL;

        d += stepL;
        stepL *= kStepLMul;
	}

    // To meter.
    float intensityMeter = intensitySum * 1000.0;

    float beersLambert = exp(-intensityMeter);

	return beersLambert; // * mix(1.0, powder, smoothstep(0.5, -0.5, cosTheta));

    // float upperMask = saturate(1.0 - intensityMeter);
}

vec4 cloudColorCompute(vec2 uv, float blueNoise, inout float cloudZ, ivec2 workPos, vec3 worldDir)
{
    AtmosphereParameters atmosphere = getAtmosphereParameters(frameData);

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
    float phase = multiPhase(cosTheta);

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
        #if 0
            float stepTransmittance = exp(-opticalDepth); 
        #else
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
                powderEffectTerm = pow(saturate(opticalDepth * 20.0), 0.5);
            #endif
            }

        #if 0
            vec3 ambientLight = skyBackgroundColor;
        #else
            vec3 ambientLight = mix(cloudBottomSkyLight, cloudTopSkyLight, normalizeHeight);
        #endif

            vec3 lightTerm;
            {
                // Amount of sunlight that reaches the sample point through the cloud 
                // is the combination of ambient light and attenuated direct light.
                vec3 sunLit = frameData.earthAtmosphere.cloudShadingSunLightScale * sunColor; 

                lightTerm = ambientLight + sunLit;
            }

            float visibilityTerm = volumetricShadow(samplePos, cosTheta, sunDirection, atmosphere);

            float sigmaS = stepCloudDensity;
            float sigmaE = sigmaS + 1e-4f;
            
            vec3 sactterLitStep = (ambientLight + visibilityTerm * lightTerm * phase * powderEffectTerm) * sigmaS;

            // See slide 28 at http://www.frostbite.com/2015/08/physically-based-unified-volumetric-rendering-in-frostbite/
            scatteredLight += atmosphereTransmittance * transmittance * (sactterLitStep - sactterLitStep * stepTransmittance) / sigmaE;

            // Beer's law.
            transmittance *= stepTransmittance;
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