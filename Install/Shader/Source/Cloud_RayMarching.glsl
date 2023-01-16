#version 460

/*
** Physical based render code, develop by engineer: qiutanguu.
*/

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable
#include "Cloud_Common.glsl"
#include "Noise.glsl"
#include "Phase.glsl"
//////////////// Paramters ///////////////////////////

// Max tracing distance, in km unit.
#define kTracingMaxDistance 50.0f

// Max start ray tracing distance. in km unit, use this avoid eye look from space very far still need tracing.
#define kTracingStartMaxDistance 350.0f

// Min max sample count define.
#define kSampleCountMin 2
#define kSampleCountMax 96

// Max sample count per distance. 15 tap/km
#define kCloudDistanceToSampleMaxCount (1.0 / 15.0)

#define kFogFade 0.005

///////////////////////////////////////

vec4 cloudColorCompute(vec2 uv, float blueNoise, inout float cloudZ)
{
    AtmosphereParameters atmosphere = getAtmosphereParameters(frameData);

    // We are revert z.
    vec4 clipSpace = vec4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, 0.0, 1.0);
    vec4 viewPosH = viewData.camInvertProj * clipSpace;
    vec3 viewSpaceDir = viewPosH.xyz / viewPosH.w;
    vec3 worldDir = normalize((viewData.camInvertView * vec4(viewSpaceDir, 0.0)).xyz);

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
    if(tMax <= tMin || tMin > kTracingStartMaxDistance)
    {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }

    // Clamp marching distance by setting.    
    const float marchingDistance = min(kTracingMaxDistance, tMax - tMin);
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
    float phase = dualLobPhase(0.5, -0.5, 0.2, cosTheta);

    vec3 transmittance  = vec3(1.0, 1.0, 1.0);
    vec3 scatteredLight = vec3(0.0, 0.0, 0.0);

    // Average ray hit pos to evaluate air perspective and height fog.
    vec3 rayHitPos = vec3(0.0);
    float rayHitPosWeight = 0.0;

    const float minTransmittance = 1e-3f;
    vec2 coverageWindOffset = getWeatherOffset();

    for(uint i = 0; i < stepCountUnit; i ++)
    {
        // World space sample pos, in km unit.
        vec3 samplePosKm = sampleT * worldDir + worldPos;

        float sampleHeightKm = length(samplePosKm);

        // Get sample normalize height [0.0, 1.0]
        float normalizeHeight = (sampleHeightKm - atmosphere.cloudAreaStartHeight)  / atmosphere.cloudAreaThickness;

        vec3 weatherData = sampleCloudWeatherMap(samplePosKm, normalizeHeight, coverageWindOffset);
        float cloudDensity = cloudMap(samplePosKm, normalizeHeight, weatherData);

        // Lighting.
        if(cloudDensity > 0.0) 
        {
            // Add ray march pos, so we can do some average fading or atmosphere sample effect.
            float meanT = mean(transmittance);
            rayHitPos += samplePosKm * meanT;
            rayHitPosWeight += meanT;

            float opticalDepth = cloudDensity * stepT * 1000.0;

        #if 0
            // Siggraph 2017's new powder formula.
            float depthProbability = 0.05 + pow(opticalDepth, remap(normalizeHeight, 0.3, 0.85, 0.5, 2.0)); // May cause nan when density is near 1.
            float verticalProbability = pow(max(1e-5, remap(normalizeHeight, 0.07, 0.14, 0.1, 1.0)), 0.8);
            float powderEffect = depthProbability * verticalProbability;
            vec3 albedo = kAlbedo * powderEffect;
        #elif 1
            // Unreal engine powder.
            vec3 albedo = pow(saturate(kAlbedo * cloudDensity * kBeerPowder), vec3(kBeerPowderPower)); 
        #else
            vec3 albedo = kAlbedo;
        #endif
	           
            vec3 extinction = kExtinctionCoefficient * cloudDensity;

            // Second evaluate transmittance due to participating media
            vec3 atmosphereTransmittance;
            {
                const vec3 upVector = samplePosKm / sampleHeightKm;
                float viewZenithCosAngle = dot(sunDirection, upVector);
                vec2 sampleUv;
                lutTransmittanceParamsToUv(atmosphere, viewHeight, viewZenithCosAngle, sampleUv);
                atmosphereTransmittance = texture(sampler2D(inTransmittanceLut, linearClampEdgeSampler), sampleUv).rgb;
            }

            // Sample participating media with multiple scattering
            ParticipatingMedia participatingMedia = getParticipatingMedia(
                albedo, 
                extinction, 
                kMultiScatteringScattering, 
                kMultiScatteringExtinction, 
                atmosphereTransmittance
            );

            // TODO: Sample skylight SH.
            vec3 ambientLight = 0.1 * frameData.directionalLight.color * mix(vec3(0.23, 0.39, 0.51), vec3(0.87, 0.98, 1.18), normalizeHeight);

            // Calculate bounced light from ground onto clouds
        #if 1
            const float maxTransmittanceToView = max(max(transmittance.x, transmittance.y), transmittance.z);
            if (maxTransmittanceToView > 0.01f)
            {
                ambientLight += getVolumetricGroundContribution(
                    atmosphere, 
                    samplePosKm, 
                    sunDirection, 
                    sunColor, 
                    atmosphereTransmittance,
                    normalizeHeight
                );
            }
        #endif

            // Calcualte volumetric shadow
	        getVolumetricShadow(participatingMedia, atmosphere, samplePosKm, sunDirection);

            // float backScatterPhase = applyBackScattering(phase, mean(extinction));
            // ParticipatingMediaPhase participatingMediaPhase = getParticipatingMediaPhase(backScatterPhase, kMultiScatteringEccentricity);
               ParticipatingMediaPhase participatingMediaPhase = getParticipatingMediaPhase(phase, kMultiScatteringEccentricity);

            // Analytical scattering integration based on multiple scattering
            for (int ms = kMsCount - 1; ms >= 0; ms--) // Should terminate at 0
            {
                const vec3 scatteringCoefficients = participatingMedia.scatteringCoefficients[ms];
                const vec3 extinctionCoefficients = participatingMedia.extinctionCoefficients[ms];
                const vec3 transmittanceToLight   = participatingMedia.transmittanceToLight[ms];
                
                vec3 sunSkyLuminance = transmittanceToLight * sunColor * participatingMediaPhase.phase[ms];
                sunSkyLuminance += (ms == 0 ? ambientLight : vec3(0.0, 0.0, 0.0)); // only apply at last
                
                const vec3 scatteredLuminance = (sunSkyLuminance * scatteringCoefficients) * weatherDensity(weatherData); // + emission. Light can be emitted when media reach high heat. Could be used to make lightning

                // See slide 28 at http://www.frostbite.com/2015/08/physically-based-unified-volumetric-rendering-in-frostbite/ 
                const vec3 clampedExtinctionCoefficients = max(extinctionCoefficients, 0.0000001);
                const vec3 sampleTransmittance = exp(-clampedExtinctionCoefficients * stepT * 1000.0); // to meter.
                vec3 luminanceIntegral = (scatteredLuminance - scatteredLuminance * sampleTransmittance) / clampedExtinctionCoefficients; // integrate along the current step segment

                scatteredLight += transmittance * luminanceIntegral; // accumulate and also take into account the transmittance from previous steps
                if (ms == 0)
                {
                    transmittance *= sampleTransmittance;
                }
            }

            if(mean(transmittance) < minTransmittance)
            {
                break;
            }   
        }

        sampleT += stepT;
    }

    vec3 srcColor = texture(sampler2D(inHdrSceneColor, linearClampEdgeSampler), uv).rgb;

    // Apply cloud transmittance.
    vec3 finalColor = srcColor;

    float approxTransmittance = mean(transmittance);

    // Apply some additional effect.
    if(approxTransmittance < 1.0 - minTransmittance)
    {
        // Get average hit pos.
        rayHitPos /= rayHitPosWeight;

        vec3 rayHitInRender = convertToCameraUnit(rayHitPos - vec3(0.0, atmosphere.bottomRadius, 0.0), viewData);
        vec4 rayInH = viewData.camViewProj * vec4(rayHitInRender, 1.0);
        cloudZ = rayInH.z / rayInH.w;

        rayHitPos -= worldPos;
        float rayHitHeight = length(rayHitPos);

        // Fade effect apply.
        float fading = exp(-rayHitHeight * kFogFade);
        float cloudTransmittanceFaded = mix(1.0, approxTransmittance, fading);

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
        approxTransmittance = cloudTransmittanceFaded;
    }
    else
    {
        cloudZ = 0.0;
    }

    // Dual mix transmittance.
    return vec4(finalColor, approxTransmittance);
}

// Evaluate quater resolution.
layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    ivec2 texSize = imageSize(imageCloudRenderTexture);
    ivec2 workPos = ivec2(gl_GlobalInvocationID.xy);

    if(workPos.x >= texSize.x || workPos.y >= texSize.y)
    {
        return;
    }

    // TODO: Tile optimize or ray classify, pre-return if behind opaque objects.

    // Get bayer offset matrix.
    uint bayerIndex = frameData.frameIndex.x % 16;
    ivec2 bayerOffset = ivec2(bayerFilter4x4[bayerIndex] % 4, bayerFilter4x4[bayerIndex] / 4);

    // Get evaluate position in full resolution.
    ivec2 fullResSize = texSize * 4;
    ivec2 fullResWorkPos = workPos * 4 + ivec2(bayerOffset);

    // Get evaluate uv in full resolution.
    const vec2 uv = (vec2(fullResWorkPos) + vec2(0.5f)) / vec2(fullResSize);

    // Offset retarget for new seeds each frame
    uvec2 offset = uvec2(vec2(0.754877669, 0.569840296) * frameData.frameIndex.x * uvec2(fullResSize));
    uvec2 offsetId = fullResWorkPos.xy + offset;
    offsetId.x = offsetId.x % fullResSize.x;
    offsetId.y = offsetId.y % fullResSize.y;
    float blueNoise = samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d(offsetId.x, offsetId.y, 0, 0u); 

    float depth = 0.0; // reverse z.
    vec4 cloudColor = cloudColorCompute(uv, blueNoise, depth);

	imageStore(imageCloudRenderTexture, workPos, cloudColor);
    imageStore(imageCloudDepthTexture, workPos, vec4(depth));
}