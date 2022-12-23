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

#define kShadowLightStepNum 6
#define kShadowLightStepBasicLen 0.167
#define kShadowLightStepMul 1.1

#define kFogFade 0.005
#define kSunLightScale kPI

/////////////////////////////////////////////////////

float multiScatter(float VoL)
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

float lightRay(vec3 posKm, float cosTheta, vec3 sunDirection, in const AtmosphereParameters atmosphere)
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
    float phase = multiScatter(cosTheta);

    float transmittance  = 1.0;
    vec3 scatteredLight = vec3(0.0, 0.0, 0.0);

    // Average ray hit pos to evaluate air perspective and height fog.
    vec3 rayHitPos = vec3(0.0);
    float rayHitPosWeight = 0.0;

    for(uint i = 0; i < stepCountUnit; i ++)
    {
        // World space sample pos, in km unit.
        vec3 samplePos = sampleT * worldDir + worldPos;

        float sampleHeight = length(samplePos);

        // Get sample normalize height [0.0, 1.0]
        float normalizeHeight = (sampleHeight - atmosphere.cloudAreaStartHeight)  / atmosphere.cloudAreaThickness;

        // Convert to meter.
        vec3 samplePosMeter = samplePos * 1000.0f;
        float alpha = cloudMap(samplePosMeter, normalizeHeight);

        // Add ray march pos, so we can do some average fading or atmosphere sample effect.
        rayHitPos += samplePos * transmittance;
        rayHitPosWeight += transmittance;

        if(alpha > 0.) 
        {
            float opticalDepth = alpha * stepT * 1000.0;

            // beer's lambert.
            float stepTransmittance = max(exp(-opticalDepth), exp(-opticalDepth * 0.25) * 0.7); // exp(-opticalDepth); // to meter unit.

            // Second evaluate transmittance due to participating media
            vec3 atmosphereTransmittance;
            {
                const vec3 upVector = samplePos / sampleHeight;
                float viewZenithCosAngle = dot(sunDirection, upVector);
                vec2 sampleUv;
                lutTransmittanceParamsToUv(atmosphere, viewHeight, viewZenithCosAngle, sampleUv);
                atmosphereTransmittance = texture(sampler2D(inTransmittanceLut, linearClampEdgeSampler), sampleUv).rgb;
            }

            vec3 ambientLight = frameData.directionalLight.color * mix(vec3(0.23, 0.39, 0.51), vec3(0.87, 0.98, 1.18), normalizeHeight * normalizeHeight);

#if 1
            // Siggraph 2017's new powder formula.
            float depthProbability = 0.05 + pow(opticalDepth, remap(normalizeHeight, 0.3, 0.85, 0.5, 2.0));
            float verticalProbability = pow(remap(normalizeHeight, 0.07, 0.14, 0.1, 1.0), 0.8);
            float powderEffect = depthProbability * verticalProbability; //powder(opticalDepth * 2.0);
#else
            // Unreal engine 5's implement powder formula.
            float powderEffect = pow(saturate(opticalDepth * 20.0), 0.5);
#endif

            // Amount of sunlight that reaches the sample point through the cloud 
            // is the combination of ambient light and attenuated direct light.
            vec3 sunLit = kSunLightScale * sunColor * lightRay(samplePos, cosTheta, sunDirection, atmosphere);

            vec3 luminance = ambientLight + sunLit * phase * powderEffect; // * alpha;
            
            luminance *= atmosphereTransmittance;
            // luminance += atmosphereTransmittance *atmosphereTransmittance;

            scatteredLight += transmittance * (luminance - luminance * stepTransmittance); // / alpha
            transmittance *= stepTransmittance;
        }

        if(transmittance <= 0.001)
        {
            break;
        }

        sampleT += stepT;
    }

    vec3 srcColor = texture(sampler2D(inHdrSceneColor, linearClampEdgeSampler), uv).rgb;

    // Apply cloud transmittance.
    vec3 finalColor = srcColor;



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
        float fading = exp(-rayHitHeight * kFogFade);
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