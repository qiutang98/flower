#version 460

#extension GL_GOOGLE_include_directive : enable

#include "VolumetricCloudCommon.glsl"

const float kTracingMaxDistance = 50.0f; // TODO: Configable.
const float kTracingStartMaxDistance = 350.0f; // TODO: Configable.
const uint kSampleCountMin = 2;
const uint kSampleCountMax = 96;
const float kCloudDistanceToSampleMaxCount = 15.0f;
const float kCloudDistanceToSampleMaxCountInv = 1.0f / kCloudDistanceToSampleMaxCount;

const float kErosionNoiseScale = 2.0f;
const float kBasicNoiseScale = 0.015f;
const float kBasicNoiseExp = 10.0f;
const float kErosionScale = 0.1f;
const float kErosionExp = 6.0f;
const float kErosionStrength = 0.08;
const float kNoiseHeightRange = 1.5f;
const float kNoiseHeightExp = 2.0f;

// #define CLOUD_SELF_SHADOW_STEPS 6
#define CLOUD_SELF_SHADOW_STEPS 20
#define CLOUDS_SHADOW_MARGE_STEP_SIZE (50)
#define CLOUDS_SHADOW_MARGE_STEP_MULTIPLY (1.3)

const float kBeersScale = 1000.0;
const float kBeersScaleShadow = 1000.0;

float getCloudCoverage(vec3 pos, float normalizeHeight)
{
    const vec3 windOffset = vec3(0) ;//(frameData.appTime.x) * vec3(1.0);

    // Offset 5km.
    vec2 sampleUv = (pos.xz + vec2(5.0, 5.0) + windOffset.xy) * 0.3 * 0.03 * 1.0;
    vec4 weatherValue = texture(sampler2D(inWeatherTexture, linearRepeatSampler), sampleUv);

    vec2 gradientUv = vec2(pow(weatherValue.g, 0.5f), clamp(1.0 - normalizeHeight, 0.01f, 0.99f));
    vec4 gradientValue = texture(sampler2D(inGradientTexture, linearClampEdgeSampler), gradientUv);

    float coverage = weatherValue.r * gradientValue.r;

    return coverage;
}

float getCloudBasicNoise(vec3 pos, float normalizeHeight)
{
    const vec3 windOffset = vec3(0) ;
    const vec3 sampleBasicUvz = pos * vec3(kErosionNoiseScale);

    ///////////
    float basicNoise;
    {
        vec3 sampleUvz = sampleBasicUvz + windOffset;
        sampleUvz *= kBasicNoiseScale;

        basicNoise = texture(sampler3D(inWorleyNoise, linearRepeatSampler), sampleUvz).g;

        basicNoise =  saturate(pow(basicNoise, kBasicNoiseExp));
    }
    
    //////////////////////////////////////////
    float detailNoise;
    {
        vec3 sampleUvz = sampleBasicUvz - windOffset; 
        sampleUvz = sampleUvz * kErosionScale;

        detailNoise = texture(sampler3D(inWorleyNoise, linearRepeatSampler), sampleUvz).a;
        detailNoise = pow(detailNoise, kErosionExp) * kErosionStrength;

        detailNoise = clamp(detailNoise, 0.0, 1.0);


        detailNoise = remap(basicNoise, detailNoise, 1.0, 0.0, 1.0);
    }

    

    float resultNoise = mix(basicNoise, detailNoise, pow(saturate(normalizeHeight * kNoiseHeightRange), kNoiseHeightExp));


    return resultNoise; // 4.
}


float getCloudDensity(vec3 pos, float normalizeHeight)
{
    float coverage = getCloudCoverage(pos, normalizeHeight);

    return coverage * getCloudBasicNoise(pos, normalizeHeight);
}



float volumetricShadow(in vec3 from, in float sundotrd, in AtmosphereParameters atmosphere) 
{
    float dd = CLOUDS_SHADOW_MARGE_STEP_SIZE * 0.001f; // km
    vec3 rd = normalize(frameData.directionalLight.direction);
    float d = dd * .5;
    float shadow = 1.0;

    for(int s=0; s < CLOUD_SELF_SHADOW_STEPS; s++) 
    {
        vec3 pos = from + rd * d; // km
        float norY = (length(pos) - atmosphere.cloudAreaStartHeight) / atmosphere.cloudAreaThickness;

        if(norY > 1.)
        {
            return shadow;
        } 

        float muE = getCloudDensity(pos, norY);

        shadow *= exp(-muE * dd * kBeersScaleShadow);

        // dd *= CLOUDS_SHADOW_MARGE_STEP_MULTIPLY;
        d += dd;
    }
    return shadow;
}

vec4 cloudColorCompute(vec2 uv, ivec2 workPos, ivec2 texSize)
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

    float viewHeight = length(worldPos);

    // 
    float tMin;
    float tMax;
    if(viewHeight < radiusCloudStart)
    {
        // Eye under cloud area.
        float shadingModelId = texture(sampler2D(inGBufferA, pointClampEdgeSampler), uv).a;
        if(isShadingModelValid(shadingModelId))
        {
            // Intersect with earth, pre-return.
            return vec4(0.0, 0.0, 0.0, 1.0);
        }

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

    if(tMax <= tMin || tMin > kTracingStartMaxDistance)
    {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }

    // Clamp marching distance by setting.    
    const float marchingDistance = min(kTracingMaxDistance, tMax - tMin);
	tMax = tMin + marchingDistance;

    const uint stepCountUnit =  uint(max(kSampleCountMin, kSampleCountMax * saturate((tMax - tMin) * kCloudDistanceToSampleMaxCountInv)));
    const float stepCount = float(stepCountUnit);
    const float stepT = (tMax - tMin) / stepCount; // Per step lenght.

    // Cloud transmittance color.
    float transmittance = 1.0;
    vec3 scatteredLight = vec3(0.0, 0.0, 0.0);

    #define CLOUDS_FORWARD_SCATTERING_G (.8)
    #define CLOUDS_BACKWARD_SCATTERING_G (-.2)
    #define CLOUDS_SCATTERING_LERP (.5)

    float sundotrd = dot(worldDir, -normalize(frameData.directionalLight.direction));
    vec3 sunColor = frameData.directionalLight.color * frameData.directionalLight.intensity;

    float scattering =  mix(
        henyeyGreenstein(sundotrd, CLOUDS_FORWARD_SCATTERING_G),
        henyeyGreenstein(sundotrd, CLOUDS_BACKWARD_SCATTERING_G), CLOUDS_SCATTERING_LERP);

    for(uint i = 0; i < stepCountUnit; i ++)
    {
        vec3 samplePos = (i * stepT + tMin) * worldDir + worldPos;
        float normalizeHeight = clamp((length(samplePos) - atmosphere.cloudAreaStartHeight)  / atmosphere.cloudAreaThickness, 0., 1.);

        float cloudDensity = getCloudDensity(samplePos, normalizeHeight);
        if(cloudDensity > 0.) 
        {
            // Beers' law. unit is meters.
            float dTrans = exp(-cloudDensity * stepT * kBeersScale);

            vec3 curL = vec3(volumetricShadow(samplePos, sundotrd, atmosphere));

            float cloudDensityClamp = max(cloudDensity, 1e-6f);
            vec3 curS = (curL - curL * dTrans) / cloudDensityClamp;

            scatteredLight += transmittance * curS * cloudDensity * scattering; 
            transmittance *= dTrans;
        }

        if(transmittance <= 0.001) 
        {
            break;
        }
    }

    return vec4(scatteredLight, transmittance);
}

layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    ivec2 texSize = imageSize(imageCloudRenderTexture);
    ivec2 workPos = ivec2(gl_GlobalInvocationID.xy);

    if(workPos.x >= texSize.x || workPos.y >= texSize.y)
    {
        return;
    }

    const vec2 uv = (vec2(workPos) + vec2(0.5f)) / vec2(texSize);
    vec4 cloudColor = cloudColorCompute(uv, workPos, texSize);

	imageStore(imageCloudRenderTexture, workPos, cloudColor);
}