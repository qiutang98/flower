#version 460

// performances.
// ~1-2ms in 4k 3070Ti GPU native render resolution. 2 slice, 4 step.

// I pretty hate noise.
// But if add a spatial filter, some detail like tree's

#extension GL_GOOGLE_include_directive : enable
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_EXT_samplerless_texture_functions : enable

#define COMPUTE_LOCAL_DIFFUSE_Illumination 0
#define COMPUTE_LOCAL_BENT_NORMAL          0

#define SHARED_SAMPLER_SET    1
#define BLUE_NOISE_BUFFER_SET 2

#include "common_shader.glsl"
#include "common_lighting.glsl"

layout (set = 0, binding = 0)  uniform texture2D inHiz;
layout (set = 0, binding = 1)  uniform texture2D inDepth;
layout (set = 0, binding = 2)  uniform texture2D inGbufferS;
layout (set = 0, binding = 3)  uniform texture2D inGbufferB;

layout (set = 0, binding = 4, r8) uniform image2D imageSSAO;

layout (set = 0, binding = 5)  uniform texture2D inHDRSceneColor;
layout (set = 0, binding = 6) uniform UniformFrameData { PerFrameData frameData; };

// Max offset 64 pixel, avoid near view texture sample cache miss.
const float kSSAOMaxPixelScreenRadius = 64.0f; 

float sampleDepth(vec2 uv)
{
    // Heavy texel load task when camera near to some objects.
    return textureLod(sampler2D(inDepth, pointClampEdgeSampler), uv, 0.0).r;
}


float integrateHalfArc(float horizonAngle, float normalAngle)
{
    return (cos(normalAngle) + 2.f * horizonAngle * sin(normalAngle) - cos(2.f * horizonAngle - normalAngle)) / 4.f;
}

vec3 integrateBentNormal(float horizonAngle0, float horizonAngle1, float normalAngle, vec3 worldEyeDir, vec3 sliceViewDir)
{
    float t0 = (6.0f * sin(horizonAngle0 - normalAngle) - sin(3.f * horizonAngle0 - normalAngle) +
                6.0f * sin(horizonAngle1 - normalAngle) - sin(3.f * horizonAngle1 - normalAngle) +
                16.f * sin(normalAngle) - 3.f * (sin(horizonAngle0 + normalAngle) + sin(horizonAngle1 + normalAngle))) / 12.f;
    float t1 = (-cos(3.f * horizonAngle0 - normalAngle)
                -cos(3.f * horizonAngle1 - normalAngle) +
                 8.f * cos(normalAngle) - 3.f * (cos(horizonAngle0 + normalAngle) + cos(horizonAngle1 + normalAngle))) / 12.f;

    vec3 viewBentNormal  = vec3(sliceViewDir.x * t0, sliceViewDir.y * t0, t1);

    vec3 worldBentNormal = normalize((frameData.camInvertView * vec4(viewBentNormal, 0.f)).xyz);
    return rotFromToMatrix(-frameData.camForward.xyz, worldEyeDir) * worldBentNormal;
}

layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    ivec2 workSize = imageSize(imageSSAO);

    uvec2 groupThreadId = remap8x8(gl_LocalInvocationIndex);
    uvec2 dispatchId = groupThreadId + gl_WorkGroupID.xy * 8;
    ivec2 workPos = ivec2(dispatchId);

    if(workPos.x >= workSize.x || workPos.y >= workSize.y)
    {
        return;
    }

    const vec2 texelSize = 1.0f / vec2(workSize);
    const vec2 uv = (vec2(workPos) + vec2(0.5f)) * texelSize;

    const vec4 inGbufferBValue = texelFetch(inGbufferB, workPos, 0);

    const float depth = texture(sampler2D(inDepth, pointClampEdgeSampler), uv).r;
    const vec3 worldNormal = unpackWorldNormal(inGbufferBValue.rgb);
    if(depth <= 0.0)
    {
        imageStore(imageSSAO, ivec2(workPos), vec4(1.0));
        return;
    }

    const vec4 inGbufferSValue = texelFetch(inGbufferS, workPos, 0);


    const float meshAo = inGbufferSValue.b;




    const float linearDepth = linearizeDepth(depth, frameData);

    vec3 worldPos = getWorldPos(uv, depth, frameData);
    vec3 worldEyeDir = normalize(frameData.camWorldPos.xyz - worldPos);

    vec2 noise; 
    {
        uvec2 offset = uvec2(vec2(0.754877669, 0.569840296) * (frameData.frameIndex.x) * uvec2(workSize));
        uvec2 offsetId = uvec2(workPos) + offset;
        offsetId.x = offsetId.x % workSize.x;
        offsetId.y = offsetId.y % workSize.y;
        noise.x = samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d(offsetId.x, offsetId.y, 0, 0u);
        noise.y = samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d(offsetId.x, offsetId.y, 0, 1u);
    }

    float  sliceUvRadius = frameData.postprocessing.ssao_uvRadius / linearDepth;
    const float maxDu = max(texelSize.x, texelSize.y);

    // NOTE: When camera close to mesh, sliceUvRadius will increase and make tons of texture cache miss.
    //       We need to clamp it.
    sliceUvRadius = min(sliceUvRadius, maxDu * kSSAOMaxPixelScreenRadius); // Max pixel search radius is 256 pixel, avoid large search step when view is close.
    int ssaoStepCount = frameData.postprocessing.ssao_stepCount;
    const float stepCountInverse = 1.0 / ssaoStepCount;
    const float sliceCountInverse = 1.0 / frameData.postprocessing.ssao_sliceCount;

    float ambientOcclusion = 0.f;
    vec3 bentNormal = vec3(0.f);

    vec3 localDiffuseIllumination = vec3(0.f);

    for(int sliceIndex = 0; sliceIndex < frameData.postprocessing.ssao_sliceCount; sliceIndex++)
    {
        float  sliceAngle  = ((sliceIndex + noise.x) * sliceCountInverse) * kPI;
        float  sliceCos    = cos(sliceAngle);
        float  sliceSin    = sin(sliceAngle);
        vec2 sliceUvDir = vec2(sliceCos, -sliceSin) * sliceUvRadius;

        vec3 sliceViewDir  = vec3(sliceCos, sliceSin, 0.f);
        vec3 sliceWorldDir = normalize((frameData.camInvertView * vec4(sliceViewDir, 0.f)).xyz);

        vec3 orthoWorldDir = sliceWorldDir - dot(sliceWorldDir, worldEyeDir) * worldEyeDir;
        vec3 projAxisDir   = normalize(cross(orthoWorldDir, worldEyeDir));
        vec3 projWorldNormal       = worldNormal - dot(worldNormal, projAxisDir) * projAxisDir;

        float  projWorldNormalLen   = length(projWorldNormal);
        float  projWorldNormalCos   = saturate(dot(projWorldNormal, worldEyeDir) / projWorldNormalLen);
        float  projWorldNormalAngle = sign(dot(orthoWorldDir, projWorldNormal)) * acos(projWorldNormalCos);

        float  sideSigns[2] = { 1.f, -1.f};
        float  horizonAngles[2];

        for (int sideIndex = 0; sideIndex < 2; ++sideIndex)
        {
            float  horizon_min = cos(projWorldNormalAngle + sideSigns[sideIndex] * kPI * 0.5);
            float  horizonCos = horizon_min;
            vec3 prevSampleWorldPos = vec3(0.f, 0.f, 0.f);

            for(int stepIndex = 0; stepIndex < ssaoStepCount; stepIndex++)
            {
                float  sampleStep = stepIndex * stepCountInverse;
                // Square distribution.
                sampleStep *= sampleStep;

                // Noise need still keep same pattern avoid destroy low bias feature.
                sampleStep += (noise.y + 1e-5f) * stepCountInverse;

                vec2 sampleUvOffset = sampleStep * sliceUvDir;

                vec2 sampleUv = uv + sideSigns[sideIndex] * sampleUvOffset;


                float sampleDepth = sampleDepth(sampleUv);
                vec3 sampleWorldPos = getWorldPos(sampleUv, sampleDepth, frameData);

                vec3 horizonWorldDir = sampleWorldPos - worldPos;
                float horizonWorldLen = length(horizonWorldDir); 
                horizonWorldDir /= horizonWorldLen;

                float sampleWeight = saturate(horizonWorldLen * frameData.postprocessing.ssao_falloffMul + frameData.postprocessing.ssao_falloffAdd);

                float sampleCos = mix(horizon_min, dot(horizonWorldDir, worldEyeDir), sampleWeight);
                float  prevHorizonCos  = horizonCos;
                horizonCos = max(horizonCos, sampleCos);

                // Performance heavy here.
                if (sampleCos >= prevHorizonCos)
                {
                    vec3 sampleWorldNormal = unpackWorldNormal(texture(sampler2D(inGbufferB, pointClampEdgeSampler), sampleUv).rgb);

                    if (stepIndex > 0)
                    {
                        vec3 closestWorldPos = prevSampleWorldPos * min(
                            intersectDirPlaneOneSided(prevSampleWorldPos, sampleWorldNormal, sampleWorldPos),
                            intersectDirPlaneOneSided(prevSampleWorldPos, worldNormal, worldPos)
                        );

                        prevHorizonCos = clamp(dot(normalize(closestWorldPos - worldPos), worldEyeDir), prevHorizonCos, horizonCos);
                    }

                    float horizonAngle0 = projWorldNormalAngle + max(sideSigns[sideIndex] * acos(prevHorizonCos) - projWorldNormalAngle, -kPI * 0.5);
                    float horizonAngle1 = projWorldNormalAngle + min(sideSigns[sideIndex] * acos(horizonCos)      - projWorldNormalAngle, +kPI * 0.5);
                    float sampleOcclusion = integrateHalfArc(horizonAngle0, projWorldNormalAngle) - integrateHalfArc(horizonAngle1, projWorldNormalAngle);

#if COMPUTE_LOCAL_DIFFUSE_Illumination
                    // A lot of cost. current don't use ssdo.
                    // From h3r2tic's demo
                    // https://github.com/h3r2tic/rtoy-samples/blob/main/assets/shaders/ssgi/ssgi.glsl
                    vec3 sampleLighting = texture(sampler2D(inHDRSceneColor, pointClampEdgeSampler), sampleUv).xyz;
                    localDiffuseIllumination += sampleLighting * sampleOcclusion * step(0, dot(-horizonWorldDir, sampleWorldNormal));
#endif
                }

                prevSampleWorldPos = sampleWorldPos;
            }

            // Ambient Occlusion
            horizonAngles[sideIndex] = sideSigns[sideIndex] * acos(horizonCos);
            ambientOcclusion += projWorldNormalLen * integrateHalfArc(horizonAngles[sideIndex], projWorldNormalAngle);

#if COMPUTE_LOCAL_DIFFUSE_Illumination
            // Global Lighting
            localDiffuseIllumination *= projWorldNormalLen;
#endif
        }

#if COMPUTE_LOCAL_BENT_NORMAL
        // Bent normal
        bentNormal += projWorldNormalLen * integrateBentNormal(
            horizonAngles[0], horizonAngles[1], projWorldNormalAngle, worldEyeDir, sliceViewDir);
#endif
    }

    ambientOcclusion *= sliceCountInverse;

    if(frameData.renderWidth == frameData.postWidth)
    {
        ambientOcclusion = 1.0 - (1.0 - pow(ambientOcclusion, frameData.postprocessing.ssao_power)) * frameData.postprocessing.ssao_intensity;
    }


#if COMPUTE_LOCAL_BENT_NORMAL
    // ..
    bentNormal = packWorldNormal(normalize(bentNormal));
#else 
    bentNormal = inGbufferBValue.rgb;
#endif

#if COMPUTE_LOCAL_DIFFUSE_Illumination
    localDiffuseIllumination *= sliceCountInverse;
    localDiffuseIllumination = max(vec3(0.0), localDiffuseIllumination);
#endif

    imageStore(imageSSAO, ivec2(workPos), vec4(ambientOcclusion));
}