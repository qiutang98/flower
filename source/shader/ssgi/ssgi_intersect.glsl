#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_EXT_samplerless_texture_functions : enable

#include "../common/shared_functions.glsl"
#include "../common/shared_struct.glsl"
#include "../common/shared_lighting.glsl"

layout (set = 0, binding = 0)  uniform texture2D inHiz;
layout (set = 0, binding = 1)  uniform texture2D inDepth;
layout (set = 0, binding = 2)  uniform texture2D inGbufferA;
layout (set = 0, binding = 3)  uniform texture2D inGbufferB;
layout (set = 0, binding = 4, rgba8) uniform image2D ssaoBentNormal;
layout (set = 0, binding = 5)  uniform texture2D inHDRSceneColor;
layout (set = 0, binding = 6) uniform UniformFrameData { PerFrameData frameData; };

#define SHARED_SAMPLER_SET 1
#include "../common/shared_sampler.glsl"

#define BLUE_NOISE_BUFFER_SET 2
#include "../common/shared_bluenoise.glsl"

layout(push_constant) uniform PushConsts
{   
    float uvRadius;
    uint sliceCount;
    float falloffMul;
    float falloffAdd;
    uint stepCount;
    float intensity;
    float power;
} SSGIPush;

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
    ivec2 workSize = imageSize(ssaoBentNormal);

    uvec2 groupThreadId = remap8x8(gl_LocalInvocationIndex);
    uvec2 dispatchId = groupThreadId + gl_WorkGroupID.xy * 8;
    ivec2 workPos = ivec2(dispatchId);

    if(workPos.x >= workSize.x || workPos.y >= workSize.y)
    {
        return;
    }

    const vec2 texelSize = 1.0f / vec2(workSize);
    const vec2 uv = (vec2(workPos) + vec2(0.5f)) * texelSize;

    float shadingModelId = texture(sampler2D(inGbufferA, pointClampEdgeSampler), uv).a;
    if(!isShadingModelValid(shadingModelId))
    {
        imageStore(ssaoBentNormal, ivec2(workPos), vec4(0.0f));
        // imageStore(ssaoLitResult, ivec2(workPos), vec4(0.0f));
        // imageStore(ssaoAo, ivec2(workPos), vec4(0.0f));
        return;
    }

    const vec3 worldNormal = normalize(texture(sampler2D(inGbufferB, pointClampEdgeSampler), uv).rgb);
    const float depth = texture(sampler2D(inDepth, pointClampEdgeSampler), uv).r;
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

    float  sliceUvRadius = SSGIPush.uvRadius / linearDepth;

    float  ambientOcclusion = 0.f;
    vec3 bentNormal = vec3(0.f, 0.f, 0.f);
    vec3 localDO = vec3(0.f, 0.f, 0.f);

    for(int sliceIndex = 0; sliceIndex < SSGIPush.sliceCount; sliceIndex++)
    {
        float  sliceAngle  = ((sliceIndex + noise.x) / SSGIPush.sliceCount) * kPI;
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

        float  sideSigns[2] = {1.f, -1.f};
        float  horizonAngles[2];

        for (int sideIndex = 0; sideIndex < 2; ++sideIndex)
        {
            float  horizon_min = cos(projWorldNormalAngle + sideSigns[sideIndex] * kPI * 0.5);
            float  horizonCos = horizon_min;
            vec3 prevSampleWorldPos = vec3(0.f, 0.f, 0.f);

            for(int stepIndex = 0; stepIndex < SSGIPush.stepCount; stepIndex++)
            {
                float  sampleStep = ((stepIndex + noise.y) / SSGIPush.stepCount);
                vec2 sampleUvOffset = sampleStep * sliceUvDir;

                vec2 sampleUv = uv + sideSigns[sideIndex] * sampleUvOffset;


                float sampleDepth = texture(sampler2D(inDepth, pointClampEdgeSampler), sampleUv).r;
                vec3 sampleWorldPos = getWorldPos(sampleUv, sampleDepth, frameData);

                vec3 horizonWorldDir = sampleWorldPos - worldPos;
                float horizonWorldLen = length(horizonWorldDir); 
                horizonWorldDir /= horizonWorldLen;

                float sampleWeight = saturate(horizonWorldLen * SSGIPush.falloffMul + SSGIPush.falloffAdd);

                float sampleCos = mix(horizon_min, dot(horizonWorldDir, worldEyeDir), sampleWeight);
                float  prevHorizonCos  = horizonCos;
                horizonCos = max(horizonCos, sampleCos);

                // Performance heavy here.
                if (sampleCos >= prevHorizonCos)
                {
                    // A lot of cost. current don't use ssdo.
                    // From h3r2tic's demo
                    // https://github.com/h3r2tic/rtoy-samples/blob/main/assets/shaders/ssgi/ssgi.glsl
                    vec3 sampleLighting = texture(sampler2D(inHDRSceneColor, pointClampEdgeSampler), sampleUv).xyz;

                    vec3 sampleWorldNormal = normalize(texture(sampler2D(inGbufferB, pointClampEdgeSampler), sampleUv).rgb);

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

                    localDO += sampleLighting * sampleOcclusion * step(0, dot(-horizonWorldDir, sampleWorldNormal));
                }

                prevSampleWorldPos = sampleWorldPos;
            }

            // Ambient Occlusion
            horizonAngles[sideIndex] = sideSigns[sideIndex] * acos(horizonCos);
            ambientOcclusion += projWorldNormalLen * integrateHalfArc(horizonAngles[sideIndex], projWorldNormalAngle);

            // Global Lighting
            localDO *= projWorldNormalLen;
        }

        // Bent normal
        bentNormal += projWorldNormalLen * integrateBentNormal(
            horizonAngles[0], horizonAngles[1], projWorldNormalAngle, worldEyeDir, sliceViewDir);
    }

    ambientOcclusion /= SSGIPush.sliceCount;
    bentNormal = normalize(bentNormal);
    localDO /= SSGIPush.sliceCount;

    localDO = max(vec3(0.0), localDO);

    ambientOcclusion = 1.0 - (1.0 - pow(ambientOcclusion, SSGIPush.power)) * SSGIPush.intensity;

    imageStore(ssaoBentNormal, ivec2(workPos), vec4(0.5f * bentNormal + 0.5f, ambientOcclusion));
}