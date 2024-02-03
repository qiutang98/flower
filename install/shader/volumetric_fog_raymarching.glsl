#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_KHR_shader_subgroup_quad : enable

#define SAMPLE_CLOUD_SHADOW   1

#define SHARED_SAMPLER_SET    1
#define BLUE_NOISE_BUFFER_SET 2
layout (set = 3, binding = 0) uniform  texture2D texture2DBindlessArray[];

#include "common_sampler.glsl"
#include "common_shader.glsl"
#include "common_lighting.glsl"

layout (set = 0, binding = 0, rgba16f) uniform image2D imageHdrSceneColor;
layout (set = 0, binding = 1) uniform texture2D inHdrSceneColor;
layout (set = 0, binding = 2) uniform texture2D inDepth;
layout (set = 0, binding = 3) uniform texture2D inTransmittanceLut;
layout (set = 0, binding = 4) uniform texture3D inFroxelScatter;
layout (set = 0, binding = 5) buffer SSBOCascadeInfoBuffer { CascadeInfo cascadeInfos[]; }; 
layout (set = 0, binding = 6) uniform texture2D inCloudShadowDepth;
layout (set = 0, binding = 7) uniform UniformFrameData { PerFrameData frameData; };
layout (set = 0, binding = 8) uniform textureCube inSkyIrradiance;
layout (set = 0, binding = 9) uniform texture2D inCloudDistantLit; 
layout (set = 0, binding = 10, rgba16f) uniform image2D imageFog;
layout (set = 0, binding = 11) uniform texture2D inHzbClosest;

layout (set = 0, binding = 12) uniform texture2D inFog;
layout (set = 0, binding = 13) uniform texture2D inFogSky;

layout (set = 0, binding = 14) uniform texture2D inHzbFar;

layout (push_constant) uniform PushConsts 
{  
    uint sdsmShadowDepthIndices[kMaxCascadeNum];
    uint cascadeCount;
    uint kGodRaySteps;
    uint kSkyPass;
};

#define kDepthStartAddTraceNum  2000.0
#define kDepthStartAddTraceNum2 4000.0
#define kMaxAddTraceTimes      2.0
#define kMinTraceDepth         -200.0
vec4 texSDSMDepth(uint cascadeId, vec2 uv)
{
    return texture(
        sampler2D(texture2DBindlessArray[nonuniformEXT(sdsmShadowDepthIndices[cascadeId])], pointClampEdgeSampler), uv);
}

float getDensity(vec3 worldPosMeter)
{
    float dis = distance(worldPosMeter, frameData.camWorldPos.xyz);
    dis = (worldPosMeter - frameData.camWorldPos.xyz).y;
    float fogHeight = 0.0;
    float fogHeightFalloff = 0.005;

    float dis2Cam = abs(worldPosMeter.y - fogHeight);

    float fog0 = exp(- dis2Cam * fogHeightFalloff * 2.0) * 0.001 * 0.001 * 4.0 * frameData.cloud.cloudGodRayScale;
    float fog1 = exp(- dis2Cam * fogHeightFalloff) * 0.001 * 0.001 * frameData.cloud.cloudGodRayScale;


    return max(fog1, fog0);
}

float computeVisibilitySDSM(vec3 worldPos)
{
    // First find active cascade.
    uint activeCascadeId = 0;
    vec3 shadowCoord;
    // Loop to find suitable cascade.
    for(uint cascadeId = 0; cascadeId < cascadeCount; cascadeId ++)
    {
        // Perspective divide to get ndc position.
        shadowCoord = projectPos(worldPos, cascadeInfos[cascadeId].viewProj);

        // Check current cascade is valid in range.
        if(onRange(shadowCoord.xyz, vec3(0.0), vec3(1.0)))
        {
            break;
        }
        activeCascadeId ++;
    }

    // Out of shadow area return lit.
    if(activeCascadeId == cascadeCount)
    {
        return 1.0f;
    }

    float depthShadow = texSDSMDepth(activeCascadeId, shadowCoord.xy).x;

    // Add bias avoid light leak.
    return shadowCoord.z > depthShadow ? 1.0 : 0.0;
}

float computeVisibilityCloud(vec3 worldPos, in AtmosphereParameters atmosphere)
{
    vec3 skyPos = convertToAtmosphereUnit(worldPos, frameData) + vec3(0.0, atmosphere.bottomRadius, 0.0);
    float cloudShadow = 1.0f;
    // Now convert cloud coordinate.
    vec3 cloudUvz = projectPos(skyPos, atmosphere.cloudShadowViewProj);
    vec2 texSize = textureSize(inCloudShadowDepth, 0).xy;

    if(onRange(cloudUvz.xy, vec2(0), vec2(1)))
    {
        float cloudExpZ = texture(sampler2D(inCloudShadowDepth, linearClampEdgeSampler), cloudUvz.xy).y;
        float cloudComputeExpZ = cloudExpZ * exp(kCloudShadowExp * cloudUvz.z);
        cloudShadow = min(cloudShadow, saturate(cloudComputeExpZ));
    }

    return cloudShadow;
}

#ifdef COMPUTE_PASS

// NOTE: Variable rate trace.
// Sky area: trace 4x4 per ray.
// 
layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    ivec2 texSize = imageSize(imageFog);
    ivec2 depthTextureSize = textureSize(inDepth, 0);
    
    uvec2 groupThreadId = remap8x8(gl_LocalInvocationIndex);
    uvec2 dispatchId = groupThreadId + gl_WorkGroupID.xy * 8;

    ivec2 workPos = ivec2(dispatchId.xy);


    if(workPos.x >= texSize.x || workPos.y >= texSize.y)
    {
        return;
    }

    if(frameData.cloud.cloudGodRay == 0)
    {
        imageStore(imageFog, workPos, vec4(0.0,0.0,0.0,1.0));
        return;
    }


    AtmosphereParameters atmosphere = getAtmosphereParameters(frameData);

    const vec2 uv = (vec2(workPos) + vec2(0.5f)) / vec2(texSize);
    float sceneZ = textureLod(sampler2D(inDepth, pointClampEdgeSampler), uv, 0.0).r;


    {
        const bool bNonSkyPass     = (kSkyPass == 0);

        const bool bSkyPass        = (kSkyPass == 1);
        const bool bMixFullResPass = (kSkyPass == 2);

        const float HZBLevel = 3.0f;

        if(bSkyPass)
        {
            float safeZ = textureLod(sampler2D(inHzbFar, pointClampEdgeSampler), uv, HZBLevel).r;
            const bool bFullSky = safeZ <= 0.0;

            // sky pass.
            if(!bFullSky)
            {
                imageStore(imageFog, workPos, vec4(0.0,0.0,0.0, -1.0));
                return;
            }
        }
        
        if(bNonSkyPass)
        {
            float safeZ = textureLod(sampler2D(inHzbClosest, pointClampEdgeSampler), uv, HZBLevel).r;
            const bool bFullSky = safeZ <= 0.0;

            // Non sky pass.
            // skip full sky area.
            if(bFullSky)
            {
                imageStore(imageFog, workPos, vec4(0.0,0.0,0.0, -1.0));
                return;
            }
        }

        if(bMixFullResPass)
        {
            float safeZ0 = textureLod(sampler2D(inHzbFar, pointClampEdgeSampler), uv,     HZBLevel + 1.0).r;
            float safeZ1 = textureLod(sampler2D(inHzbClosest, pointClampEdgeSampler), uv, HZBLevel + 1.0).r;

            if(sceneZ > 0.0 && safeZ0 > 0.0)
            {
                // linear filter is enough.
                vec4 fog = texture(sampler2D(inFog, linearClampEdgeSampler), uv);

                imageStore(imageFog, workPos, fog);
                return;
            }
            else
            {
                vec4 pointFog = texture(sampler2D(inFog, pointClampEdgeSampler), uv);
                vec4 pointFogSky = texture(sampler2D(inFogSky, pointClampEdgeSampler), uv);

                if(pointFog.w < -0.5 && pointFogSky.w > - 0.5 && safeZ1 <= 0.0)
                {
                    // Guassian 3x3.
                    const float kernel[2][2] = 
                    {
                        { 1.0 / 4.0, 1.0 / 8.0 },
                        { 1.0 / 8.0, 1.0 / 16.0 }
                    };
                    vec2 texelSkyFogSize = 1.0 / vec2(textureSize(inFogSky, 0));

                    vec4 sum = pointFogSky * kernel[0][0];
                    sum += texture(sampler2D(inFogSky, pointClampEdgeSampler), uv + vec2( 1,  1) * texelSkyFogSize) * kernel[1][1];
                    sum += texture(sampler2D(inFogSky, pointClampEdgeSampler), uv + vec2(-1, -1) * texelSkyFogSize) * kernel[1][1];
                    sum += texture(sampler2D(inFogSky, pointClampEdgeSampler), uv + vec2( 0,  1) * texelSkyFogSize) * kernel[0][1];
                    sum += texture(sampler2D(inFogSky, pointClampEdgeSampler), uv + vec2( 0, -1) * texelSkyFogSize) * kernel[0][1];
                    sum += texture(sampler2D(inFogSky, pointClampEdgeSampler), uv + vec2( 1,  0) * texelSkyFogSize) * kernel[1][0];
                    sum += texture(sampler2D(inFogSky, pointClampEdgeSampler), uv + vec2(-1,  0) * texelSkyFogSize) * kernel[1][0];
                    sum += texture(sampler2D(inFogSky, pointClampEdgeSampler), uv + vec2( 1, -1) * texelSkyFogSize) * kernel[1][1];
                    sum += texture(sampler2D(inFogSky, pointClampEdgeSampler), uv + vec2(-1,  1) * texelSkyFogSize) * kernel[1][1];

                    imageStore(imageFog, workPos, sum);
                    return;
                }
            }
        }
    }


    bool bSky  = (sceneZ <= 0.0);

    // Offset retarget for new seeds each frame
    uvec2 offset = uvec2(vec2(0.754877669, 0.569840296) * (frameData.frameIndex.z) * uvec2(texSize));
    uvec2 offsetId = workPos.xy + offset;
    offsetId.x = offsetId.x % texSize.x;
    offsetId.y = offsetId.y % texSize.y;
    float blueNoise2 = samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d(offsetId.x, offsetId.y, 0, 0u); 

    vec4 clipSpace = vec4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, 0.0, 1.0);
	vec4 viewPosH = frameData.camInvertProj * clipSpace;
    vec3 viewDir = viewPosH.xyz / viewPosH.w;
    vec3 worldDir = normalize((frameData.camInvertView * vec4(viewDir, 0.0)).xyz);

    vec3 worldPosWP      = getWorldPos(uv, sceneZ, frameData);
    vec3 pixelToCameraWP = worldPosWP - frameData.camWorldPos.xyz;

    float pixelToCameraDistanceWP = max(1e-5f, length(pixelToCameraWP));
    vec3 rayDirWP = normalize(pixelToCameraWP);

    float marchingDistance = pixelToCameraDistanceWP;
    uint rayStepNum = kGodRaySteps;



    float transmittance2  = 1.0;
    vec3 scatteredLight2 = vec3(0.0, 0.0, 0.0);

    if(bSky)
    {
        vec3 c = convertToAtmosphereUnit(frameData.camWorldPos.xyz, frameData) + vec3(0.0, atmosphere.bottomRadius, 0.0);
        marchingDistance = 1000.0f * (atmosphere.cloudAreaStartHeight -  c.y) / clamp(worldDir.y, 0.1, 1.0);

        // 5km min trace.
        marchingDistance = max(marchingDistance, 8.0 * 1000.0);
    }

    rayStepNum = uint(float(rayStepNum) * mix(1.0, kMaxAddTraceTimes, saturate((marchingDistance - kDepthStartAddTraceNum) / kDepthStartAddTraceNum2)));

    vec3 sunDirection = -normalize(frameData.sunLightInfo.direction);
    float VoL = dot(worldDir, sunDirection);
    
    float stepLength = marchingDistance / float(rayStepNum);
    vec3 stepRay = rayDirWP * stepLength;

    vec3 rayPosWP = frameData.camWorldPos.xyz + stepRay * (blueNoise2 + 0.05);

    float miePhaseValue0 = hgPhase( 0.5, -VoL);
    float miePhaseValue1 = hgPhase(-0.4, -VoL);
    float miePhaseValue = mix(miePhaseValue0, miePhaseValue1, 0.5);

    vec3 sunColor = frameData.sunLightInfo.color * frameData.sunLightInfo.intensity;
    vec3 distantLit = texelFetch(inCloudDistantLit, ivec2(0, 0), 0).xyz;



    for(uint i = 0; i < rayStepNum; i ++)
    {
        vec3 disToCam = rayPosWP - frameData.camWorldPos.xyz;

        float visibilityTerm = 1.0f;
        {
            visibilityTerm = computeVisibilitySDSM(rayPosWP);
            float cloudShadow = computeVisibilityCloud(rayPosWP, atmosphere);

            cloudShadow = remap(cloudShadow, 0.0, 1.0, saturate(1.0 - sunDirection.y * 2.0), 1.0);

            visibilityTerm = min(visibilityTerm, cloudShadow);
        }


        // Second evaluate transmittance due to participating media
        vec3 atmosphereTransmittance;
        {
            vec3 P0 = rayPosWP * 0.001 + vec3(0.0, atmosphere.bottomRadius, 0.0); // meter -> kilometers.
            float viewHeight = length(P0);
            const vec3 upVector = P0 / viewHeight;

            float viewZenithCosAngle = dot(sunDirection, upVector);
            vec2 sampleUv;
            lutTransmittanceParamsToUv(atmosphere, viewHeight, viewZenithCosAngle, sampleUv);
            atmosphereTransmittance = texture(sampler2D(inTransmittanceLut, linearClampEdgeSampler), sampleUv).rgb;
        }

        vec3 airLit;
        {
            float tDepth = 0.001 * length(rayPosWP - frameData.camWorldPos.xyz); // meter -> kilometers.
            float slice = distantGridDepthToSlice(tDepth);

            float weight = 1.0;
            if (slice < 0.5)
            {
                // We multiply by weight to fade to 0 at depth 0. That works for luminance and opacity.
                weight = saturate(slice * 2.0);
                slice = 0.5;
            }
            ivec3 sliceLutSize = textureSize(inFroxelScatter, 0);
            float w = sqrt(slice / float(sliceLutSize.z));	// squared distribution

            airLit = weight * texture(sampler3D(inFroxelScatter, linearClampEdgeSampler), vec3(uv, w)).xyz;
        }

        float density = getDensity(rayPosWP);

        float sigmaS = density;
        float sigmaE = max(sigmaS, 1e-8f);

        vec3 S = (airLit + visibilityTerm * sunColor * miePhaseValue * atmosphereTransmittance) * sigmaS;
        vec3 Sint = (S - S * exp(-sigmaE * stepLength)) / sigmaE;

        scatteredLight2 += Sint * transmittance2;
        transmittance2  *= exp(-sigmaE * stepLength);

        if(transmittance2 < 1e-3f)
        {
            break;
        }

        rayPosWP += stepRay;
    }

    vec4 result = vec4(scatteredLight2, transmittance2);
    imageStore(imageFog, workPos, result);
}

#endif 