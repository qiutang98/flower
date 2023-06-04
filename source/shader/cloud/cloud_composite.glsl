
#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable

#include "cloud_common.glsl"

vec3 drawSun(vec3 rayDir, vec3 sunDir) 
{
    const float dT = dot(rayDir, sunDir);

    const float theta = 0.1 * kPI / 180.0;
    const vec3 sunCenterColor = vec3(1.0f, 0.92549, 0.87843) * 39.0f * 100;

    const float cT = cos(theta);
    if (dT >= cT) 
    {
        return sunCenterColor;
    }

    return vec3(0.0);
}

float getDensity2(float heightMeter)
{
    // TODO: 
    const float fogHeight = 0.0f;
    const float fogConst = 0.000f;

    return getDensity(heightMeter)
     + exp(-(frameData.camWorldPos.y - fogHeight) * 0.001) * 0.001 * 0.001 * 100.0 + fogConst;
}



layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    ivec2 texSize = imageSize(imageHdrSceneColor);
    ivec2 depthTextureSize = textureSize(inDepth, 0);
    ivec2 workPos = ivec2(gl_GlobalInvocationID.xy);

    if(workPos.x >= texSize.x || workPos.y >= texSize.y)
    {
        return;
    }

    const vec2 uv = (vec2(workPos) + vec2(0.5f)) / vec2(texSize);

    vec4 clipSpace = vec4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, 0.0, 1.0);
	vec4 viewPosH = frameData.camInvertProj * clipSpace;
    vec3 viewDir = viewPosH.xyz / viewPosH.w;
    vec3 worldDir = normalize((frameData.camInvertView * vec4(viewDir, 0.0)).xyz);

    vec4 srcColor = imageLoad(imageHdrSceneColor, workPos);
    float sceneZ = texture(sampler2D(inDepth, pointClampEdgeSampler), uv).r;
    // vec4 cloudColor = kuwaharaFilter(inCloudReconstructionTexture, linearClampEdgeSampler,uv);


    vec4 cloudColor = texture(sampler2D(inCloudReconstructionTexture, linearClampEdgeSampler), uv);
    vec4 fogColor = texture(sampler2D(inCloudFogReconstructionTexture, linearClampEdgeSampler), uv);

    float cloudDepth = texture(sampler2D(inCloudDepthReconstructionTexture, linearClampEdgeSampler), uv).r;
    cloudDepth = max(1e-5f, cloudDepth); // very far cloud may be negative, use small value is enough.

    vec3 result = srcColor.rgb;

    if(sceneZ <= 0.0f) // reverse z.
    {
        // Composite planar cloud.

        result = srcColor.rgb * cloudColor.a + cloudColor.rgb;  

        if(fogColor.a >= 0.0f)
        {
            result.rgb = result.rgb * fogColor.a + max(vec3(0.0f), fogColor.rgb);
        }
    }
    else
    {
        const uint  kGodRaySteps = 64;
        const float kMaxMarchingDistance = 400.0f;

        // We are revert z.
        vec4 clipSpace = vec4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, 0.0, 1.0);
        vec4 viewPosH = frameData.camInvertProj * clipSpace;
        vec3 viewSpaceDir = viewPosH.xyz / viewPosH.w;
        vec3 worldDir = normalize((frameData.camInvertView * vec4(viewSpaceDir, 0.0)).xyz);


        AtmosphereParameters atmosphere = getAtmosphereParameters(frameData);
        const SkyInfo sky = frameData.sky;
        vec3 worldPosWP      = getWorldPos(uv, sceneZ, frameData);
        vec3 pixelToCameraWP = frameData.camWorldPos.xyz - worldPosWP;

        float pixelToCameraDistanceWP = max(1e-5f, length(pixelToCameraWP));
        vec3 rayDirWP = pixelToCameraWP / pixelToCameraDistanceWP;

        float marchingDistance = min(kMaxMarchingDistance, pixelToCameraDistanceWP);
        if(pixelToCameraDistanceWP > kMaxMarchingDistance)
        {
            worldPosWP = frameData.camWorldPos.xyz - rayDirWP * marchingDistance;
        }

        vec3 sunDirection = -normalize(frameData.sky.direction);
        float VoL = dot(worldDir, sunDirection);
        
        float stepLength = marchingDistance / float(kGodRaySteps);
        vec3 stepRay = rayDirWP * stepLength;
    
        // Interval noise is better than blue noise here.
        float taaOffset = interleavedGradientNoise(workPos, frameData.frameIndex.x % frameData.jitterPeriod);

        vec3 rayPosWP = worldPosWP + stepRay * (taaOffset + 0.05);

        float transmittance2  = 1.0;
        vec3 scatteredLight2 = vec3(0.0, 0.0, 0.0);

        float miePhaseValue = hgPhase(atmosphere.miePhaseG, -VoL);
        float rayleighPhaseValue = rayleighPhase(VoL);
        vec3 sunColor = frameData.sky.color * frameData.sky.intensity;
        vec3 groundToCloudTransfertIsoScatter =  texture(samplerCube(inSkyIrradiance, linearClampEdgeSampler), vec3(0, 1, 0)).rgb;

        for(uint i = 0; i < kGodRaySteps; i ++)
        {
            float visibilityTerm = 1.0;
            {
                // First find active cascade.
                uint activeCascadeId = 0;
                vec3 shadowCoord;

                // Loop to find suitable cascade.
                for(uint cascadeId = 0; cascadeId < sky.cacsadeConfig.cascadeCount; cascadeId ++)
                {
                    shadowCoord = projectPos(rayPosWP, cascadeInfos[cascadeId].viewProj);
                    if(onRange(shadowCoord.xyz, vec3(sky.cacsadeConfig.cascadeBorderAdopt), vec3(1.0f - sky.cacsadeConfig.cascadeBorderAdopt)))
                    {
                        break;
                    }
                    activeCascadeId ++;
                }

                if(activeCascadeId < sky.cacsadeConfig.cascadeCount)
                {
                    const float perCascadeOffsetUV = 1.0f / sky.cacsadeConfig.cascadeCount;
                    const float shadowTexelSize = 1.0f / float(sky.cacsadeConfig.percascadeDimXY);

                    // Main cascsade shadow compute.
                    {
                        vec3 shadowPosOnAltas = shadowCoord;
                        
                        // Also add altas bias and z bias.
                        shadowPosOnAltas.x = (shadowPosOnAltas.x + float(activeCascadeId)) * perCascadeOffsetUV;
                        shadowPosOnAltas.z += 0.001 * (activeCascadeId + 1.0);

                        float depthShadow = texture(sampler2D(inSDSMShadowDepth, pointClampEdgeSampler), shadowPosOnAltas.xy).r;
                        visibilityTerm = shadowPosOnAltas.z > depthShadow ? 1.0 : 0.0;
                    }
                }

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

            float density = getDensity2(pixelToCameraWP.y);

            float sigmaS = density;
            float sigmaE = max(sigmaS, 1e-8f);


            vec3 phaseTimesScattering = vec3(miePhaseValue + rayleighPhaseValue);
            vec3 sunSkyLuminance = groundToCloudTransfertIsoScatter + visibilityTerm * sunColor * phaseTimesScattering * atmosphereTransmittance;

            vec3 sactterLitStep = sunSkyLuminance * sigmaS;

            float stepTransmittance = exp(-sigmaE * stepLength);
            scatteredLight2 += transmittance2 * (sactterLitStep - sactterLitStep * stepTransmittance) / sigmaE; 
            transmittance2 *= stepTransmittance;

            // Step.
            rayPosWP += stepRay;
        }

        result.rgb = result.rgb * transmittance2 + scatteredLight2;
    }




    imageStore(imageHdrSceneColor, workPos, vec4(result.rgb, 1.0));
}