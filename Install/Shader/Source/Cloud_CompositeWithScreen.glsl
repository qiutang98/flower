
#version 460

/*
** Physical based render code, develop by engineer: qiutanguu.
*/

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable
#include "Cloud_Common.glsl"
#include "KuwaharaFilter.glsl"

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

float getDensity(vec3 worldPosition, float distToEye)
{
    const float fogStartHeight = 0.0;
    const float heightFallOff  = 0.1;

    float heightFog = exp(-(worldPosition.y - fogStartHeight) * heightFallOff);
    
    // Height fog.
    return 0.0001 + heightFog; // 1.0;
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
	vec4 viewPosH = viewData.camInvertProj * clipSpace;
    vec3 viewDir = viewPosH.xyz / viewPosH.w;
    vec3 worldDir = normalize((viewData.camInvertView * vec4(viewDir, 0.0)).xyz);

    vec4 srcColor = imageLoad(imageHdrSceneColor, workPos);
    float sceneZ = texture(sampler2D(inDepth, pointClampEdgeSampler), uv).r;
    // vec4 cloudColor = kuwaharaFilter(inCloudReconstructionTexture, linearClampEdgeSampler,uv);

    vec4 cloudColor = texture(sampler2D(inCloudReconstructionTexture, linearClampEdgeSampler), uv);

    float cloudDepth = texture(sampler2D(inCloudDepthReconstructionTexture, linearClampEdgeSampler), uv).r;
    cloudDepth = max(1e-5f, cloudDepth); // very far cloud may be negative, use small value is enough.

    vec3 result = srcColor.rgb;
    if(sceneZ <= cloudDepth) // reverse z.
    {
        result = mix(srcColor.rgb, cloudColor.rgb, 1.0 - cloudColor.a);
    }

    // God ray for light.
    #if 0
    {
        AtmosphereParameters atmosphere = getAtmosphereParameters(frameData);

        const uint  kGodRaySteps = 64;
        const float kMaxMarchingDistance = 400.0f;

        const DirectionalLightInfo light = frameData.directionalLight;

        vec3 worldPosWP      = getWorldPos(uv, sceneZ, viewData);
        vec3 pixelToCameraWP = viewData.camWorldPos.xyz - worldPosWP;

        float pixelToCameraDistanceWP = max(1e-5f, length(pixelToCameraWP));
        vec3 rayDirWP = pixelToCameraWP / pixelToCameraDistanceWP;

        float marchingDistance = min(kMaxMarchingDistance, pixelToCameraDistanceWP);
        if(pixelToCameraDistanceWP > kMaxMarchingDistance)
        {
            worldPosWP = viewData.camWorldPos.xyz - rayDirWP * marchingDistance;
        }
        
        float stepLength = marchingDistance / float(kGodRaySteps);
        vec3 stepRay = rayDirWP * stepLength;
    
        // Interval noise is better than blue noise here.
        float taaOffset = interleavedGradientNoise(workPos, frameData.frameIndex.x % frameData.jitterPeriod);

        vec3 rayPosWP = worldPosWP + stepRay * (taaOffset + 0.05);

        float transmittance  = 1.0;
        vec3 scatteredLight = vec3(0.0, 0.0, 0.0);

        vec3 sunColor = frameData.directionalLight.color * frameData.directionalLight.intensity;

        vec3 sunDirection = -normalize(frameData.directionalLight.direction);
        float VoL = dot(-rayDirWP, sunDirection);
        const float cosTheta = -VoL;
        float phase = hgPhase(0.3, cosTheta);

        for(uint i = 0; i < kGodRaySteps; i ++)
        {
            float visibilityTerm = 1.0;
            {
                // First find active cascade.
                uint activeCascadeId = 0;
                vec3 shadowCoord;

                // Loop to find suitable cascade.
                for(uint cascadeId = 0; cascadeId < light.cascadeCount; cascadeId ++)
                {
                    shadowCoord = projectPos(rayPosWP, cascadeInfos[cascadeId].viewProj);
                    if(onRange(shadowCoord.xyz, vec3(light.cascadeBorderAdopt), vec3(1.0f - light.cascadeBorderAdopt)))
                    {
                        break;
                    }
                    activeCascadeId ++;
                }

                if(activeCascadeId < light.cascadeCount)
                {
                    const float perCascadeOffsetUV = 1.0f / light.cascadeCount;
                    const float shadowTexelSize = 1.0f / float(light.perCascadeXYDim);

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

                float viewZenithCosAngle = dot(-normalize(frameData.directionalLight.direction), upVector);
                vec2 sampleUv;
                lutTransmittanceParamsToUv(atmosphere, viewHeight, viewZenithCosAngle, sampleUv);
                atmosphereTransmittance = texture(sampler2D(inTransmittanceLut, linearClampEdgeSampler), sampleUv).rgb;
            }

            float density = getDensity(rayPosWP, pixelToCameraDistanceWP);

            float sigmaS = density * 0.01;
            float sigmaE = 0.001 * density + 1e-4f;

            vec3 sunSkyLuminance = vec3(0.1) + visibilityTerm * sunColor * phase; // TODO: Sample SH as ambient light.

            vec3 sactterLitStep = sunSkyLuminance * sigmaS;

            float stepTransmittance = exp(-sigmaE * stepLength);
            scatteredLight += atmosphereTransmittance * transmittance * (sactterLitStep - sactterLitStep * stepTransmittance) / max(1e-4f, sigmaE); // TODO: Add ambient light and atmosphere transmittance.

            transmittance *= stepTransmittance;

            // Step.
            rayPosWP += stepRay;
        }

        result = result.rgb * transmittance + scatteredLight; 

        // result = scatteredLight;
    }
    #endif


    imageStore(imageHdrSceneColor, workPos, vec4(result.rgb, 1.0));
}