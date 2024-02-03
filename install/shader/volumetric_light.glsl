#version 460

#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable

// In fact, we should call it volumetric fog.

// Voxel cover 160 meter in front of camera.
const float kVolumetricFogVoxelDistance = 160.0f; 

#define SHARED_SAMPLER_SET    1
#define BLUE_NOISE_BUFFER_SET 2
#include "common_shader.glsl"

layout (set = 0, binding = 0) uniform UniformFrameData { PerFrameData frameData; };
layout (set = 0, binding = 1, rgba16f) uniform image3D imageFroxelScatter;
layout (set = 0, binding = 2) uniform texture3D inFroxelScatter;
layout (set = 0, binding = 3, rgba16f) uniform image2D imageHdrSceneColor;
layout (set = 0, binding = 4) uniform texture2D inDepth;
layout (set = 0, binding = 5) buffer SSBOCascadeInfoBuffer { CascadeInfo cascadeInfos[]; }; 
layout (set = 0, binding = 6) uniform texture2D inCloudShadowDepth;
layout (set = 0, binding = 7, rgba16f) uniform image3D imageScatterTransmittance;
layout (set = 0, binding = 8) uniform texture3D inScatterTransmittance;
layout (set = 0, binding = 9) uniform texture3D inFroxelScatterHistory;
layout (set = 3, binding = 0) uniform  texture2D texture2DBindlessArray[];

layout (push_constant) uniform PushConsts 
{  
    uint sdsmShadowDepthIndices[kMaxCascadeNum];
    uint cascadeCount;
};

vec4 texSDSMDepth(uint cascadeId, vec2 uv)
{
    return texture(
        sampler2D(texture2DBindlessArray[nonuniformEXT(sdsmShadowDepthIndices[cascadeId])], pointClampEdgeSampler), uv);
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
    return shadowCoord.z - 0.002f > depthShadow ? 1.0 : 0.0;
}

#ifdef INJECT_LIGHTING_PASS

// 160x88x64 -> 20x11x64
layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main()
{
    ivec3 lutSize = imageSize(imageFroxelScatter);
    ivec3 workPos = ivec3(gl_GlobalInvocationID.xyz);

    AtmosphereParameters atmosphere = getAtmosphereParameters(frameData);

    float jitter = 0.0f;
    {
        // Jitter in 3d coordinate.
        uvec2 lut2dSize = lutSize.xy;
        lut2dSize.x *= lutSize.z;

        uvec2 work2dPos = workPos.xy;
        work2dPos.x += workPos.z * lutSize.x;


        uvec2 offset = uvec2(vec2(0.754877669, 0.569840296) * (frameData.frameIndex.x) * lut2dSize);
        uvec2 offsetId = uvec2(work2dPos) + offset;

        offsetId.x = offsetId.x % lut2dSize.x;
        offsetId.y = offsetId.y % lut2dSize.y;

        jitter = samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d(offsetId.x, offsetId.y, 0, 0u);
    }

    vec3 froxelUvZ = (vec3(workPos) + vec3(0.5) + vec3(0.0, 0.0, jitter - 0.5)) / vec3(lutSize);

    // Get world space direction, then do a ray cast.

    vec3 worldDir;
    {
        vec4 clipSpaceEnd   = vec4(froxelUvZ.x * 2.0f - 1.0f, 1.0f - froxelUvZ.y * 2.0f, 0.0, 1.0);
        vec4 worldPosEndH   = frameData.camInvertViewProj * clipSpaceEnd;
        vec3 worldEnd       = worldPosEndH.xyz   / worldPosEndH.w;

        // Now get world direction.
        worldDir = normalize(worldEnd - frameData.camWorldPos.xyz);
    }

    vec3 worldPos = frameData.camWorldPos.xyz + worldDir * froxelUvZ.z * kVolumetricFogVoxelDistance;

    float visibility = computeVisibilitySDSM(worldPos);

    // Compute froxel lighting info.
    vec3 sunColor = frameData.sunLightInfo.color * frameData.sunLightInfo.intensity;

    vec3 scatteredLight = sunColor * visibility;
    float density = frameData.cloud.cloudGodRayScale * 5e-5f;

    vec4 result = vec4(scatteredLight, density);

    // Temporal accumulate.
    if(frameData.bCameraCut == 0)
    {
        vec3 worldPosNoJitter;
        float uvz = (workPos.z + 0.5) / lutSize.z;
        {
            // World end is current froxel position.
            worldPosNoJitter = frameData.camWorldPos.xyz + worldDir * uvz * kVolumetricFogVoxelDistance;
        }

        // Project get prev frame froxelUvZ.
        vec3 prevViewPos = (frameData.camViewPrev * vec4(worldPosNoJitter, 1.0)).xyz;
        vec3 prevFroxelUvZNoJitter = projectPos(worldPosNoJitter, frameData.camViewProjPrev);
        prevFroxelUvZNoJitter.z = -prevViewPos.z / kVolumetricFogVoxelDistance;

        if (onRange(prevFroxelUvZNoJitter, vec3(0.0), vec3(1.0)))
        {
            vec4 sampleGridHistory = texture(sampler3D(inFroxelScatterHistory, linearClampEdgeSampler), prevFroxelUvZNoJitter);
            result = mix(sampleGridHistory, result, 0.05f);
        }
    }

    imageStore(imageFroxelScatter, workPos, result);
}

#endif

#ifdef ACCUMUALTE_PASS

layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main()
{
    const ivec3 lutSize = imageSize(imageScatterTransmittance);
    const float stepLength = kVolumetricFogVoxelDistance / lutSize.z;

    vec3  accumulateScatter = vec3(0.0);
    float accumulateTransmittance = 1.0;

    for(int z = 0; z < lutSize.z; z ++)
    {
        ivec3 workPos = ivec3(gl_GlobalInvocationID.xy, z);

        // Sample prev compute density and scattered light.
        vec4 sampleGrid = texelFetch(inFroxelScatter, workPos, 0);

        vec3 scatteredLight = sampleGrid.xyz;
        float density = sampleGrid.w;

        float sigmaS = density;
        float sigmaE = max(sigmaS, 1e-8f);

        vec3 sactterLitStep = scatteredLight * sigmaS;
        float stepTransmittance = exp(-sigmaE * stepLength);

        accumulateScatter += accumulateTransmittance * (sactterLitStep - sactterLitStep * stepTransmittance) / sigmaE;
        accumulateTransmittance *= stepTransmittance;

        imageStore(imageScatterTransmittance, workPos, vec4(accumulateScatter, accumulateTransmittance));
    }
}

#endif

#ifdef COMPOSITE_PASS 

layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    ivec2 texSize = imageSize(imageHdrSceneColor);
    ivec2 workPos = ivec2(gl_GlobalInvocationID.xy);

    if(workPos.x >= texSize.x || workPos.y >= texSize.y)
    {
        return;
    }

    const vec2 uv = (vec2(workPos) + vec2(0.5f)) / vec2(texSize);

    vec4 srcColor = imageLoad(imageHdrSceneColor, workPos);
    float sceneZ = texture(sampler2D(inDepth, pointClampEdgeSampler), uv).r;
    float linearDepth = linearizeDepth(sceneZ, frameData);

    if(linearDepth < kVolumetricFogVoxelDistance)
    {
        vec3 uvZ;

        uvZ.xy = uv;
        uvZ.z  = linearDepth / kVolumetricFogVoxelDistance;

        vec4 fog = textureTricubic(inScatterTransmittance, linearClampEdgeSampler, uvZ);
        srcColor.xyz = srcColor.xyz * fog.w + fog.xyz;
    }

    imageStore(imageHdrSceneColor, workPos, srcColor);
}

#endif 