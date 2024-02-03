#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable

#define SHARED_SAMPLER_SET 1
#include "common_shader.glsl"
#include "common_sampler.glsl"

layout (set = 0, binding = 0) buffer SSBOLensFlare { float ssboLensFlareDatas[]; };
layout (set = 0, binding = 1) uniform UniformFrameData { PerFrameData frameData; };
layout (set = 0, binding = 2) uniform texture2D inDepth;
layout (set = 0, binding = 3) uniform texture2D inFog;
layout (set = 0, binding = 4) uniform texture2D inCloudReconstructionTexture;
layout (set = 0, binding = 5) uniform texture2D inTransmittanceLut;

layout(push_constant) uniform PushConsts
{   
    uint bCloud;
    uint bFog;
};

layout (local_size_x = 1) in;
void main()
{
    vec4 projectPos = frameData.camViewProjNoJitter * vec4(frameData.sunLightInfo.direction * 9999999.0f, 1.0);
    projectPos.xyz /= projectPos.w;

    projectPos.xy = 0.5 * projectPos.xy + 0.5;
    projectPos.y  = 1.0 - projectPos.y;

    vec2 sunUv = projectPos.xy;
    if(!onRange(projectPos.xy, vec2(0.0), vec2(1.0)) || projectPos.w > 0.0f)
    {
        ssboLensFlareDatas[3] = 0.0f;
        return;
    }

    ssboLensFlareDatas[3] = 1.0f;

    vec4 cloudColor = texture(sampler2D(inCloudReconstructionTexture, linearClampEdgeSampler), sunUv);
    vec4 fogColor = texture(sampler2D(inFog, linearClampEdgeSampler), sunUv);

    float sceneZ = texture(sampler2D(inDepth, pointClampEdgeSampler), sunUv).r;

    if(sceneZ <= 0.0f)
    {
        if(bCloud > 0)
        {
            ssboLensFlareDatas[3] = cloudColor.a;
        }

        if(fogColor.a > 0.0f && bFog > 0)
        {
            ssboLensFlareDatas[3] *= fogColor.a;
        }
    }
    else
    {
        ssboLensFlareDatas[3] = 0.0f;
    }

    vec3 atmosphereTransmittance;
    {
        AtmosphereParameters atmosphere = getAtmosphereParameters(frameData);
        vec3 samplePos = vec3(0.0, atmosphere.cloudAreaStartHeight + atmosphere.cloudAreaThickness, 0.0);
        float sampleHeight = length(samplePos);

        const vec3 upVector = samplePos / sampleHeight;
        float viewZenithCosAngle = dot(-normalize(frameData.sunLightInfo.direction), upVector);
        vec2 sampleUv;
        lutTransmittanceParamsToUv(atmosphere, sampleHeight, viewZenithCosAngle, sampleUv);
        atmosphereTransmittance = texture(sampler2D(inTransmittanceLut, linearClampEdgeSampler), sampleUv).rgb;
    }

    ssboLensFlareDatas[0] = atmosphereTransmittance.x;
    ssboLensFlareDatas[1] = atmosphereTransmittance.y;
    ssboLensFlareDatas[2] = atmosphereTransmittance.z;
}