
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
        
#if 0
        // fog overlay.
        const float fogFalloff = 1.0f;
        
        const vec3 kBetaR = atmosphere.rayleighScattering; 
        const vec3 kBetaM = atmosphere.mieScattering;


        vec3 miePhaseValue = kBetaM * hgPhase(atmosphere.miePhaseG, -VoL);
        vec3 rayleighPhaseValue = kBetaR * rayleighPhase(VoL);


        vec3 fogColor = sunColor * (miePhaseValue + rayleighPhaseValue);
        fogColor /= (kBetaR + kBetaM);


        float opticalDepth = rayHitHeight * fogFalloff;
        vec3 transmittanceFog = exp(-opticalDepth * (kBetaR + kBetaM));
#endif
        // scatteredLight = mix(fogColor, scatteredLight, transmittanceFog);
    }



    imageStore(imageHdrSceneColor, workPos, vec4(result.rgb, 1.0));
}