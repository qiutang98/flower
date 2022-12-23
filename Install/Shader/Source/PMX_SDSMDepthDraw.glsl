#version 460
#extension GL_EXT_nonuniform_qualifier : require

#include "../../glsl/common.glsl"
#include "../../glsl/common_framedata.glsl"
#include "../../glsl/common_shadow.glsl"
#include "../../glsl/brdf.glsl"

layout (location = 0) in  vec3 inWorldNormal;
layout (location = 1) in  vec2 inUV0;
layout (location = 2) in  vec3 inWorldPos;
layout (location = 3) in  vec4 inPrevPosNoJitter;
layout (location = 4) in  vec4 inCurPosNoJitter;
layout (location = 5) in  vec4 inGLPos;

layout (location = 0) out vec4  outHdrColor; 
layout (location = 1) out vec2  outVelocity;
layout (location = 2) out vec4  outSSSSDiffuse;
layout (location = 3) out float outShadingModel;


#include "pmx_pass.glsl"
#include "../../glsl/sky_common.glsl"

layout(set = 2, binding = 0) uniform sampler2D BindlessSampler2D[];

vec4 texlod(uint id,vec2 uv, float lod){ return textureLod(BindlessSampler2D[nonuniformEXT(id)], uv, lod);}
vec4 tex(uint id,vec2 uv){ return texture(BindlessSampler2D[nonuniformEXT(id)],uv);}

layout (set = 1, binding = 0) uniform sampler2DArray inShadowDepthBilinearTexture;
layout (set = 1, binding = 1) readonly buffer CascadeInfoBuffer
{
	CascadeInfo cascadeInfosbuffer[];
};
layout (set = 1, binding = 2) uniform sampler2D inBRDFLut;
layout (set = 1, binding = 3) uniform samplerCube inIrradiancePrefilter;
layout (set = 1, binding = 4) uniform samplerCube inEnvSpecularPrefilter;


bool unValid(vec3 vector)
{
    return isnan(vector.x) || isnan(vector.y) || isnan(vector.z) || isinf(vector.x) || isinf(vector.y) || isinf(vector.z);
}

vec3 biasNormalOffset(vec3 vertexNormal, float NoLSafe, float bias, float texelSize)
{
    return vertexNormal * (1.0f - NoLSafe) * bias * texelSize * 10.0f;
}

float autoBias(float safeNoL,float biasMul, float bias)
{
    float fixedFactor = 0.0005f;
    float slopeFactor = 1.0f - safeNoL;

    return fixedFactor + slopeFactor * bias * biasMul * 0.0001f;
}

float evaluateDirectShadow(
    float NoLSafe, 
    vec3 fragWorldPos,
    vec3 normal,
    out uint outIndex,
    vec3 lightDirection,
    vec3 vertexNormal)
{
    ivec2 texDim = textureSize(inShadowDepthBilinearTexture,0).xy;
	vec2 texelSize = 1.0f / vec2(texDim);
    float shadowFactor = 1.0f;
    const uint cascadeCount = 4;
    float VertexNoLSafe = clamp(dot(vertexNormal,lightDirection), 0.0f, 1.0f);

    const float biasFactor = 2.0f;
    vec3 worldPosProcess = fragWorldPos + biasNormalOffset(vertexNormal, VertexNoLSafe, biasFactor, texelSize.x);

    for(uint cascadeIndex = 0; cascadeIndex < cascadeCount; cascadeIndex ++)
	{
        vec4 shadowClipPos = cascadeInfosbuffer[cascadeIndex].cascadeViewProjMatrix * vec4(worldPosProcess, 1.0);
        vec4 shadowCoordNdc = shadowClipPos / shadowClipPos.w;
        vec4 shadowCoord = shadowCoordNdc;
		shadowCoord.st = shadowCoord.st * 0.5f + 0.5f;
        shadowCoord.y = 1.0f - shadowCoord.y;

        if( shadowCoord.x > 0.01f && shadowCoord.y > 0.01f && 
			shadowCoord.x < 0.99f && shadowCoord.y < 0.99f &&
			shadowCoord.z > 0.01f  && shadowCoord.z < 0.99f)
		{
            // const bool bReverseZOpen = pushConstants.bReverseZ != 0;
            const bool bReverseZOpen = true; // Now always reverse z
            
            const float pcfDilation = 1.0f;
            float bias = autoBias(VertexNoLSafe,cascadeIndex + 1, biasFactor);
            shadowCoord.z += bias;
            shadowFactor = shadowPcf(
                cascadeIndex,
                inShadowDepthBilinearTexture,
                shadowCoord,
                texelSize,
                pcfDilation,
                bReverseZOpen,
                NoLSafe
            );

            const float shadowCascadeBlendThreshold = 0.8f;
            vec2 posNdc = abs(shadowCoordNdc.xy);
            float cascadeFade = (max(posNdc.x,posNdc.y) - shadowCascadeBlendThreshold) * 4.0f;

            if (cascadeFade > 0.0f && cascadeIndex < cascadeCount - 1)
            {
                uint nextIndexCascade = cascadeIndex + 1;
                vec4 nextShadowClipPos = cascadeInfosbuffer[nextIndexCascade].cascadeViewProjMatrix * vec4(worldPosProcess, 1.0);
                vec4 nextShadowCoord = nextShadowClipPos / nextShadowClipPos.w;
                nextShadowCoord.st = nextShadowCoord.st * 0.5f + 0.5f;
                nextShadowCoord.y = 1.0f - nextShadowCoord.y;

                bias = autoBias(VertexNoLSafe,nextIndexCascade + 1, biasFactor);
                nextShadowCoord.z += bias;

                float nextShadowFactor = shadowPcf(
                    nextIndexCascade,
                    inShadowDepthBilinearTexture,
                    nextShadowCoord,
                    texelSize,
                    pcfDilation,
                    bReverseZOpen,
                    NoLSafe
                );

                shadowFactor = mix(shadowFactor, nextShadowFactor, cascadeFade);

                outIndex = cascadeIndex;
            }
            break;
        }
    }
    return shadowFactor;
}

layout(push_constant) uniform constants{   
   GPUPushConstants pushConstant;
};

vec3 getCascadeDebugColor(uint index)
{
    if(index == 0)
    {
        return vec3(1.0f,0.0f,0.0f);
    }
    else if(index == 1)
    {
        return vec3(0.0f,1.0f,0.0f);
    }
    else if(index == 2)
    {
        return vec3(0.0f,0.0f,1.0f);
    }
    else
    {
        return vec3(1.0f,0.0f,1.0f);
    }
}

vec3 PBRDirectLighting(float VoH, vec3 F0, float NoH, float NoL,float NoV, vec3 shadowTem, float roughness, float metal, 
vec3 baseColor, out vec3 diffuseItem, out vec3 specularItem)
{
    vec3 F    = FresnelSchlick(VoH, F0);
    float D   = DistributionGGX(NoH, roughness);   
    float G   = GeometrySmith(NoV, NoL, roughness);      
    vec3 kD = mix(vec3(1.0) - F, vec3(0.0), metal);

    vec3 diffuseBRDF = kD * baseColor;
    vec3 specularBRDF = (F * D * G) / max(0.00001, 4.0 * NoL * NoV);

    diffuseItem = diffuseBRDF * NoL * shadowTem;
    specularItem = specularBRDF * NoL * shadowTem;

    return (diffuseBRDF + specularBRDF) * NoL * shadowTem;
}

vec3 PBRAmbientLighting(float VoH, vec3 F0, float NoH, float NoL,float NoV, vec3 n, vec3 Lr, vec3 shadowTem, float roughness, 
float metal, vec3 baseColor, out vec3 diffuseItem, out vec3 specularItem)
{
    vec3 irradiance = texture(inIrradiancePrefilter, n).rgb;
    vec3 F = FresnelSchlick(NoV,F0);
    vec3 kd = mix(vec3(1.0) - F, vec3(0.0), metal);
    vec3 diffuseIBL = kd *baseColor * irradiance;

    int specularTextureLevels = textureQueryLevels(inEnvSpecularPrefilter);
    vec3 specularIrradiance = textureLod(inEnvSpecularPrefilter, Lr, roughness * specularTextureLevels).rgb;
    vec2 specularBRDF = texture(inBRDFLut, vec2(NoV, roughness)).rg;
    vec3 specularIBL = (F0 * specularBRDF.x + specularBRDF.y) * specularIrradiance;

    diffuseItem = diffuseIBL;
    specularItem = specularIBL;
    return diffuseIBL + specularIBL;
}


void main() 
{
    vec4 baseColorTex = tex(pushConstant.baseColorTexId, inUV0);
    if(baseColorTex.a < 0.01f)
    {
        discard;
    }

    vec3 worldSpaceNormal = normalize(inWorldNormal);

    vec3 l = normalize(-frameData.sunLightDir.xyz);
    vec3 n = worldSpaceNormal;
    vec3 v = normalize(frameData.camWorldPos.xyz - inWorldPos);
    vec3 h = normalize(v + l);

    float NoL = max(0.0f,dot(n, l));
    float LoH = max(0.0f,dot(l, h));
    float VoH = max(0.0f,dot(v, h));
    float NoV = max(0.0f,dot(n, v));
    float NoH = max(0.0f,dot(n, h));
    vec3 Lr   = 2.0 * NoV * n - v; // 反射向量

    const float metal = 0.0f;
    float roughness = 0.2f;
    vec3 F0   = mix(vec3(0.04f), baseColorTex.rgb, vec3(metal));
    

    float safeNoL = max(NoL, 0.0f);
    float safeNoH = max(NoH, 0.0f);

    float ln = clamp(NoL + 0.5, 0.0, 1.0);

    
    uint cascadeIndex = 0;
    float shadowDirect = evaluateDirectShadow(
        safeNoL, 
        inWorldPos,
        n,
        cascadeIndex,
        l,
        n
    );
    shadowDirect = clamp(shadowDirect + 0.5f, 0.0f, 1.0f);
    // shadowDirect = 1.0f;
    vec3 debugCascadeColor = getCascadeDebugColor(cascadeIndex);

    
    vec3 DiffuseColor = baseColorTex.rgb;
    float alpha =  baseColorTex.a;

    vec3 backLightDir = -l;
    float NoBL = max(0.0f, dot(backLightDir, n));

    if(pushConstant.shadingModelId == SHADING_MODEL_SSS)
    {
        vec3 basicDarkColor = vec3(1.0f, 0.5f, 0.5f) * 0.8;
        vec3 lightColor     = basicDarkColor * 2.2f;

        vec3 diffsueItem = mix(basicDarkColor, lightColor, pow(NoH,0.5f)) * baseColorTex.xyz;

        vec3 rimLight    = clamp(pow(1.0f - NoV, 4.0f) * pow(NoH, 4.f) - 0.01f, 0.0f, 1.0f) * lightColor * 20.0;

        outSSSSDiffuse.xyz = diffsueItem; 
        outSSSSDiffuse.w   = 1.0f;

        DiffuseColor =  rimLight;
    }
    else if(pushConstant.shadingModelId == SHADING_MODEL_EYE)
    {
        DiffuseColor = baseColorTex.rgb;
        outSSSSDiffuse = vec4(DiffuseColor, 1.0f);
    }
    else if(pushConstant.shadingModelId == SHADING_MODEL_HAIR)
    {
        vec3 rimLight    = clamp(pow(1.0f - NoV, 3.0f) * pow(NoH, 1.f) - 0.1f, 0.0f, 1.0f) * frameData.sunLightColor.xyz * 100.f;

        DiffuseColor =  (NoH + NoBL * 0.5f + gAmbient + 0.1f) * baseColorTex.rgb + rimLight;
        outSSSSDiffuse = vec4(0.0f,0.0f,0.0f,1.0f);
    }
    else
    {
        vec3 lightRadiance = frameData.sunLightColor.xyz; // todo: 
        roughness = 0.6f;
        vec3 diffsueItem; vec3 specularItem;

        vec3 directLighting = PBRDirectLighting(VoH, F0, NoH, NoL, NoV, lightRadiance, roughness, metal, baseColorTex.xyz,diffsueItem,specularItem);
        
        vec3 diffsueItemAmbinet; vec3 specularItemAmbinet;
        vec3 ambientLighting = PBRAmbientLighting(VoH, F0, NoH, NoL, NoV, n, Lr, lightRadiance, roughness, metal, baseColorTex.xyz,diffsueItemAmbinet,specularItemAmbinet);

        DiffuseColor = directLighting + ambientLighting + baseColorTex.rgb * (0.1f + NoBL * 0.5f);

        outSSSSDiffuse = vec4(0.0f,0.0f,0.0f,1.0f);
    }

    outShadingModel = float(pushConstant.shadingModelId) * SHADING_MODEL_SCALE;

    if(pushConstant.shadingModelId == SHADING_MODEL_pmxHairShadow)
    {
        discard;

        DiffuseColor = baseColorTex.rgb;
        outSSSSDiffuse = vec4(DiffuseColor, 1.0f);
        alpha = 0.0f;
    }

    outHdrColor = vec4(DiffuseColor, alpha);

    

    // remove camera jitter velocity factor. 
    vec2 curPosNDC = inCurPosNoJitter.xy  /  inCurPosNoJitter.w;
    vec2 prePosNDC = inPrevPosNoJitter.xy / inPrevPosNoJitter.w;
    outVelocity = (curPosNDC - prePosNDC) * 0.5f;
    outVelocity.y *= -1.0f;
}