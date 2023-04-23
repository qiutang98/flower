#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable

#include "atmosphere_common.glsl"
#include "../common/shared_shading_model.glsl"

//https://www.shadertoy.com/view/MdGSWy

#define ORB_FLARE_COUNT	6.0
#define DISTORTION_BARREL 1.0

vec2 GetDistOffset(vec2 uv, vec2 pxoffset)
{
    vec2 tocenter = uv.xy;
    vec3 prep = normalize(vec3(tocenter.y, -tocenter.x, 0.0));
    float angle = length(tocenter.xy) * 2.221 * DISTORTION_BARREL;
    vec3 oldoffset = vec3(pxoffset, 0.0);
    vec3 rotated = oldoffset * cos(angle) + cross(prep, oldoffset) * sin(angle) + prep * dot(prep, oldoffset) * (1.0 - cos(angle));
    return rotated.xy;
}

vec3 flare(vec2 uv, vec2 pos, float dist, float chromaOffset, float size)
{
    pos = GetDistOffset(uv, pos);
    float r = max(0.01 - pow(length(uv + (dist - chromaOffset) * pos), 2.4) *( 1.0 / (size * 2.0)), 0.0) * 0.85;
    float g = max(0.01 - pow(length(uv +  dist                 * pos), 2.4) * (1.0 / (size * 2.0)), 0.0) * 1.0;
    float b = max(0.01 - pow(length(uv + (dist + chromaOffset) * pos), 2.4) * (1.0 / (size * 2.0)), 0.0) * 1.5;
    return vec3(r, g, b);
}

vec3 orb(vec2 uv, vec2 pos, float dist, float size)
{
    vec3 c = vec3(0.0);

    for (float i = 0.0; i < ORB_FLARE_COUNT; i++)
    {
        float j = i + 1;
        float offset = j / (j + 0.1);
        float colOffset = j / ORB_FLARE_COUNT * 0.5;

        float ss = size / (j + 1.0);

        c += flare(uv, pos, dist + offset, ss * 2.0, ss) * vec3(1.0 - colOffset, 1.0, 0.5 + colOffset) * j;
    }

    c += flare(uv, pos, dist + 0.8, 0.05, 3.0 * size) * 0.5;
    return c;
}

vec3 ring(vec2 uv, vec2 pos, float dist, float chromaOffset, float blur)
{
    vec2 uvd = uv * length(uv);
    float r = max(1.0 / (1.0 + 250.0 * pow(length(uvd + (dist - chromaOffset) * pos), blur)), 0.0) * 0.8;
    float g = max(1.0 / (1.0 + 250.0 * pow(length(uvd +  dist                 * pos), blur)), 0.0) * 1.0;
    float b = max(1.0 / (1.0 + 250.0 * pow(length(uvd + (dist + chromaOffset) * pos), blur)), 0.0) * 1.5;
    return vec3(r, g, b);
}

vec3 LensFlare()
{
    vec2 coord = texcoord - 0.5;
    vec2 sunPos = sunCoord - 0.5;

    coord.x *= aspectRatio;
    sunPos.x *= aspectRatio;

    vec2 v = coord - sunPos;

    float dist = length(v);
    float fovFactor = max(gbufferProjection[1][1], 2.0);
    float gDist = dist * 13.0 / fovFactor;
    float phase = atan2(v) + 0.131;

    float gl = 2.0 - saturate(gDist) + sin(phase * 12.0) * saturate(gDist * 2.5 - 0.2);
    gl = gl * gl;
    gDist = gDist * gDist;
    gl *= 3e-4 / (gDist * gDist);

    float size = 0.5 * fovFactor;
    vec3 fl = vec3(0.0);

    fl += orb(coord, sunPos, 0.0, size * 0.02) * 0.15;
    fl += ring(coord, sunPos,  1.0, 0.02, 1.4) * 0.02;
    fl += ring(coord, sunPos, -1.0, 0.02, 1.4) * 0.01;

    fl += flare(coord, sunPos, -2.00, 0.05, size * 0.05) * 0.5;
    fl += flare(coord, sunPos, -0.90, 0.02, size * 0.03) * 0.25;
    fl += flare(coord, sunPos, -0.70, 0.01, size * 0.06) * 0.5;
    fl += flare(coord, sunPos, -0.55, 0.02, size * 0.02) * 0.25;
    fl += flare(coord, sunPos, -0.35, 0.02, size * 0.04) * 1.0;
    fl += flare(coord, sunPos, -0.25, 0.01, size * 0.15) * vec3(0.3, 0.4, 0.38);
    fl += flare(coord, sunPos, -0.25, 0.02, size * 0.08) * 0.3;
    fl += flare(coord, sunPos,  0.05, 0.01, size * 0.03) * 0.1;
    fl += flare(coord, sunPos,  0.30, 0.02, size * 0.20) * vec3(0.3, 0.25, 0.15);
    fl += flare(coord, sunPos,  1.20, 0.03, size * 0.10) * 0.5;

    vec3 lf = colorSunlight * (vec3(gl * GLARE_BRIGHTNESS) + fl * FLARE_BRIGHTNESS);
    return lf;
}

layout (local_size_x = 8, local_size_y = 8) in;
void main() 
{
    ivec2 colorSize = imageSize(imageHdrSceneColor);

    uvec2 groupThreadId = remap8x8(gl_LocalInvocationIndex);
    uvec2 dispatchId = groupThreadId + gl_WorkGroupID.xy * 8;

    ivec2 workPos = ivec2(dispatchId);

    if(workPos.x >= colorSize.x || workPos.y >= colorSize.y)
    {
        return;
    }

    const vec2 pixPos = vec2(workPos) + vec2(0.5f);
    const vec2 uv = pixPos / vec2(colorSize);

    AtmosphereParameters atmosphere = getAtmosphereParameters();

    // We are revert z.
    vec4 clipSpace = vec4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, 0.0, 1.0);
	vec4 viewPosH = frameData.camInvertProj * clipSpace;
    vec3 viewDir = viewPosH.xyz / viewPosH.w;
    vec3 worldDir = normalize((frameData.camInvertView * vec4(viewDir, 0.0)).xyz);
	vec3 worldPos = convertToAtmosphereUnit(frameData.camWorldPos.xyz) + vec3(0.0, atmosphere.bottomRadius, 0.0);

    float depthBufferValue = -1.0;
    float viewHeight = length(worldPos);

	vec3 L = vec3(0);
	depthBufferValue = texture(sampler2D(inDepth, pointClampEdgeSampler), uv).r;

    float shadingModelId = texture(sampler2D(inGBufferA, pointClampEdgeSampler), uv).a;
    const bool bShadingModelValid = isShadingModelValid(shadingModelId);

    const vec3 sunDirection = -normalize(frameData.sky.direction);

    const bool bUnderAtmosphere =  viewHeight < atmosphere.topRadius;

    vec3 upVector = normalize(worldPos);

    // Back ground and under atmosphere pixel, sample sky view lut.
    if (bUnderAtmosphere && (!bShadingModelValid))
	{
        float viewZenithCosAngle = dot(worldDir, upVector);

        // Assumes non parallel vectors
		vec3 sideVector = normalize(cross(upVector, worldDir));		

        // aligns toward the sun light but perpendicular to up vector
		vec3 forwardVector = normalize(cross(sideVector, upVector));	

		vec2 lightOnPlane = vec2(dot(sunDirection, forwardVector), dot(sunDirection, sideVector));
		lightOnPlane = normalize(lightOnPlane);
		float lightViewCosAngle = lightOnPlane.x;

		bool bIntersectGround = raySphereIntersectNearest(worldPos, worldDir, vec3(0.0), atmosphere.bottomRadius) >= 0.0f;

        vec2 sampleUv;
		skyViewLutParamsToUv(atmosphere, bIntersectGround, viewZenithCosAngle, lightViewCosAngle, viewHeight, vec2(textureSize(inSkyViewLut, 0)), sampleUv);

		vec3 luminance = texture(sampler2D(inSkyViewLut, linearClampEdgeSampler), sampleUv).rgb;

		imageStore(imageHdrSceneColor, workPos, vec4(luminance, 1.0f));
        return;
	}

    float opacity = 0.0;
    if(bUnderAtmosphere) // Composite air perspective.
    {
        // Exist pre-compute data, sample it.

        // Build world position.
        clipSpace.z = depthBufferValue;
        vec4 depthBufferWorldPos = frameData.camInvertViewProj * clipSpace;
        depthBufferWorldPos.xyz /= depthBufferWorldPos.w;

        float tDepth = length((depthBufferWorldPos.xyz * 0.001) - (worldPos + vec3(0.0, -atmosphere.bottomRadius, 0.0))); // meter -> kilometers.
        float slice = aerialPerspectiveDepthToSlice(tDepth);

        float weight = 1.0;
        if (slice < 0.5)
        {
            // We multiply by weight to fade to 0 at depth 0. That works for luminance and opacity.
            weight = saturate(slice * 2.0);
            slice = 0.5;
        }
        ivec3 sliceLutSize = textureSize(inFroxelScatter, 0);
        float w = sqrt(slice / float(sliceLutSize.z));	// squared distribution

        const vec4 airPerspective = weight * texture(sampler3D(inFroxelScatter, linearClampEdgeSampler), vec3(uv, w));
        L.rgb += airPerspective.rgb;
        opacity = airPerspective.a;
    }
    else if(!bShadingModelValid)
    {
        // No precompute data can use. compute new data.

        // Move to top atmosphere as the starting point for ray marching.
        // This is critical to be after the above to not disrupt above atmosphere tests and voxel selection.
        if (!moveToTopAtmosphere(worldPos, worldDir, atmosphere.topRadius))
        {
            // Ray is not intersecting the atmosphere, return.	
            vec3 srcColor = imageLoad(imageHdrSceneColor, workPos).rgb;
            imageStore(imageHdrSceneColor, workPos, vec4(srcColor, 1.0f));
            return;
        }

        const bool bGround = false;
        const float sampleCountIni = 0.0;
        const bool bVariableSampleCount = true;
        const bool bMieRayPhase = true;
        const float tMaxMax = kDefaultMaxT;
        depthBufferValue = -1.0;
        SingleScatteringResult ss = integrateScatteredLuminance(
            pixPos, 
            worldPos, 
            worldDir, 
            sunDirection, 
            atmosphere, 
            bGround, 
            sampleCountIni, 
            depthBufferValue, 
            bMieRayPhase,
            tMaxMax,
            bVariableSampleCount
        );

        L += ss.scatteredLight;
        vec3 throughput = ss.transmittance;

        const float transmittance = mean(throughput);
        opacity = 1.0 - transmittance;
    }

    vec3 srcColor = imageLoad(imageHdrSceneColor, workPos).rgb;
    vec3 outColor = L.rgb + (1.0 - opacity) * srcColor;


    imageStore(imageHdrSceneColor, workPos, vec4(outColor, 1.0f));
}