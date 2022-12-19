#version 460

/*
** Physical based render code, develop by engineer: qiutanguu.
*/

#extension GL_EXT_samplerless_texture_functions : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_shader_subgroup_basic : enable

#include "SDSM_Common.glsl"

layout (set = 1, binding = 0) uniform UniformView { ViewData viewData; };
layout (set = 2, binding = 0) uniform UniformFrame { FrameData frameData; };

float logCascadeSplit(
    in const float nearZ, 
    in const float farDepthPlane, 
    in const float nearDepthPlane, 
    in const float clipRange, 
    in const uint  cascadeId)
{
    float range = farDepthPlane - nearDepthPlane;
    float ratio = farDepthPlane / nearDepthPlane;

    // get current part factor.
    float p = float(cascadeId + 1) / float(frameData.directionalLight.cascadeCount);

    // get log scale factor and uniform scale factor.
    float logScale = nearDepthPlane * pow(abs(ratio), p);
    float uniformScale = nearDepthPlane + range * p;

    // final get split distance.
    float lambda = frameData.directionalLight.splitLambda;

    float d = lambda * (logScale - uniformScale) + uniformScale;
    return (d - nearZ) / clipRange;
}

// This pass build sdsm cascade info.
layout(local_size_x = 32) in; 
void main()
{
    const uint idx = gl_GlobalInvocationID.x;
    const uint cascadeId  = idx;

    if(cascadeId >= frameData.directionalLight.cascadeCount)
    {
        return;
    }

    // camera info get.
    const float nearZ = viewData.camInfo.z;
    const float farZ  = viewData.camInfo.w;

    const float minDepth = uintDepthUnpack(depthRange.minDepth);
    const float maxDepth = uintDepthUnpack(depthRange.maxDepth);

    // We reverse z, so min dpeth value is far plane, max depth value is near plane.
    float nearPlaneLinear = linearizeDepth(maxDepth, nearZ, farZ);
    float farPlaneLinear = linearizeDepth(minDepth, nearZ, farZ);
    farPlaneLinear = min(farPlaneLinear, nearPlaneLinear + frameData.directionalLight.maxDrawDepthDistance);

    // Get depth start and end pos which in range [0, 1].
    const float clipRange = farZ - nearZ;
    float depthStartPos = clamp((nearPlaneLinear - nearZ) / clipRange, .0f, 1.f);
    float depthEndPos   = clamp((farPlaneLinear - nearZ) / clipRange, .0f, 1.f);

    // Now setup each cascade frustum corners.
    vec3 frustumCornersWS[8];
    frustumCornersWS[0] = vec3(-1.0f,  1.0f, 1.0f);
    frustumCornersWS[1] = vec3( 1.0f,  1.0f, 1.0f);
    frustumCornersWS[2] = vec3( 1.0f, -1.0f, 1.0f);
    frustumCornersWS[3] = vec3(-1.0f, -1.0f, 1.0f);
    frustumCornersWS[4] = vec3(-1.0f,  1.0f, 0.0f);
    frustumCornersWS[5] = vec3( 1.0f,  1.0f, 0.0f);
    frustumCornersWS[6] = vec3( 1.0f, -1.0f, 0.0f);
    frustumCornersWS[7] = vec3(-1.0f, -1.0f, 0.0f);
    for(uint i = 0; i < 8; i ++)
    {
        vec4 invCorner = viewData.camInvertViewProj * vec4(frustumCornersWS[i], 1.0f);
        frustumCornersWS[i] = invCorner.xyz / invCorner.w;
    }

    const vec3 upDir = vec3(0.f, 1.f, 0.f);
    const vec3 lightDir = frameData.directionalLight.direction;

    // Prev split.
    float prevSplitDist = (cascadeId == 0) ? 
        depthStartPos :
        logCascadeSplit(nearZ, farPlaneLinear, nearPlaneLinear, clipRange, cascadeId - 1);

    // Current split.
    float splitDist = logCascadeSplit(nearZ, farPlaneLinear, nearPlaneLinear, clipRange, cascadeId);

    // Calculate 4 corner world pos of cascade view frustum.
    for(uint i = 0; i < 4; i ++)
    {
        vec3 cornerRay = frustumCornersWS[i + 4] - frustumCornersWS[i]; // distance ray.

        vec3 nearCornerRay = cornerRay * prevSplitDist;
        vec3 farCornerRay  = cornerRay * splitDist;

        frustumCornersWS[i + 4] = frustumCornersWS[i] + farCornerRay;
        frustumCornersWS[i + 0] = frustumCornersWS[i] + nearCornerRay;
    }

    // Calculate center pos of view frustum.
    vec3 frustumCenter = vec3(0.0f);
    for(uint i = 0; i < 8; i ++)
    {
        frustumCenter += frustumCornersWS[i];
    }
    frustumCenter /= 8.0f;

    // Get view sphere bounds radius.
    float sphereRadius = 0.0f;
    for(uint i = 0; i < 8; ++i)
    {
        float dist = length(frustumCornersWS[i] - frustumCenter);
        sphereRadius = max(sphereRadius, dist);
    }

    // Round 16.
    sphereRadius = ceil(sphereRadius * 16.0f) / 16.0f;
    vec3 maxExtents = vec3(sphereRadius);
    vec3 minExtents = -maxExtents;
    vec3 cascadeExtents = maxExtents - minExtents;

    // create temporary view project matrix for cascade.
    vec3 shadowCameraPos = frustumCenter - normalize(lightDir) * cascadeExtents.z * 0.5f;

    float nearZProj = 0.0f; 
    float farZProj  = cascadeExtents.z;     

    mat4 shadowView = lookAtRH(shadowCameraPos,frustumCenter,upDir);
    mat4 shadowProj = orthoRHZeroOne(
        minExtents.x, 
        maxExtents.x, 
        minExtents.y, 
        maxExtents.y,
        farZProj, // Also reverse z for shadow depth.
        nearZProj
    );

    // Texel align.
    const float sMapSize = float(frameData.directionalLight.perCascadeXYDim);
    mat4 shadowViewProjMatrix = shadowProj * shadowView;
    vec4 shadowOrigin = vec4(0.0f,0.0f,0.0f,1.0f);
    shadowOrigin = shadowViewProjMatrix * shadowOrigin;
    shadowOrigin *= (sMapSize / 2.0f);  
    
    // Move to center uv pos
    vec3 roundedOrigin = round(shadowOrigin.xyz);
    vec3 roundOffset = roundedOrigin - shadowOrigin.xyz;
    roundOffset   = roundOffset * (2.0f / sMapSize);
    roundOffset.z = 0.0f;

    // Push back round offset data to project matrix.
    shadowProj[3][0] += roundOffset.x;
    shadowProj[3][1] += roundOffset.y;

    // Final proj view matrix
    mat4 shadowFinalViewProj = shadowProj * shadowView;

    // push to buffer.
    cascadeInfos[cascadeId].viewProj = shadowFinalViewProj;
    mat4 reverseToWorld = inverse(shadowFinalViewProj);

    // Build frustum plane.
    vec3 p[8];
    {
        p[0] = vec3(-1.0f,  1.0f, 1.0f);
        p[1] = vec3( 1.0f,  1.0f, 1.0f);
        p[2] = vec3( 1.0f, -1.0f, 1.0f);
        p[3] = vec3(-1.0f, -1.0f, 1.0f);
        p[4] = vec3(-1.0f,  1.0f, 0.0f);
        p[5] = vec3( 1.0f,  1.0f, 0.0f);
        p[6] = vec3( 1.0f, -1.0f, 0.0f);
        p[7] = vec3(-1.0f, -1.0f, 0.0f);

        for(uint i = 0; i < 8; i++)
        {
            vec4 invCorner = reverseToWorld * vec4(p[i], 1.0f);
            p[i] = invCorner.xyz / invCorner.w;
        }

        // left
        vec3 leftN = normalize(cross((p[4] - p[7]), (p[3] - p[7])));
        cascadeInfos[cascadeId].frustumPlanes[0] = vec4(leftN, -dot(leftN, p[7]));

        // down
        vec3 downN = normalize(cross((p[6] - p[2]), (p[3] - p[2])));
        cascadeInfos[cascadeId].frustumPlanes[1] = vec4(downN, -dot(downN, p[2]));
        
        // right
        vec3 rightN = normalize(cross((p[6] - p[5]), (p[1] - p[5])));
        cascadeInfos[cascadeId].frustumPlanes[2] = vec4(rightN, -dot(rightN, p[5]));
        
        // top
        vec3 topN = normalize(cross((p[5] - p[4]), (p[0] - p[4])));
        cascadeInfos[cascadeId].frustumPlanes[3] = vec4(topN, -dot(topN, p[4]));

        // front
        vec3 frontN = normalize(cross((p[1] - p[0]), (p[3] - p[0])));
        cascadeInfos[cascadeId].frustumPlanes[4] = vec4(frontN, -dot(frontN, p[0]));

        // back
        vec3 backN  = normalize(cross((p[5] - p[6]), (p[7] - p[6])));
        cascadeInfos[cascadeId].frustumPlanes[5] = vec4(backN, -dot(frontN, p[6]));
    }

    groupMemoryBarrier();
    barrier();

    mat4 coarseShadowMapVP = cascadeInfos[frameData.directionalLight.cascadeCount - 1].viewProj;

    {
        // Construct cascade shadow map corner position and reproject to world space.
        vec3 worlPosition00 = constructPos(vec2(0.0, 0.0), 1.0, reverseToWorld); // reverse z.
        vec3 worlPosition11 = constructPos(vec2(1.0, 1.0), 0.0, reverseToWorld); // reverse z.

        // Project to coarse shadow map uv space.
        vec4 v00 = coarseShadowMapVP * vec4(worlPosition00, 1.0f);
        v00.xyz /= v00.w;
        v00.xy = v00.xy * 0.5f + 0.5f;
        v00.y = 1.0f - v00.y;

        // Project to coarse shadow map uv space.
        vec4 v11 = coarseShadowMapVP * vec4(worlPosition11, 1.0f);
        v11.xyz /= v11.w;
        v11.xy = v11.xy * 0.5f + 0.5f;
        v11.y = 1.0f - v11.y;

        // Scale filter size on accurate cascade.
        cascadeInfos[cascadeId].cascadeScale = vec4(1.0f / abs(v11.xy - v00.xy), 0.0, 0.0); 
    }
}