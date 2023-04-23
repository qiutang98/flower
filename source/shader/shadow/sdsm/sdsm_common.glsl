#ifndef SDSM_COMMON_GLSL
#define SDSM_COMMON_GLSL

// My personal sample distribution shadow map tech implement.
// Total four pass.
// pass #0. evaluate scene depth z min max uint value. See SDSMDepthRange.glsl file.
// pass #1. build shadow project matrix for every cascade. See SDSMPrepareCascade.glsl file.
// pass #2. culling each cascade draw call. See SDSMCulling.glsl file.
// pass #3. shadow depth drawing for this directional light. See SDSMDepth.glsl file.
// pass #4: eavluate soft shadow attention commonly. See SDSMEvaluateSoftShadow.glsl file.

#include "../../common/shared_functions.glsl"

struct DepthRange
{
    uint minDepth;
    uint maxDepth;
};

layout(set = 0, binding = 0) uniform texture2D inDepth; // Depth z.
layout(set = 0, binding = 1) buffer SSBODepthRangeBuffer { DepthRange depthRange; }; // Depth range min max buffer
layout(set = 0, binding = 2) buffer SSBOCascadeInfoBuffer{ CascadeInfo cascadeInfos[]; }; // Cascade infos.
layout(set = 0, binding = 3) uniform texture2D inGbufferA;
layout(set = 0, binding = 4) uniform texture2D inSDSMShadowDepth;
layout(set = 0, binding = 5) uniform texture2D inGbufferB;
layout(set = 0, binding = 6) uniform texture2D inGbufferS;
layout(set = 0, binding = 7, r8) uniform image2D imageShadowMask;
layout(set = 0, binding = 8) uniform texture2D inHeightmap; 
layout(set = 0, binding = 9) uniform UniformFrameData { PerFrameData frameData; };
layout(set = 0, binding = 10) buffer SSBOPerObject { StaticMeshPerObjectData objectDatas[]; };
layout(set = 0, binding = 11) buffer SSBOIndirectDraws { StaticMeshDrawCommand indirectCommands[]; };
layout(set = 0, binding = 12) buffer SSBODrawCount{ uint drawCount[]; };

layout (push_constant) uniform PushConsts 
{  
    // For culling.
    uint cullCountPercascade;
    uint cascadeCount; 

    // For draw.
    uint cascadeId;
    uint perCascadeMaxCount;

    uint bHeightmapValid;
    float heightfiledDump;
};

uint depthPackUnit(float depth)
{
    return floatBitsToUint(depth);
}

float uintDepthUnpack(uint uintDepth)
{
    return uintBitsToFloat(uintDepth);
}

// RH look at function for compute shadow camera eye matrix.
mat4 lookAtRH(vec3 eye,vec3 center,vec3 up)
{
    const vec3 f = normalize(center - eye);
    const vec3 s = normalize(cross(f, up));
    const vec3 u = cross(s, f);

    mat4 ret =  
    {
        {1.0f,0.0f,0.0f,0.0f},
        {0.0f,1.0f,0.0f,0.0f},
        {0.0f,0.0f,1.0f,0.0f},
        {1.0f,0.0f,0.0f,1.0f}
    };

    ret[0][0] = s.x; ret[0][1] = u.x; ret[0][2] =-f.x; ret[3][0] =-dot(s, eye);
    ret[1][0] = s.y; ret[1][1] = u.y; ret[1][2] =-f.y; ret[3][1] =-dot(u, eye);
    ret[2][0] = s.z; ret[2][1] = u.z; ret[2][2] =-f.z; ret[3][2] = dot(f, eye);

    return ret;
}

// RH ortho projection function for light matrix.
mat4 orthoRHZeroOne(float left, float right, float bottom, float top, float zNear, float zFar)
{
    mat4 ret =  
    {
        {1.0f,0.0f,0.0f,0.0f},
        {0.0f,1.0f,0.0f,0.0f},
        {0.0f,0.0f,1.0f,0.0f},
        {1.0f,0.0f,0.0f,1.0f}
    };

    ret[0][0] =   2.0f / (right - left);
    ret[1][1] =   2.0f / (top - bottom);
    ret[2][2] =  -1.0f / (zFar - zNear);
    ret[3][0] = -(right + left) / (right - left);
    ret[3][1] = -(top + bottom) / (top - bottom);
    ret[3][2] = -zNear / (zFar - zNear);

	return ret;
}

#endif