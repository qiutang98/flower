#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable

#include "post_common.glsl"

layout (set = 0, binding = 0) uniform texture2D inGBufferB;
layout (set = 0, binding = 1, rgba16f) uniform image2D outImage;
layout (set = 0, binding = 2) uniform UniformFrameData { PerFrameData frameData; };
layout (set = 0, binding = 3) uniform texture2D inGBufferA;
layout (set = 0, binding = 4) uniform texture2D inDepth;
layout(set = 0, binding = 5) buffer SSBODepthRangeBuffer { DepthRange depthRange; }; 

#define SHARED_SAMPLER_SET 1
#include "../common/shared_sampler.glsl"

layout (push_constant) uniform PushConsts 
{  
    int kContourMethod;
    float kNormalDiffCoeff;
    float kDepthDiffCoeff;
};

float Fdepth(in float Z, in float zNear, in float zFar)
{
  return abs((1. / Z - 1. / zNear) / ((1. / zFar) - (1. / zNear)));
}

layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    ivec2 colorSize = imageSize(outImage);

    uvec2 groupThreadId = remap8x8(gl_LocalInvocationIndex);
    uvec2 dispatchId = groupThreadId + gl_WorkGroupID.xy * 8;
    ivec2 workPos = ivec2(dispatchId);
    if(workPos.x >= colorSize.x || workPos.y >= colorSize.y)
    {
        return;
    }
    const vec2 uv = (vec2(workPos) + vec2(0.5f)) / vec2(colorSize);

    vec3 baseColor = texelFetch(inGBufferA, workPos, 0).xyz;


    vec3 outColor = vec3(0.0);

    vec4 A_Src = texelFetchOffset(inGBufferB, workPos, 0, ivec2(-1.0, +1.0));  //  +---+---+---+
    vec4 B_Src = texelFetchOffset(inGBufferB, workPos, 0, ivec2(+0.0, +1.0));  //  | A | B | C |
    vec4 C_Src = texelFetchOffset(inGBufferB, workPos, 0, ivec2(+1.0, +1.0));  //  +---+---+---+
    vec4 D_Src = texelFetchOffset(inGBufferB, workPos, 0, ivec2(-1.0, +0.0));  //  | D | X | E |
    vec4 X_Src = texelFetchOffset(inGBufferB, workPos, 0, ivec2(+0.0, +0.0));  //  +---+---+---+
    vec4 E_Src = texelFetchOffset(inGBufferB, workPos, 0, ivec2(+1.0, +0.0));  //  | F | G | H |
    vec4 F_Src = texelFetchOffset(inGBufferB, workPos, 0, ivec2(-1.0, -1.0));  //  +---+---+---+
    vec4 G_Src = texelFetchOffset(inGBufferB, workPos, 0, ivec2(+0.0, -1.0));
    vec4 H_Src = texelFetchOffset(inGBufferB, workPos, 0, ivec2(+1.0, -1.0));



    // floatBitsToUint
    int A = int(A_Src.w);  //  +---+---+---+
    int B = int(B_Src.w);  //  | A | B | C |
    int C = int(C_Src.w);  //  +---+---+---+
    int D = int(D_Src.w);  //  | D | X | E |
    int X = int(X_Src.w);  //  +---+---+---+
    int E = int(E_Src.w);  //  | F | G | H |
    int F = int(F_Src.w);  //  +---+---+---+
    int G = int(G_Src.w);
    int H = int(H_Src.w);

    switch(kContourMethod)
    {
        case 0:  // smaller
        if(X < A || X < B || X < C || X < D || X < E || X < F || X < G || X < H)
        {
            outColor = vec3(1);
        }
        break;
        case 1:  // bigger
        if(X > A || X > B || X > C || X > D || X > E || X > F || X > G || X > H)
        {
            outColor = vec3(1);
        }
        break;
        case 2:  // thicker
        if(X != A || X != B || X != C || X != D || X != E || X != F || X != G || X != H)
        {
            outColor = vec3(1);
        }
        case 3:  // different
            outColor = vec3((int(X != A) + int(X != C) + int(X != F) + int(X != H)) * (1. / 6.) + 
                            (int(X != B) + int(X != D) + int(X != E) + int(X != G)) * (1. / 3.));
        break;
    }

    vec3 An = A_Src.xyz;
    vec3 Bn = B_Src.xyz;
    vec3 Cn = C_Src.xyz;
    vec3 Dn = D_Src.xyz;
    vec3 Xn = X_Src.xyz;
    vec3 En = E_Src.xyz;
    vec3 Fn = F_Src.xyz;
    vec3 Gn = G_Src.xyz;
    vec3 Hn = H_Src.xyz;

    // Normal Gradient
    float Ngrad = 0;
    {
        // compute length of gradient using Sobel/Kroon operator
        const float k0     = 17. / 23.75;
        const float k1     = 61. / 23.75;
        const vec3  grad_y = k0 * An + k1 * Bn + k0 * Cn - k0 * Fn - k1 * Gn - k0 * Hn;
        const vec3  grad_x = k0 * Cn + k1 * En + k0 * Hn - k0 * An - k1 * Dn - k0 * Fn;
        const float g      = length(grad_x) + length(grad_y);

        Ngrad = smoothstep(2.f, 3.f, g * kNormalDiffCoeff);  //!! magic
    }

    const float zNear = uintDepthUnpack(depthRange.minDepth);
    const float zFar = uintDepthUnpack(depthRange.maxDepth);

    float A_D = Fdepth(texelFetchOffset(inDepth, workPos, 0, ivec2(-1.0, +1.0)).x, zNear, zFar);  //  +---+---+---+
    float B_D = Fdepth(texelFetchOffset(inDepth, workPos, 0, ivec2(+0.0, +1.0)).x, zNear, zFar);  //  | A | B | C |
    float C_D = Fdepth(texelFetchOffset(inDepth, workPos, 0, ivec2(+1.0, +1.0)).x, zNear, zFar);  //  +---+---+---+
    float D_D = Fdepth(texelFetchOffset(inDepth, workPos, 0, ivec2(-1.0, +0.0)).x, zNear, zFar);  //  | D | X | E |
    float X_D = Fdepth(texelFetchOffset(inDepth, workPos, 0, ivec2(+0.0, +0.0)).x, zNear, zFar);  //  +---+---+---+
    float E_D = Fdepth(texelFetchOffset(inDepth, workPos, 0, ivec2(+1.0, +0.0)).x, zNear, zFar);  //  | F | G | H |
    float F_D = Fdepth(texelFetchOffset(inDepth, workPos, 0, ivec2(-1.0, -1.0)).x, zNear, zFar);  //  +---+---+---+
    float G_D = Fdepth(texelFetchOffset(inDepth, workPos, 0, ivec2(+0.0, -1.0)).x, zNear, zFar); 
    float H_D = Fdepth(texelFetchOffset(inDepth, workPos, 0, ivec2(+1.0, -1.0)).x, zNear, zFar); 

    float Dgrad = 0;
    {
        float g = (abs(A_D + 2 * B_D + C_D - F_D - 2 * G_D - H_D) + abs(C_D + 2 * E_D + H_D - A_D - 2 * D_D - F_D)) / 8.0;
        float l = (8 * X_D - A_D - B_D - C_D - D_D - E_D - F_D - G_D - H_D) / 3.0;

        Dgrad = (l + g) * kDepthDiffCoeff;
        Dgrad = smoothstep(0.03f, 0.1f, Dgrad);  // !magic values
    }

    vec4 result = imageLoad(outImage, workPos);
    result.xyz *= vec3(1.0) - outColor;

    result.xyz *= 1.0 - (Ngrad + Dgrad);

    // Final store.
    imageStore(outImage, workPos, result);
}