#ifndef SHARED_CBT_GLSL
#define SHARED_CBT_GLSL

#define CBT_SET_INDEX 0
#define CBT_BINDING_INDEX 0
#include "cbt.glsl"

layout (push_constant) uniform PushConsts 
{  
    int u_CbtID;
    int u_PassID;
};

#endif