#ifndef POST_COMMON_GLSL
#define POST_COMMON_GLSL

#include "../common/shared_functions.glsl"

vec3 toneMapperFunction(vec3 inColor)
{
    vec3 colorAP0 = inColor * sRGB_2_AP0;
    vec3 toneColor = acesFilm(colorAP0);
    return encodeSRGB(toneColor);
}

#define kFusionGaussianTapRadius 2.0
#define kFusionGaussianTapNum    5
const float kFusionGaussianWeight[kFusionGaussianTapNum] = 
{
    0.0625, 0.25, 0.375, 0.25, 0.0625
};

#endif