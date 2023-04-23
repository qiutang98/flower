#ifndef SHARED_SHADING_MODEL_GLSL
#define SHARED_SHADING_MODEL_GLSL

// For 8-bit unorm texture, float error range = 1.0 / 255.0 = 0.004
#define kShadingModelRangeCheck        0.005

// Shading model count is 50, Step value is 0.02
#define kShadingModelUnvalid           0.0
#define kShadingModelStandardPBR       0.02

bool isInShadingModelRange(float v, float shadingModel)
{
    return (v > (shadingModel - kShadingModelRangeCheck)) && (v < (shadingModel + kShadingModelRangeCheck));
}

bool isShadingModelValid(float v)
{
    return v > (kShadingModelUnvalid + kShadingModelRangeCheck);
}

#endif