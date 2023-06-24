#ifndef SHARED_SHADING_MODEL_GLSL
#define SHARED_SHADING_MODEL_GLSL

// For 8-bit unorm texture, float error range = 1.0 / 255.0 = 0.004
#define kShadingModelRangeCheck        0.005

// Shading model count is 50, Step value is 0.02
#define kShadingModelUnvalid           0.00
#define kShadingModelStandardPBR       0.02
#define kShadingModelPMXBasic          0.04
#define kShadingModelSSSS              0.06
#define kShadingModelEye               0.08

bool isInShadingModelRange(float v, float shadingModel)
{
    return (v > (shadingModel - kShadingModelRangeCheck)) && (v < (shadingModel + kShadingModelRangeCheck));
}

bool isShadingModelValid(float v)
{
    return v > (kShadingModelUnvalid + kShadingModelRangeCheck);
}

bool isPMXMeshShadingModelCharacter(float v)
{
    return isInShadingModelRange(v, kShadingModelPMXBasic);
}

#endif