#version 460

#extension GL_GOOGLE_include_directive : enable

#define NO_MULTISCATAPPROX_ENABLED
#include "UE4_AtmosphereCommon.glsl"


layout (local_size_x = 8, local_size_y = 8) in;
void main() 
{
    ivec2 lutSize = imageSize(imageTransmittanceLut);
    ivec2 workPos = ivec2(gl_GlobalInvocationID.xy);

    AtmosphereParameters atmosphere = getAtmosphereParameters();

    const vec2 pixPos = vec2(workPos) + vec2(0.5f);
    const vec2 uv = pixPos / vec2(lutSize);

    float viewHeight;
	float viewZenithCosAngle;
	uvToLutTransmittanceParams(atmosphere, viewHeight, viewZenithCosAngle, uv);

    
    const vec3 worldPos = vec3(0.0f, viewHeight, 0.0f);
    const vec3 worldDir = vec3(0.0f, viewZenithCosAngle, sqrt(1.0 - viewZenithCosAngle * viewZenithCosAngle));
    const vec3 sunDir = -normalize(frameData.directionalLight.direction);
    const bool bGround = false;
    const float sampleCountIni = 40.0;
    const float depthBufferValue = -1.0;
    const bool bMieRayPhase = false;
    const float tMaxMax = kDefaultMaxT;
    const bool bVariableSampleCount = false;
    vec3 opticalDepth = integrateScatteredLuminance(
        pixPos, 
        worldPos, 
        worldDir, 
        sunDir, 
        atmosphere,
        bGround, 
        sampleCountIni, 
        depthBufferValue, 
        bMieRayPhase, 
        tMaxMax,
        bVariableSampleCount
    ).opticalDepth;

	vec3 transmittance = exp(-opticalDepth);
    transmittance = min(transmittance, vec3(kMaxHalfFloat));
	imageStore(imageTransmittanceLut, workPos, vec4(transmittance, 1.0f));
}