#version 460
#extension GL_GOOGLE_include_directive : enable

#include "atmosphere_common.glsl"

layout (local_size_x = 8, local_size_y = 8) in;
void main() 
{
    ivec2 lutSize = imageSize(imageSkyViewLut);
    ivec2 workPos = ivec2(gl_GlobalInvocationID.xy);

    if(workPos.x >= lutSize.x || workPos.y >= lutSize.y)
    {
        return;
    }

    const vec2 pixPos = vec2(workPos) + vec2(0.5f);
    const vec2 uv = pixPos / vec2(lutSize);

    AtmosphereParameters atmosphere = getAtmosphereParameters();

    {
	    vec3 worldPos = convertToAtmosphereUnit(frameData.camWorldPos.xyz) + vec3(0.0, atmosphere.bottomRadius, 0.0);
        vec3 scatteredLight = getPosScatterLight(atmosphere, worldPos, uv, true, pixPos);
        imageStore(imageSkyViewLut, workPos, vec4(scatteredLight, 1.0f));
    }
}