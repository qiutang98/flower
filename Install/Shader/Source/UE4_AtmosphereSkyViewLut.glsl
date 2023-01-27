#version 460

/*
** Physical based render code, develop by engineer: qiutanguu.
*/

#extension GL_GOOGLE_include_directive : enable

#include "UE4_AtmosphereCommon.glsl"

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
	    vec3 worldPos = convertToAtmosphereUnit(viewData.camWorldPos.xyz) + vec3(0.0, atmosphere.bottomRadius, 0.0);
        vec3 scatteredLight = getPosScatterLight(atmosphere, worldPos, uv, true, pixPos);
        imageStore(imageSkyViewLut, workPos, vec4(scatteredLight, 1.0f));
    }

    // TODO: These cloud lookup just need compute once when atmosphere change or cloud change.
    {
        vec3 worldPosCloudTop = vec3(0.0, atmosphere.cloudAreaStartHeight + atmosphere.cloudAreaThickness, 0.0);
        worldPosCloudTop.y = max(worldPosCloudTop.y, atmosphere.bottomRadius +  1e-3f);

        vec3 scatteredLightCloudTop = getPosScatterLight(atmosphere, worldPosCloudTop, uv, true, pixPos);
        imageStore(imageSkyViewLutCloudTop, workPos, vec4(scatteredLightCloudTop, 1.0f));
    }
    {
        vec3 worldPosCloudBottom = vec3(0.0, atmosphere.cloudAreaStartHeight, 0.0);
        worldPosCloudBottom.y = max(worldPosCloudBottom.y, atmosphere.bottomRadius + 1e-3f);

        vec3 scatteredLightCloudBottom = getPosScatterLight(atmosphere, worldPosCloudBottom, uv, true, pixPos);
        imageStore(imageSkyViewLutCloudBottom, workPos, vec4(scatteredLightCloudBottom, 1.0f));
    }

}