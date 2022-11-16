#version 460

#extension GL_GOOGLE_include_directive : enable

#include "SDSM_Common.glsl"
#include "StaticMeshCommon.glsl"

layout (set = 1, binding = 0) readonly buffer SSBOPerObject { PerObjectData objectDatas[]; };
layout (set = 2, binding = 0) buffer SSBOIndirectDraws { DrawIndirectCommand indirectCommands[]; };
layout (set = 3, binding = 0) buffer SSBODrawCount{ DrawIndirectCount drawCount[]; };

layout (push_constant) uniform PushConsts 
{  
    uint cullCountPercascade;
    uint cascadeCount; 
};

void visibileCulling(uint idx, uint cascadeId)
{
    PerObjectData objectData = objectDatas[idx];

	vec3 localPos = objectData.sphereBounds.xyz;
	vec4 worldPos = objectData.modelMatrix * vec4(localPos, 1.0f);

	// local to world normal matrix.
	mat3 normalMatrix = transpose(inverse(mat3(objectData.modelMatrix)));
	mat3 world2Local = inverse(normalMatrix);

	// frustum test.
	for (int i = 0; i < 4; i++) // frustum 4, 5 is back and front face, don't test.
	{
        vec3 worldSpaceN = cascadeInfos[cascadeId].frustumPlanes[i].xyz;
        float castDistance = dot(worldPos.xyz, worldSpaceN);

		// transfer to local matrix and use abs get first dimensions project value,
		// use that for test.
		vec3 localNormal = world2Local * worldSpaceN;
		float absDiff = dot(abs(localNormal), objectData.extents.xyz);
		if (castDistance + absDiff + cascadeInfos[cascadeId].frustumPlanes[i].w < 0.0)
		{
			return;
		}
	}
    
    // Build draw command if visible.
    uint drawId = atomicAdd(drawCount[cascadeId].count, 1) + cascadeId * cullCountPercascade;
    indirectCommands[drawId].objectId = idx;

    // We fetech vertex by index, so vertex count is index count.
    indirectCommands[drawId].vertexCount = objectData.indexCount;
    indirectCommands[drawId].firstVertex = objectData.indexStartPosition;

    // We fetch vertex in vertex shader, so instancing is unused when rendering.
    indirectCommands[drawId].instanceCount = 1;
    indirectCommands[drawId].firstInstance = 0; 
}

layout (local_size_x = 64) in;
void main()
{
    uint idx = gl_GlobalInvocationID.x;
    
    if(idx < cullCountPercascade * cascadeCount)
    {
        // Use cascade id ensure easy fetech cascade infos when render depth.
        uint cascadeIndex = idx / cullCountPercascade;

        // Object id fetech.
		uint objectId = idx % cullCountPercascade;

        visibileCulling(objectId, cascadeIndex);
    }
}