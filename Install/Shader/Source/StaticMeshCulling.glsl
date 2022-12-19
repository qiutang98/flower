#version 460

/*
** Physical based render code, develop by engineer: qiutanguu.
*/

#extension GL_GOOGLE_include_directive : enable

#include "StaticMeshCommon.glsl"

layout (set = 0, binding = 0) readonly buffer SSBOPerObject { PerObjectData objectDatas[]; };
layout (set = 1, binding = 0) buffer SSBOIndirectDraws { DrawIndirectCommand indirectCommands[]; };
layout (set = 2, binding = 0) buffer SSBODrawCount{ DrawIndirectCount drawCount; };
layout (set = 3, binding = 0) uniform UniformViewData{ ViewData viewData; };

layout (push_constant) uniform PushConsts 
{  
    uint cullCount; 
};

void visibileCulling(uint idx)
{
    PerObjectData objectData = objectDatas[idx];

	vec3 localPos = objectData.sphereBounds.xyz;
	vec4 worldPos = objectData.modelMatrix * vec4(localPos, 1.0f);

	// local to world normal matrix.
	mat3 normalMatrix = transpose(inverse(mat3(objectData.modelMatrix)));
	mat3 world2Local = inverse(normalMatrix);

	// frustum test.
	for (int i = 0; i < 6; i++) 
	{
        vec3 worldSpaceN = viewData.frustumPlanes[i].xyz;
        float castDistance = dot(worldPos.xyz, worldSpaceN);

		// transfer to local matrix and use abs get first dimensions project value,
		// use that for test.
		vec3 localNormal = world2Local * worldSpaceN;
		float absDiff = dot(abs(localNormal), objectData.extents.xyz);
		if (castDistance + absDiff + viewData.frustumPlanes[i].w < 0.0)
		{
            return; // no visibile
		}
	}
    
    // Build draw command if visible.
    uint drawId = atomicAdd(drawCount.count, 1);
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

    if(idx < cullCount)
    {
        visibileCulling(idx);
    }
}