#version 460
#extension GL_GOOGLE_include_directive : enable

#include "../common/shared_struct.glsl"
#include "../common/shared_functions.glsl"

layout (set = 0, binding = 0) uniform UniformFrameData{ PerFrameData frameData; };
layout (set = 0, binding = 1) readonly buffer SSBOPerObject { StaticMeshPerObjectData objectDatas[]; };
layout (set = 0, binding = 2) buffer SSBOIndirectDraws { StaticMeshDrawCommand drawCommands[]; };
layout (set = 0, binding = 3) buffer SSBODrawCount{ uint drawCount; };

layout (push_constant) uniform PushConsts 
{
    // Total static mesh count need to cull.  
    uint cullCount; 
};

layout(local_size_x = 64) in;
void main()
{
    // get working id.
    uint idx = gl_GlobalInvocationID.x;
    if(idx >= cullCount)
    {
        return;
    }

    const StaticMeshPerObjectData objectData = objectDatas[idx];

    vec3 localPos = objectData.sphereBounds.xyz;
	vec4 worldPos = objectData.modelMatrix * vec4(localPos, 1.0f);

    // local to world normal matrix.
	mat3 normalMatrix = transpose(inverse(mat3(objectData.modelMatrix)));
	mat3 world2Local = inverse(normalMatrix);

	// frustum culling test.
	for (int i = 0; i < 6; i++) 
	{
        vec3 worldSpaceN = frameData.frustumPlanes[i].xyz;
        float castDistance = dot(worldPos.xyz, worldSpaceN);

		// transfer to local matrix and use abs get first dimensions project value,
		// use that for test.
		vec3 localNormal = world2Local * worldSpaceN;
		float absDiff = dot(abs(localNormal), objectData.extents.xyz);
		if (castDistance + absDiff + frameData.frustumPlanes[i].w < 0.0)
		{
            // no visibile
            return; 
		}
	}

    // Build draw command if visible.
    {
        uint drawId = atomicAdd(drawCount, 1);
        drawCommands[drawId].objectId = idx;

        // We fetech vertex by index, so vertex count is index count.
        drawCommands[drawId].vertexCount = objectData.indexCount;
        drawCommands[drawId].firstVertex = objectData.indexStartPosition;

        // We fetch vertex in vertex shader, so instancing is unused when rendering.
        drawCommands[drawId].instanceCount = 1;
        drawCommands[drawId].firstInstance = 0; 
    }
}
