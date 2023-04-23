#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_samplerless_texture_functions : enable

#include "../common/shared_struct.glsl"
#include "../common/shared_functions.glsl"

layout (set = 0, binding = 0) uniform UniformFrameData{ PerFrameData frameData; };
layout (set = 0, binding = 1) readonly buffer SSBOPerObject { StaticMeshPerObjectData objectDatas[]; };
layout (set = 0, binding = 2) buffer SSBOIndirectDraws { StaticMeshDrawCommand drawCommands[]; };
layout (set = 0, binding = 3) buffer SSBODrawCount{ uint drawCount; };
layout (set = 0, binding = 4) uniform texture2D inHzbFurthest;

layout (push_constant) uniform PushConsts 
{
    // Total static mesh count need to cull.  
    uint cullCount; 
    uint hzbMipCount;
    vec2 hzbSrcSize;
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

    const mat4 mvp = frameData.camViewProj * objectData.modelMatrix;
    const vec3 exten = objectData.extents;

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
            return; // no visibile
		}
	}


    // Hzb culling test.
    {
        // Cast eight vertex to screen space, then compute texel size, then sample hzb, then compare depth occlusion state.


        const vec3 uvZ0 = projectPos(localPos + exten * vec3( 1.0,  1.0,  1.0), mvp);
        const vec3 uvZ1 = projectPos(localPos + exten * vec3(-1.0,  1.0,  1.0), mvp);
        const vec3 uvZ2 = projectPos(localPos + exten * vec3( 1.0, -1.0,  1.0), mvp);
        const vec3 uvZ3 = projectPos(localPos + exten * vec3( 1.0,  1.0, -1.0), mvp);
        const vec3 uvZ4 = projectPos(localPos + exten * vec3(-1.0, -1.0,  1.0), mvp);
        const vec3 uvZ5 = projectPos(localPos + exten * vec3( 1.0, -1.0, -1.0), mvp);
        const vec3 uvZ6 = projectPos(localPos + exten * vec3(-1.0,  1.0, -1.0), mvp);
        const vec3 uvZ7 = projectPos(localPos + exten * vec3(-1.0, -1.0, -1.0), mvp);

        vec3 maxUvz = max(max(max(max(max(max(max(uvZ0, uvZ1), uvZ2), uvZ3), uvZ4), uvZ5), uvZ6), uvZ7);
        vec3 minUvz = min(min(min(min(min(min(min(uvZ0, uvZ1), uvZ2), uvZ3), uvZ4), uvZ5), uvZ6), uvZ7);

        if(maxUvz.z < 1.0f && minUvz.z > 0.0f)
        {
            const vec2 bounds = maxUvz.xy - minUvz.xy;

            const float edge = max(1.0, max(bounds.x, bounds.y) * max(hzbSrcSize.x, hzbSrcSize.y));
            int mipLevel = int(min(ceil(log2(edge)), hzbMipCount - 1));

            const vec2 mipSize = vec2(textureSize(inHzbFurthest, mipLevel));
            const ivec2 samplePosMax = ivec2(saturate(maxUvz.xy) * mipSize);
            const ivec2 samplePosMin = ivec2(saturate(minUvz.xy) * mipSize);

            vec4 occ = vec4(
                texelFetch(inHzbFurthest, samplePosMax.xy, mipLevel).x, 
                texelFetch(inHzbFurthest, samplePosMin.xy, mipLevel).x, 
                texelFetch(inHzbFurthest, ivec2(samplePosMax.x, samplePosMin.y), mipLevel).x, 
                texelFetch(inHzbFurthest, ivec2(samplePosMin.x, samplePosMax.y), mipLevel).x);

            float occDepth = min(occ.w, min(occ.z, min(occ.x, occ.y)));

            // Occlusion, pre-return.
            if(occDepth > maxUvz.z)
            {
                return;
            }
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
