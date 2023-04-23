#ifndef TERRAIN_COMMON_GLSL
#define TERRAIN_COMMON_GLSL

// TODO: Frustum Culling

#include "../common/shared_functions.glsl"

#define CBT_SET_INDEX 0
#define CBT_BINDING_INDEX 0
#include "../cbt/cbt.glsl"

#include "../leb/leb.glsl"

layout (set = 0, binding = 1) uniform UniformFrameData { PerFrameData frameData; };
layout (set = 0, binding = 2) uniform texture2D inHeightmap; 
layout (set = 0, binding = 3, r8) uniform image2D outSelectionMask;
layout (set = 0, binding = 4) buffer SSBOCascadeInfoBuffer{ CascadeInfo cascadeInfos[]; }; // Cascade infos.

struct RenderTerrainDynamicData
{
    mat4 u_ModelMatrixPrev;
    uint maskTexId;
    uint pad0;
    uint pad1;
    uint pad2;
};

#define SHARED_SAMPLER_SET 1
#include "../common/shared_sampler.glsl"

layout (set = 2, binding = 0) uniform  texture2D texture2DBindlessArray[];
layout (set = 3, binding = 0) uniform UniformRenderDynamicData { RenderTerrainDynamicData dynamicData; };



layout(push_constant) uniform PushConstants
{
    mat4  u_ModelMatrix;

    float u_LodFactor;
    float u_DmapFactor;
    float u_MinLodVariance;
    uint sceneNodeId;

    uint bSelected;
    uint cascadeId;
};

// DecodeTriangleVertices -- Decodes the triangle vertices in local space
vec4[3] DecodeTriangleVertices(in const cbt_Node node)
{
    vec3 xPos = vec3(0, 0, 1), yPos = vec3(1, 0, 0);
    mat2x3 pos = leb_DecodeNodeAttributeArray_Square(node, mat2x3(xPos, yPos));
    vec4 p1 = vec4(pos[0][0], pos[1][0], 0.0, 1.0);
    vec4 p2 = vec4(pos[0][1], pos[1][1], 0.0, 1.0);
    vec4 p3 = vec4(pos[0][2], pos[1][2], 0.0, 1.0);

    p1.z = u_DmapFactor * texture(sampler2D(inHeightmap, linearClampEdgeMipFilterSampler), p1.xy).r;
    p2.z = u_DmapFactor * texture(sampler2D(inHeightmap, linearClampEdgeMipFilterSampler), p2.xy).r;
    p3.z = u_DmapFactor * texture(sampler2D(inHeightmap, linearClampEdgeMipFilterSampler), p3.xy).r;

    return vec4[3](p1, p2, p3);
}

float TriangleLevelOfDetail(in const vec4[3] patchVertices)
{
    mat4 modelView = frameData.camView * u_ModelMatrix;

    vec3 v0 = (modelView * patchVertices[0]).xyz;
    vec3 v2 = (modelView * patchVertices[2]).xyz;

    vec3 edgeCenter = (v0 + v2); // division by 2 was moved to u_LodFactor
    vec3 edgeVector = (v2 - v0);
    float distanceToEdgeSqr = dot(edgeCenter, edgeCenter);
    float edgeLengthSqr = dot(edgeVector, edgeVector);
    return u_LodFactor + log2(edgeLengthSqr / distanceToEdgeSqr);
}

bool FrustumCulling(in const vec4[3] patchVertices)
{
    vec3 worldP0 = (u_ModelMatrix * patchVertices[0]).xyz;
    vec3 worldP1 = (u_ModelMatrix * patchVertices[1]).xyz;
    vec3 worldP2 = (u_ModelMatrix * patchVertices[2]).xyz;

    vec3 maxP = max(max(worldP0, worldP1), worldP2);
    vec3 minP = min(min(worldP0, worldP1), worldP2);

    vec3 extents = (maxP - minP) * 0.5f;
    vec3 worldPos = (worldP0 + worldP1 + worldP2) / 3.0f;

    // local to world normal matrix.
	mat3 normalMatrix = transpose(inverse(mat3(u_ModelMatrix)));
	mat3 world2Local = inverse(normalMatrix);

	// frustum culling test.
	for (int i = 0; i < 6; i++) 
	{
        vec3 worldSpaceN = frameData.frustumPlanes[i].xyz;
        float castDistance = dot(worldPos.xyz, worldSpaceN);

		// transfer to local matrix and use abs get first dimensions project value,
		// use that for test.
		vec3 localNormal = world2Local * worldSpaceN;
		float absDiff = dot(abs(localNormal), extents);
		if (castDistance + absDiff + frameData.frustumPlanes[i].w < 0.0)
		{
            // no visibile
            return false;  
		}
	}

    return true;
}

/*
 * LevelOfDetail -- Computes the level of detail of associated to a triangle
 *
 * The first component is the actual LoD value. The second value is 0 if the
 * triangle is culled, and one otherwise.
 */
vec2 LevelOfDetail(in const vec4[3] patchVertices)
{
    if(!FrustumCulling(patchVertices))
    {
        return vec2(0.0f, 0.0f);
    }

    // Compute triangle LOD
    return vec2(TriangleLevelOfDetail(patchVertices), 1.0f);
}

vec2 BarycentricInterpolation(in vec2 v[3], in vec2 u)
{
    return v[1] + u.x * (v[2] - v[1]) + u.y * (v[0] - v[1]);
}

vec4 BarycentricInterpolation(in vec4 v[3], in vec2 u)
{
    return v[1] + u.x * (v[2] - v[1]) + u.y * (v[0] - v[1]);
}

struct VertexAttribute 
{
    vec4 position;
    vec2 texCoord;
};

VertexAttribute TessellateTriangle(in const vec2 texCoords[3], in vec2 tessCoord) 
{
    vec2 texCoord = BarycentricInterpolation(texCoords, tessCoord);
    vec4 position = vec4(texCoord, 0, 1);

    position.z = u_DmapFactor * textureLod(sampler2D(inHeightmap, linearClampEdgeMipFilterSampler), texCoord, 0.0).r;

    return VertexAttribute(position, texCoord);
}

#endif