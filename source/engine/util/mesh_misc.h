#pragma once
#include "math.h"
#include "uuid.h"

namespace engine
{
    struct StaticMeshRenderBounds
	{
		// AABB min = origin - extents.
		// AABB max = origin + extents.
		// AABB center = origin.
		math::vec3 origin;
		math::vec3 extents;
		float radius;

		template<class Archive> void serialize(Archive& archive) 
		{ 
			archive(origin, extents, radius); 
		}
	};

	struct StaticMeshSubMesh
	{
		uint32_t indicesStart = 0;
		uint32_t indicesCount = 0;

		// Material of this submesh.
		UUID material = {};
		StaticMeshRenderBounds bounds = {};

		template<class Archive> void serialize(Archive& archive)
		{
			archive(indicesStart, indicesCount, material, bounds);
		}
	};

    // Standard index type in this engine.
    using VertexIndexType = uint32_t;

    // No sure which vertex layout is better.
    // We default use seperate method instead of interleave.
    // https://frostbite-wp-prd.s3.amazonaws.com/wp-content/uploads/2016/03/29204330/GDC_2016_Compute.pdf 
    // https://developer.android.com/games/optimize/vertex-data-management?hl=zh-tw
    using VertexPosition = math::vec3; static_assert(sizeof(VertexPosition) == sizeof(float) * 3);
    using VertexNormal = math::vec3; static_assert(sizeof(VertexNormal) == sizeof(float) * 3);
    using VertexTangent = math::vec4; static_assert(sizeof(VertexTangent) == sizeof(float) * 4);
    using VertexUv0 = math::vec2; static_assert(sizeof(VertexUv0) == sizeof(float) * 2);
}