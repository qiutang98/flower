#pragma once
#include "RendererCommon.h"

namespace Flower
{
	// standard index type.
	using VertexIndexType = uint32_t;

	// Standard static mesh vertex.
	struct StaticMeshVertex
	{
		glm::vec3 position = { 0.0f, 0.0f, 0.0f };
		glm::vec3 normal = { 0.0f, 1.0f, 0.0f }; // normal to up as default.
		glm::vec4 tangent = { 1.0f, 0.0f, 0.0f, 1.0f };
		glm::vec2 uv0 = { 0.0f, 0.0f };

		StaticMeshVertex() = default;

		StaticMeshVertex(
			float px, float py, float pz,
			float nx, float ny, float nz,
			float tx, float ty, float tz,
			float u, float v) :
			position(px, py, pz),
			normal(nx, ny, nz),
			tangent(tx, ty, tz, 1.0f),
			uv0(u, v) {}

		template<class Archive>
		void serialize(Archive& archive)
		{
			archive(position, normal, tangent, uv0);
		}
	};
	// Explicit assert struct size avoid some glm macro pad memory to float4.
	static_assert(sizeof(StaticMeshVertex) == (3 + 3 + 4 + 2) * sizeof(float));

	// render bounds of submesh.
	struct StaticMeshRenderBounds
	{
		glm::vec3 origin;
		float radius;
		glm::vec3 extents;

		template<class Archive>
		void serialize(Archive& archive)
		{
			archive(origin);
			archive(radius);
			archive(extents);
		}

		static void toExtents(
			const StaticMeshRenderBounds& in,
			float& zmin,
			float& zmax,
			float& ymin,
			float& ymax,
			float& xmin,
			float& xmax,
			float scale = 1.5f);

		static StaticMeshRenderBounds combine(
			const StaticMeshRenderBounds& b0,
			const StaticMeshRenderBounds& b1);

	};

	struct StaticMeshSubMesh
	{
		StaticMeshRenderBounds renderBounds = {};
		uint32_t indexStartPosition = 0;
		uint32_t indexCount = 0;
		UUID material = {};

		template<class Archive>
		void serialize(Archive& archive)
		{
			archive(renderBounds, indexStartPosition, indexCount, material);
		}
	};
}