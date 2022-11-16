#pragma once

#include "../Core/Core.h"
#include "../Renderer/MeshMisc.h"

namespace Flower
{
	static const glm::vec4 BuildInSphereBounds = { 0.0f, 0.0f, 0.0f, 2.0f };
	static const glm::vec4 BuildInExtent = glm::vec4{ 1.0f, 1.0f, 1.0f, 0.0f };

	struct MeshBuilder
	{
		struct MeshData
		{
			std::vector<StaticMeshVertex> vertices;
			std::vector<VertexIndexType> indices;

			inline size_t getVertexSize() const
			{
				return vertices.size() * sizeof(vertices[0]);
			}

			inline size_t getIndexSize() const
			{
				return indices.size() * sizeof(indices[0]);
			}
		};

		static MeshData buildBox();
	};
	
}