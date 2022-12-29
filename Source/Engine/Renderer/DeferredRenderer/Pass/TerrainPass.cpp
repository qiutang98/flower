#include "Pch.h"
#include "../DeferredRenderer.h"
#include "../../Renderer.h"
#include "../../RenderSceneData.h"
#include "../../SceneTextures.h"
#include "../../RenderSettingContext.h"

namespace Flower
{
	// GPU driven terrain.
	struct TerrainPlaneVertex
	{
		glm::vec3 position;
		glm::vec2 uv0;
	};

	// 16x16 plane mesh, per grid is 0.5m, all size is 8m x 8m.
	inline static void buildTerrainPlane(
		std::vector<uint32_t>& inOutIndices, 
		std::vector<TerrainPlaneVertex>& inOutVertices)
	{
		uint32_t dim = 16;
		float gridDimMeter = 0.5f;

		// 8x8 meter.
		float dimSizeMeter = dim * gridDimMeter;

		// Total 16 * 16 grid.
		uint32_t gridCount = dim * dim;

		// Per grid keep two triangle.
		uint32_t triangleCount = gridCount * 2;

		// Offset center.
		float centerOffset = -0.5f * dimSizeMeter;
		float uvStrip = 1.0f / float(dim);

		// Fill vertex info.
		for (uint32_t z = 0; z <= dim; z++) // [0, 16]
		{
			for (uint32_t x = 0; x <= dim; x++) // [0, 16]
			{
				TerrainPlaneVertex vertex;
				vertex.position = { centerOffset + x * gridDimMeter, 0.0f, centerOffset + z * gridDimMeter };
				vertex.uv0 = { x * uvStrip, z * uvStrip };

				// Add new vertex.
				inOutVertices.push_back(vertex);
			}
		}

		// Fill indices info.
		inOutIndices.resize(triangleCount * 3);
		for (uint32_t gridIndex = 0; gridIndex < gridCount; gridIndex++)
		{
			// Per grid keep two triangle and six indices.
			uint32_t basicOffset = gridIndex * 6;

			uint32_t vertexIndex = (gridIndex / dim) * (dim + 1) + (gridIndex % dim);

			inOutIndices[basicOffset + 0] = vertexIndex;
			inOutIndices[basicOffset + 1] = vertexIndex + dim + 1;
			inOutIndices[basicOffset + 2] = vertexIndex + 1;
			inOutIndices[basicOffset + 3] = vertexIndex + 1;
			inOutIndices[basicOffset + 4] = vertexIndex + dim + 1;
			inOutIndices[basicOffset + 5] = vertexIndex + dim + 2;
		}
	}

	class TerrainPass : public PassInterface
	{
	public:
		VkPipeline terrainDrawPipeline = VK_NULL_HANDLE;
		VkPipelineLayout terrainDrawPipelineLayout = VK_NULL_HANDLE;

	protected:
		virtual void init() override
		{

		}

		virtual void release() override
		{
			RHISafeRelease(terrainDrawPipeline);
			RHISafeRelease(terrainDrawPipelineLayout);
		}
	};

	void DeferredRenderer::renderTerrain(
		VkCommandBuffer cmd, 
		Renderer* renderer, 
		SceneTextures* inTextures, 
		RenderSceneData* scene, 
		BufferParamRefPointer& viewData, 
		BufferParamRefPointer& frameData)
	{
		// Build terrain content if no exist.
		if (m_terrain == nullptr)
		{
			m_terrain = std::make_unique<TerrainContent>();

			std::vector<uint32_t> indices{};
			std::vector<TerrainPlaneVertex> vertices{};

			buildTerrainPlane(indices, vertices);

			auto bufferFlagBasic = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
			VmaAllocationCreateFlags bufferFlagVMA = {};

			// Upload vertex buffer.
			{
				auto vbMemSize = uint32_t(sizeof(TerrainPlaneVertex) * vertices.size());
				m_terrain->planeVerticesBuffer = VulkanBuffer::create2(
					"TerrainPlaneVertices",
					bufferFlagBasic | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
					VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
					bufferFlagVMA,
					vbMemSize
				);

				auto stageBuffer = VulkanBuffer::create(
					"CopyBuffer",
					VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
					VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
					EVMAUsageFlags::StageCopyForUpload,
					vbMemSize,
					(void*)(vertices.data())
				);
				m_terrain->planeVerticesBuffer->stageCopyFrom(stageBuffer->getVkBuffer(),vbMemSize,0,0);
			}
			
			// Upload index buffer.
			{
				// Create buffer
				auto ibMemSize = uint32_t(indices.size() * sizeof(indices[0]));
				m_terrain->planeIndicesBuffer = VulkanBuffer::create2(
					"PlaneIndicesBuffer",
					bufferFlagBasic | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
					VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
					bufferFlagVMA,
					ibMemSize
				);

				// Copy index to GPU.

				auto stageBuffer = VulkanBuffer::create(
					"CopyBuffer",
					VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
					VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
					EVMAUsageFlags::StageCopyForUpload,
					ibMemSize,
					(void*)(indices.data())
				);

				m_terrain->planeIndicesBuffer->stageCopyFrom(stageBuffer->getVkBuffer(), ibMemSize, 0, 0);
			}
		}


	}
}