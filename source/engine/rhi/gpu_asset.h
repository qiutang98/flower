#pragma once
#include "resource.h"
#include <util/util.h>
#include <util/lru.h>
#include "async_upload.h"
#include "accelerate_struct.h"

namespace engine
{


	class GPUImageAsset : public LRUAssetInterface
	{
	public:
		GPUImageAsset(
			VulkanContext* context,
			GPUImageAsset* fallback,
			VkFormat format,
			const std::string& name,
			uint32_t mipmapCount,
			uint32_t width,
			uint32_t height,
			uint32_t depth
		);

		virtual ~GPUImageAsset();

		virtual size_t getSize() const override { return m_image->getSize(); }


		// Prepare image layout when start to upload.
		void prepareToUpload(RHICommandBufferBase& cmd, VkImageSubresourceRange range);

		// Finish image layout when ready for shader read.
		void finishUpload(RHICommandBufferBase& cmd, VkImageSubresourceRange range);

		uint32_t getBindlessIndex() { return getReadyAsset<GPUImageAsset>()->m_bindlessIndex; }

		VulkanImage& getImage() { return *m_image; }

	protected:
		// Vulkan context.
		VulkanContext* m_context;

		// Image handle.
		std::unique_ptr<VulkanImage> m_image = nullptr;

		// Bindless index.
		uint32_t m_bindlessIndex = ~0;
	};

	struct AssetTextureLoadTask : public AssetLoadTask
	{
		AssetTextureLoadTask() = default;

		// Working image.
		std::shared_ptr<GPUImageAsset> imageAssetGPU = nullptr;
		virtual void finishCallback() override final { imageAssetGPU->setAsyncLoadState(false); }
		virtual uint32_t uploadSize() const override { return uint32_t(imageAssetGPU->getSize()); }
	};

	// Load from raw data, no mipmap, persistent, no compress, no mipmap, used for engine texture.
	struct RawAssetTextureLoadTask : public AssetTextureLoadTask
	{
		RawAssetTextureLoadTask();

		std::unique_ptr<struct AssetTextureBin> cacheBin = nullptr;

		virtual void uploadFunction(
			uint32_t stageBufferOffset, 
			void* bufferPtrStart,
			RHICommandBufferBase& commandBuffer, 
			VulkanBuffer& stageBuffer) override;

		// Build load task from file path.
		static std::shared_ptr<RawAssetTextureLoadTask> buildTexture(
			bool bEngineTex,
			VulkanContext* context, 
			const std::filesystem::path& path, 
			const UUID& uuid, 
			VkFormat format,
			bool bSRGB,
			bool bMipmap = false);

		static std::shared_ptr<RawAssetTextureLoadTask> buildAssetTexture(
			VulkanContext* context,
			const std::filesystem::path& path,
			const UUID& uuid,
			VkFormat format,
			bool bSRGB,
			bool bMipmap = false);

		static std::shared_ptr<RawAssetTextureLoadTask> buildEngine3dTexture(
			VulkanContext* context,
			const std::filesystem::path& path,
			const UUID& uuid,
			VkFormat format,
			math::uvec3 dim);

		// Build load task from same value for engine.
		static std::shared_ptr<RawAssetTextureLoadTask> buildEngineFlatTexture(
			VulkanContext* context, 
			const std::string& name, 
			const UUID& uuid, 
			const glm::uvec4& color, 
			const glm::uvec3& size = { 1u, 1u, 1u }, 
			VkFormat format = VK_FORMAT_R8G8B8A8_UNORM);
	};


	class GPUStaticMeshAsset : public LRUAssetInterface
	{
	public:
		GPUStaticMeshAsset(
			VulkanContext* context,
			UUID assetId,
			GPUStaticMeshAsset* fallback,
			const std::string& name,
			VkDeviceSize tangentSize,
			VkDeviceSize tangentStripSize,
			VkDeviceSize normalSize,
			VkDeviceSize normalStripSize,
			VkDeviceSize uv0Size,
			VkDeviceSize uv0StripSize,
			VkDeviceSize positionsSize,
			VkDeviceSize positionStripSize,
			VkDeviceSize indicesSize,
			VkDeviceSize indexStripSize
		);

		virtual ~GPUStaticMeshAsset();

		virtual size_t getSize() const override 
		{ 
			return
				m_indices->getSize() +
				m_positions->getSize() +
				m_tangents->getSize() +
				m_normals->getSize() +
				m_uv0s->getSize();
		}

		const auto* getIndices()   const { return m_indices.get(); }
		const auto* getTangents() const { return m_tangents.get(); }
		const auto* getNormals() const { return m_normals.get(); }
		const auto* getUv0s() const { return m_uv0s.get(); }
		const auto* getPosition()  const { return m_positions.get(); }

		auto* getIndices()   { return m_indices.get(); }
		auto* getTangents() { return m_tangents.get(); }
		auto* getNormals() { return m_normals.get(); }
		auto* getUv0s() { return m_uv0s.get(); }
		auto* getPosition()  { return m_positions.get(); }

		const auto& getIndicesBindless()  const { return m_indicesBindless; }
		const auto& getTangentsBindless() const { return m_tangentsBindless; }
		const auto& getNormalsBindless() const { return m_normalsBindless; }
		const auto& getUv0sBindless() const { return m_uv0sBindless; }
		const auto& getPositionBindless() const { return m_positionBindless; }

		const auto getVerticesCount() const { return m_positionsSize / m_positionStripSize; }
		const auto getIndicesCount() const { return m_indicesSize / m_indexStripSize; }

		// Return BLAS cache, if it unbuild, will insert one build task to GPU, which need flush GPU.
		BLASBuilder& getOrBuilddBLAS();

		bool isEngineAsset() const;
		const UUID& getAssetUUID() const { return m_assetId; }
	private:
		// Vulkan context.
		VulkanContext* m_context;

		// Indices buffer handle.
		std::unique_ptr<VulkanBuffer> m_indices = nullptr;
		uint32_t m_indicesBindless = ~0;

		// Vertices buffer.
		std::unique_ptr<VulkanBuffer> m_tangents = nullptr;
		uint32_t m_tangentsBindless = ~0;
		std::unique_ptr<VulkanBuffer> m_normals = nullptr;
		uint32_t m_normalsBindless = ~0;
		std::unique_ptr<VulkanBuffer> m_uv0s = nullptr;
		uint32_t m_uv0sBindless = ~0;
		std::unique_ptr<VulkanBuffer> m_positions = nullptr;
		uint32_t m_positionBindless = ~0;

		VkDeviceSize m_tangentsSize;
		VkDeviceSize m_tangentStripSize;
		VkDeviceSize m_normalSize;
		VkDeviceSize m_normalStripSize;
		VkDeviceSize m_uv0Size;
		VkDeviceSize m_uv0StripSize;

		VkDeviceSize m_positionsSize;
		VkDeviceSize m_positionStripSize;
		VkDeviceSize m_indicesSize;
		VkDeviceSize m_indexStripSize;

		// Every mesh asset hold one bottom level accelerate structure.
		BLASBuilder m_blasBuilder;

		// Cache asset id.
		UUID m_assetId;
	};

	struct AssetStaticMeshLoadTask : public AssetLoadTask
	{
		AssetStaticMeshLoadTask() { }

		// Working static mesh.
		std::shared_ptr<GPUStaticMeshAsset> meshAssetGPU = nullptr;

		virtual uint32_t uploadSize() const override final { return uint32_t(meshAssetGPU->getSize()); }
		virtual void finishCallback() override final { meshAssetGPU->setAsyncLoadState(false); }
	};

	struct AssetRawStaticMeshLoadTask : public AssetStaticMeshLoadTask
	{
		AssetRawStaticMeshLoadTask() { }

		std::vector<uint8_t> cacheTangents;
		std::vector<uint8_t> cacheNormals;
		std::vector<uint8_t> cacheUv0s;
		std::vector<uint8_t> cachePositions;
		std::vector<uint8_t> cacheIndices;

		virtual void uploadFunction(uint32_t stageBufferOffset, void* bufferPtrStart, RHICommandBufferBase& commandBuffer, VulkanBuffer& stageBuffer) override;

		// Build persistent.
		static std::shared_ptr<AssetRawStaticMeshLoadTask> buildFromPath(
			VulkanContext* context,
			const std::filesystem::path& path,
			const UUID& uuid,
			StaticMeshRenderBounds& bounds
		);
	};
}