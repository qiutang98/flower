#pragma once

#include "resource.h"
#include "uploader.h"
#include <common_header.h>
#include "accelerate_struct.h"

namespace engine
{
	struct StaticMeshRenderBounds;

	class UploadAssetInterface : public StorageInterface
	{
	public:
		explicit UploadAssetInterface(UploadAssetInterface* fallback)
			: m_fallback(fallback)
		{
			if (m_fallback)
			{
				ASSERT(m_fallback->isAssetReady(), "Fallback asset must already load.");
			}
		}

		// Is this asset still loading or ready.
		bool isAssetLoading() const { return  m_bAsyncLoading; }
		bool isAssetReady()   const { return !m_bAsyncLoading; }

		// Set async load state.
		void setAsyncLoadState(bool bState) 
		{ 
			m_bAsyncLoading = bState;
		}

		template<typename T>
		T* getReadyAsset()
		{
			static_assert(std::is_base_of_v<UploadAssetInterface, T>, "Type must derived from UploadAssetInterface");
			if (isAssetLoading())
			{
				CHECK(m_fallback && "Loading asset must exist one fallback.");
				return dynamic_cast<T*>(m_fallback);
			}
			return dynamic_cast<T*>(this);
		}

		bool existFallback() const
		{
			return m_fallback;
		}

	protected:
		// The asset is under async loading.
		std::atomic<bool> m_bAsyncLoading = true;

		// Fallback asset when the asset is still loading.
		UploadAssetInterface* m_fallback = nullptr;
	};


	// GPU image asset, used for material texture2D sample.
	class GPUImageAsset : public UploadAssetInterface
	{
	public:
		GPUImageAsset(
			GPUImageAsset*     fallback,
			VkFormat           format,
			const std::string& name,
			uint32_t           mipmapCount,
			math::uvec3        dimension
		);

		virtual ~GPUImageAsset();

		virtual uint32_t getSize() const override { return (uint32_t)m_image->getSize(); }

		// Prepare image layout when start to upload.
		void prepareToUpload(RHICommandBufferBase& cmd, VkImageSubresourceRange range);

		// Finish image layout when ready for shader read.
		void finishUpload(RHICommandBufferBase& cmd, VkImageSubresourceRange range);

		// Get ready asset bindless index.
		uint32_t getBindlessIndex(VkImageSubresourceRange range = buildBasicImageSubresource())
		{ 
			auto* asset = getReadyAsset<GPUImageAsset>();

			// Use bindless asset.
			const auto imageViewType = asset->m_image->getInfo().imageType == VK_IMAGE_TYPE_3D ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_2D;
			return asset->m_image->getOrCreateView(range, imageViewType).srvBindless;
		}

		VulkanImage* getReadyImage()
		{
			if (m_fallback)
			{
				return getReadyAsset<GPUImageAsset>()->m_image.get();
			}
			return m_image.get();
		}

		// Self owner image.
		const VulkanImage& getSelfImage() const { return *m_image; }
		VulkanImage& getSelfImage() { return *m_image; }

	protected:
		// Image handle.
		std::unique_ptr<VulkanImage> m_image = nullptr;
	};

	struct AssetTextureLoadTask : public AssetLoadTask
	{
	public:
		virtual void finishCallback() override final
		{ 
			imageAssetGPU->setAsyncLoadState(false); 
		}

		virtual uint32_t uploadSize() const override 
		{ 
			return uint32_t(imageAssetGPU->getSize()); 
		}

	public:
		// Working image.
		std::shared_ptr<GPUImageAsset> imageAssetGPU = nullptr;
	};

	// Load from raw data, no mipmap, persistent, no compress, no mipmap, used for engine texture.

	enum class EImageFormatExr
	{
		RGBA,
		Greyscale,
	};

	struct RawAssetTextureLoadTask : public AssetTextureLoadTask
	{
	public:
		RawAssetTextureLoadTask();

		virtual void uploadFunction(
			uint32_t stageBufferOffset,
			void* bufferPtrStart,
			RHICommandBufferBase& commandBuffer,
			VulkanBuffer& stageBuffer) override;

	public:
		// Build load task from same value for flat texture.
		static std::shared_ptr<RawAssetTextureLoadTask> buildFlatTexture(
			const std::string& name,
			const UUID& uuid,
			const glm::uvec4& color,
			const glm::uvec3& size = { 1U, 1U, 1U },
			VkFormat format = VK_FORMAT_R8G8B8A8_UNORM);

		// Build load task from file path.
		static std::shared_ptr<RawAssetTextureLoadTask> buildTexture(
			const std::filesystem::path& path,
			const UUID& uuid,
			VkFormat format,
			bool bSRGB,
			uint channel,
			bool bMipmap = false,
			float alphaCoverage = 1.0f);


		static std::shared_ptr<RawAssetTextureLoadTask> buildExrTexture(
			const std::filesystem::path& path,
			const UUID& uuid,
			EImageFormatExr format,
			const char* layerName
		);

	public:
		// Cache asset texture bin data.
		std::unique_ptr<struct AssetTextureBin> cacheBin = nullptr;
	};

	///
	class AssetStaticMesh;
	class GPUStaticMeshAsset : public UploadAssetInterface
	{
	public:
		struct ComponentBuffer
		{
			std::unique_ptr<VulkanBuffer> buffer = nullptr;
			uint32_t bindless = ~0U;
			VkDeviceSize stripeSize = ~0U;
			uint32_t num = ~0U;
		};

		virtual ~GPUStaticMeshAsset();
		virtual uint32_t getSize() const override;

		GPUStaticMeshAsset(
			std::weak_ptr<AssetStaticMesh> asset,
			GPUStaticMeshAsset* fallback,
			const std::string& name,
			uint32_t verticesNum,
			uint32_t indicesNum
		);

		const ComponentBuffer& getIndices() const { return m_indices; }
		const ComponentBuffer& getPositions() const { return m_positions; }
		const ComponentBuffer& getNormals() const { return m_normals; }
		const ComponentBuffer& getUV0s() const { return m_uv0s; }
		const ComponentBuffer& getTangents() const { return m_tangents; }

		const uint32_t getVerticesCount() const { return m_verticesNum; }
		const uint32_t getIndicesCount() const { return m_indicesNum; }

		// Return BLAS cache, if it unbuild, will insert one build task to GPU, which need flush GPU.
		BLASBuilder& getOrBuilddBLAS();
		bool isBLASInit() const { return m_blasBuilder.isInit(); }
	protected:
		friend struct AssetStaticMeshLoadTask;

		void makeComponent(
			ComponentBuffer* in,
			VkBufferUsageFlags flags,
			const std::string name,
			VmaAllocationCreateFlags vmaFlags,
			uint32_t stripe,
			uint32_t num);

		void freeComponent(ComponentBuffer* in, VulkanBuffer* fallback);

	protected:
		std::weak_ptr<AssetStaticMesh> m_asset = {};

		ComponentBuffer m_indices;
		ComponentBuffer m_positions;
		ComponentBuffer m_normals;
		ComponentBuffer m_uv0s;
		ComponentBuffer m_tangents;

		uint32_t m_verticesNum;
		uint32_t m_indicesNum;

		// Every mesh asset hold one bottom level accelerate structure.
		BLASBuilder m_blasBuilder;
	};

	struct AssetStaticMeshLoadTask : public AssetLoadTask
	{
	public:
		virtual uint32_t uploadSize() const override final 
		{ 
			return uint32_t(meshAssetGPU->getSize()); 
		}

		virtual void finishCallback() override final;

	public:
		// Working static mesh.
		std::shared_ptr<GPUStaticMeshAsset> meshAssetGPU = nullptr;
		std::shared_ptr<AssetStaticMesh>    meshAsset = nullptr;
	};

	struct AssetRawStaticMeshLoadTask : public AssetStaticMeshLoadTask
	{
		virtual void uploadFunction(
			uint32_t stageBufferOffset, 
			void* bufferPtrStart, 
			RHICommandBufferBase& commandBuffer, 
			VulkanBuffer& stageBuffer) override;

		// Build persistent.
		static std::shared_ptr<AssetRawStaticMeshLoadTask> buildFromPath(
			GPUStaticMeshAsset* fallback,
			const std::filesystem::path& path,
			const UUID& uuid,
			// If no asset input, it is builtin.
			std::shared_ptr<AssetStaticMesh> assetIn 
		);

	public:
		std::vector<uint8_t> cacheTangents;
		std::vector<uint8_t> cacheNormals;
		std::vector<uint8_t> cacheUv0s;
		std::vector<uint8_t> cachePositions;
		std::vector<uint8_t> cacheIndices;
	};



}