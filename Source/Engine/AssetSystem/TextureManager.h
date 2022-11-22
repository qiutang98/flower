#pragma once
#include "AssetCommon.h"
#include "LRUCache.h"
#include "AsyncUploader.h"


namespace Flower
{
	class GPUImageAsset;
	namespace EngineTextures
	{
		extern std::weak_ptr<GPUImageAsset> GWhiteTexturePtr;
		extern const UUID GWhiteTextureUUID; // 255, 255, 255, 255
		extern uint32_t GWhiteTextureId; // BindlessIndex

		extern const UUID GGreyTextureUUID; // 128, 128, 128, 255
		extern uint32_t GGreyTextureId; // BindlessIndex

		extern const UUID GBlackTextureUUID; // 0, 0, 0, 255
		extern uint32_t GBlackTextureId; // BindlessIndex

		extern const UUID GTranslucentTextureUUID; // 0, 0, 0, 0
		extern uint32_t GTranslucentTextureId; // BindlessIndex

		extern const UUID GNormalTextureUUID; // 125, 130, 255, 0
		extern uint32_t GNormalTextureId; // BindlessIndex

		extern const UUID GDefaultSpecularUUID; // 255, 255, 0, 0 .r AO, .g roughness, .b metal
		extern uint32_t GDefaultSpecularId; // BindlessIndex

		extern const UUID GCloudWeatherUUID;
		extern uint32_t GCloudWeatherId;

		extern const UUID GCloudGradientUUID;
		extern uint32_t GCloudGradientId;
	}

	class ImageAssetBin;
	class ImageAssetHeader : public AssetHeaderInterface
	{
	private:
		uint32_t m_width;
		uint32_t m_height;
		uint32_t m_depth;
		uint32_t m_mipmapCount;

		size_t m_format;
		bool m_bSrgb;
		bool m_bHdr;

		uint32_t m_widthSnapShot;
		uint32_t m_heightSnapShot;
		std::vector<uint8_t> m_snapshotData;

		UUID m_snapshotUUID;

	private:
		friend class cereal::access;

		template<class Archive>
		void serialize(Archive& archive)
		{
			archive(
				cereal::base_class<AssetHeaderInterface>(this),
				m_width, m_widthSnapShot,
				m_height, m_heightSnapShot,
				m_depth,
				m_format,
				m_mipmapCount,
				m_bSrgb,
				m_bHdr,
				m_snapshotData,
				m_snapshotUUID
			);
		}

		void buildSnapshotData2D(std::shared_ptr<ImageAssetBin> inBin);

	public:
		ImageAssetHeader() = default;
		ImageAssetHeader(const std::string& name)
			: AssetHeaderInterface(buildUUID(), name), m_snapshotUUID(buildUUID())
		{

		}

		void setHdr(bool bHdr)
		{
			m_bHdr = bHdr;
		}

		bool isHdr() const { return m_bHdr; }

		VkFormat getFormat() const
		{
			return VkFormat(m_format);
		}

		const UUID getSnapShotUUID() const
		{
			return m_snapshotUUID;
		}

		virtual EAssetType getType() const
		{
			return EAssetType::Texture;
		}

		bool isSRGB() const
		{
			return m_bSrgb;
		}

		const auto& getSnapShotData() const
		{
			return m_snapshotData;
		}

		uint32_t getSnapShotWidth() const
		{
			return m_widthSnapShot;
		}

		uint32_t getSnapShotHeight() const
		{
			return m_heightSnapShot;
		}

		uint32_t getWidth() const
		{
			return m_width;
		}

		uint32_t getHeight() const
		{
			return m_height;
		}

		uint32_t getMipmapCount() const
		{
			return m_mipmapCount;
		}

	public:
		// Cutoff is alpha cut off factor, use for mipmap generate and keep alpha coverage same.
		bool initFromRaw2DLDR(const std::filesystem::path& rawPath, bool bSRGB, float cutOff, bool bBuildMipmap);

		bool initFromRaw2DHDR(const std::filesystem::path& rawPath, bool bBuildMipmap);
	};

	class ImageAssetBin : public AssetBinInterface
	{
	private:
		friend ImageAssetHeader;

		// Raw image data, store for asset data callback.
		std::vector<uint8_t> m_rawData;

		std::vector<std::vector<uint8_t>> m_mipmapData;

	private:
		friend class cereal::access;

		template<class Archive>
		void serialize(Archive& archive)
		{
			archive(cereal::base_class<AssetBinInterface>(this));
			archive(m_rawData);
			archive(m_mipmapData);
		}

		void buildMipmapDataRGBA8(ImageAssetHeader* header, float cutOff);

	public:
		ImageAssetBin() = default;
		ImageAssetBin(const std::string& name)
			: AssetBinInterface(buildUUID(), name)
		{

		}

		virtual EAssetType getType() const override
		{
			return EAssetType::Texture;
		}

		const std::vector<std::vector<uint8_t>>& getMipmapDatas() const
		{
			return m_mipmapData;
		}

		const std::vector<uint8_t>& getRawDatas() const
		{
			return m_rawData;
		}
	};

	class GPUImageAsset : public LRUAssetInterface
	{
	private:
		std::shared_ptr<VulkanImage> m_image = nullptr;
		uint32_t m_bindlessIndex = ~0;

	public:
		GPUImageAsset(
			bool bPersistent,
			GPUImageAsset* fallback,
			VkFormat format,
			const std::string& name,
			uint32_t mipmapCount,
			uint32_t width,
			uint32_t height,
			uint32_t depth
		);

		virtual ~GPUImageAsset();


		uint32_t getBindlessIndex()
		{
			return getReadyAsset()->m_bindlessIndex;
		}

		// Prepare image layout when start to upload.
		void prepareToUpload(RHICommandBufferBase& cmd, VkImageSubresourceRange range);

		// Finish image layout when ready for shader read.
		void finishUpload(RHICommandBufferBase& cmd, VkImageSubresourceRange range);

		virtual size_t getSize() const override
		{
			return m_image->getMemorySize();
		}

		auto& getImage()
		{
			return *m_image;
		}

		GPUImageAsset* getReadyAsset()
		{
			if (isAssetLoading())
			{
				CHECK(m_fallback && "Loading asset must exist one fallback.");
				return dynamic_cast<GPUImageAsset*>(m_fallback);
			}
			return this;
		}
	};

	class TextureContext : NonCopyable
	{
	private:
		std::unique_ptr<LRUAssetCache<GPUImageAsset>> m_lruCache;

	public:
		TextureContext() = default;

		void shrinkLRU();

		void init();
		void release();

		bool isAssetExist(const UUID& id)
		{
			return m_lruCache->contain(id);
		}

		void insertGPUAsset(const UUID& uuid, std::shared_ptr<GPUImageAsset> image)
		{
			m_lruCache->insert(uuid, image);
		}

		std::shared_ptr<GPUImageAsset> getImage(const UUID& id)
		{
			return m_lruCache->tryGet(id);
		}

		std::shared_ptr<GPUImageAsset> getOrCreateLRUSnapShot(std::shared_ptr<ImageAssetHeader> asset);
		std::shared_ptr<GPUImageAsset> getOrCreateImage(std::shared_ptr<ImageAssetHeader> asset);
	};

	using TextureManager = Singleton<TextureContext>;

	struct AssetTextureLoadTask : public AssetLoadTask
	{
		AssetTextureLoadTask() = default;

		// Working image.
		std::shared_ptr<GPUImageAsset> imageAssetGPU = nullptr;

		virtual uint32_t uploadSize() const override
		{
			return uint32_t(imageAssetGPU->getSize());
		}
	};

	// Load from raw data, no mipmap, persistent, no compress, no mipmap, used for engine texture.
	struct RawAssetTextureLoadTask : public AssetTextureLoadTask
	{
		RawAssetTextureLoadTask() = default;

		std::vector<uint8_t> cacheRawData;

		virtual void finishCallback() override
		{
			imageAssetGPU->setAsyncLoadState(false);
		}

		virtual void uploadFunction(
			uint32_t stageBufferOffset,
			RHICommandBufferBase& commandBuffer,
			VulkanBuffer& stageBuffer) override;

		// Build load task from file path.
		// Slow.
		static std::shared_ptr<RawAssetTextureLoadTask> build(
			const std::filesystem::path& path,
			const UUID& uuid,
			VkFormat format);

		// Build load task from same value.
		static std::shared_ptr<RawAssetTextureLoadTask> buildFlatTexture(
			const std::string& name,
			const UUID& uuid,
			const glm::uvec4& color,
			const glm::uvec3& size = { 1u, 1u, 1u },
			VkFormat format = VK_FORMAT_R8G8B8A8_UNORM);
	};

	// Load from asset header snapshot data, no compress, cache in lru map.
	struct SnapshotAssetTextureLoadTask : public AssetTextureLoadTask
	{
		explicit SnapshotAssetTextureLoadTask(std::shared_ptr<ImageAssetHeader> inHeader)
			: cacheHeader(inHeader)
		{

		}

		std::shared_ptr<ImageAssetHeader> cacheHeader;

		virtual void finishCallback() override
		{
			imageAssetGPU->setAsyncLoadState(false);
		}

		virtual void uploadFunction(
			uint32_t stageBufferOffset,
			RHICommandBufferBase& commandBuffer,
			VulkanBuffer& stageBuffer) override;

		static std::shared_ptr<SnapshotAssetTextureLoadTask> build(std::shared_ptr<ImageAssetHeader> inHeader);
	};

	struct ImageAssetTextureLoadTask : public AssetTextureLoadTask
	{
		explicit ImageAssetTextureLoadTask(std::shared_ptr<ImageAssetHeader> inHeader)
			: cacheHeader(inHeader)
		{

		}

		std::shared_ptr<ImageAssetHeader> cacheHeader;

		virtual void finishCallback() override
		{
			imageAssetGPU->setAsyncLoadState(false);
		}

		virtual void uploadFunction(
			uint32_t stageBufferOffset,
			RHICommandBufferBase& commandBuffer,
			VulkanBuffer& stageBuffer) override;

		static std::shared_ptr<ImageAssetTextureLoadTask> build(std::shared_ptr<ImageAssetHeader> inHeader);
	};
}

CEREAL_REGISTER_TYPE(Flower::ImageAssetHeader)
CEREAL_REGISTER_POLYMORPHIC_RELATION(Flower::AssetHeaderInterface, Flower::ImageAssetHeader)

CEREAL_REGISTER_TYPE(Flower::ImageAssetBin)
CEREAL_REGISTER_POLYMORPHIC_RELATION(Flower::AssetBinInterface, Flower::ImageAssetBin)