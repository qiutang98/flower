#pragma once

#include "asset.h"
#include "asset_common.h"
#include "texture_helper.h"
#include <engine/graphics/gpu_asset.h>

namespace engine
{
	class AssetTexture;

	struct AssetTextureImportConfig : public AssetImportConfigInterface
	{
		// Texture is encoded in srgb color space?
		bool bSRGB = false;

		// Generate mipmap for this texture?
		bool bGenerateMipmap = false;

		// Alpha coverage mipmap cutoff.
		float alphaMipmapCutoff = 0.5f;

		// Texture format.
		ETextureFormat format;
	};

	// Load from asset header snapshot data, no compress, cache in lru map.
	struct SnapshotAssetTextureLoadTask : public AssetTextureLoadTask
	{
	public:
		virtual void uploadFunction(
			uint32_t stageBufferOffset,
			void* bufferPtrStart,
			RHICommandBufferBase& commandBuffer,
			VulkanBuffer& stageBuffer) override;

	public:
		std::shared_ptr<AssetTexture> cacheAsset;
	};

	struct AssetTextureCacheLoadTask : public AssetTextureLoadTask
	{
	public:
		virtual void uploadFunction(
			uint32_t stageBufferOffset,
			void* bufferPtrStart,
			RHICommandBufferBase& commandBuffer,
			VulkanBuffer& stageBuffer) override;

	public:
		std::shared_ptr<AssetTexture> cacheAsset;
	};

	class AssetTexture : public AssetInterface
	{
		REGISTER_BODY_DECLARE(AssetInterface);
		friend bool importTextureFromConfigThreadSafe(std::shared_ptr<AssetImportConfigInterface> ptr);
		friend bool loadLdrTexture(std::shared_ptr<AssetImportConfigInterface> ptr);

	public:
		AssetTexture() = default;
		explicit AssetTexture(const AssetSaveInfo& saveInfo);
		virtual ~AssetTexture() = default;

		// ~AssetInterface virtual function.
		virtual EAssetType getType() const override 
		{ 
			return EAssetType::darktexture; 
		}
		virtual void onPostAssetConstruct() override;
		virtual VulkanImage* getSnapshotImage() override;
		// ~AssetInterface virtual function.

		static const AssetReflectionInfo& uiGetAssetReflectionInfo();
		const static AssetTexture* getCDO();

		void initBasicInfo(
			bool bSrgb, 
			uint32_t mipmapCount,
			VkFormat format,
			const math::uvec3& dimension,
			float alphaCutoff)
		{
			m_bSRGB             = bSrgb;
			m_mipmapCount       = mipmapCount;
			m_format            = format;
			m_dimension         = dimension;
			m_alphaMipmapCutoff = alphaCutoff;
		}

	protected:
		// ~AssetInterface virtual function.
		virtual bool saveImpl() override;

		virtual void unloadImpl() override;
		// ~AssetInterface virtual function.

	public:
		bool isSRGB() const { return m_bSRGB; }

		VulkanImage* getImage();
		std::weak_ptr<GPUImageAsset> getGPUImage();

		uint32_t getMipmapCount() const { return m_mipmapCount; }
		VkFormat getFormat() const { return m_format; }

		const auto& getDimension() const { return m_dimension; }

		std::shared_ptr<SnapshotAssetTextureLoadTask> buildSnapShotLoadTask();
		std::shared_ptr<AssetTextureCacheLoadTask> buildTextureLoadTask();

		void buildSnapshot(std::vector<uint8_t>& data, unsigned char* pixels, int numChannel);

		float getAlphaMipmapCutOff() const { return m_alphaMipmapCutoff; }

	private:
		std::weak_ptr<GPUImageAsset> m_cacheSnapshotImage { };
		std::weak_ptr<GPUImageAsset> m_cacheImage{ };

	private:
		// Texture under srgb encode?
		bool m_bSRGB;

		// Mipmap infos.
		uint32_t m_mipmapCount;

		// Texture dimension.
		math::uvec3 m_dimension;

		// Texture format.
		VkFormat m_format;

		// Texture mipmap alpha coverage cutoff.
		float m_alphaMipmapCutoff;
	};

	// Asset texture binary.
	struct AssetTextureBin
	{
		std::vector<std::vector<uint8_t>> mipmapDatas;

		template<class Archive> 
		void serialize(Archive& archive, std::uint32_t const version)
		{
			archive(mipmapDatas);
		}
	};


}