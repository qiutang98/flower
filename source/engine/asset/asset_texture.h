#pragma once

#include "asset_common.h"
#include "asset_archive.h"

namespace engine
{
	class AssetTexture : public AssetInterface
	{
	public:


		struct ImportConfig
		{
			bool bSRGB = false;
			bool bGenerateMipmap = false;
			bool bCompressed = false;
			float cutoffAlpha = 1.0f;
			bool bHalfFixed = false;
			bool bExr = false; // Exr use 32bit depth format.
		
			enum class EChannel
			{
				RGBA,
				R,
				G,
				B,
				A,
				RGB,
			};
			EChannel channel = EChannel::RGBA;
		};


		AssetTexture() = default;
		AssetTexture(const std::string& assetNameUtf8, const std::string& assetRelativeRootProjectPathUtf8);

		virtual EAssetType getType() const override { return EAssetType::Texture; }
		virtual const char* getSuffix() const { return ".texture"; }

		static bool buildFromConfigs(
			const ImportConfig& config,
			const std::filesystem::path& projectRootPath,
			const std::filesystem::path& savePath, 
			const std::filesystem::path& srcPath,
			AssetTexture& outMeta,
			const UUID& overriderUUID = {});

		uint32_t getMipmapCount() const { return m_mipmapCount; }
		bool isSrgb() const { return m_bSRGB; }

		uint32_t getHeight() const { return m_height; }
		uint32_t getWidth() const { return m_width; }
		uint32_t getDepth() const { return m_depth; }

		VkFormat getFormat() const;

		float getAlphaCutoff() const { return m_alphaCutoff; }
	protected:
		// Serialize field.
		ARCHIVE_DECLARE;

		bool m_bSRGB;
		bool m_bMipmap;
		bool m_bHdr;
		bool m_bCompressed;

		// Texture mipmap count.
		uint32_t m_mipmapCount;
		float m_alphaCutoff;
		uint32_t m_format;

		// Texture dimension.
		uint32_t m_width;
		uint32_t m_height;
		uint32_t m_depth;
	};

	struct AssetTextureBin
	{
		// Mipmap datas.
		std::vector<std::vector<uint8_t>> mipmapDatas;

		template<class Archive> void serialize(Archive& archive, std::uint32_t const version)
		{
			archive(mipmapDatas);
		}
	};

	struct AssetTextureCacheLoadTask : public AssetTextureLoadTask
	{
		explicit AssetTextureCacheLoadTask(std::shared_ptr<AssetTexture> inAsset)
			: cacheAsset(inAsset)
		{

		}

		std::shared_ptr<AssetTexture> cacheAsset;

		virtual void uploadFunction(
			uint32_t stageBufferOffset,
			void* bufferPtrStart,
			RHICommandBufferBase& commandBuffer,
			VulkanBuffer& stageBuffer) override;

		static std::shared_ptr<AssetTextureCacheLoadTask> build(VulkanContext* context, std::shared_ptr<AssetTexture> asset);
	};
}

ASSET_ARCHIVE_IMPL_INHERIT(AssetTexture, AssetInterface)
{
	ARCHIVE_NVP_DEFAULT(m_width);
	ARCHIVE_NVP_DEFAULT(m_height);
	ARCHIVE_NVP_DEFAULT(m_depth);
	ARCHIVE_NVP_DEFAULT(m_bSRGB);
	ARCHIVE_NVP_DEFAULT(m_bMipmap);
	ARCHIVE_NVP_DEFAULT(m_bCompressed);
	ARCHIVE_NVP_DEFAULT(m_bHdr);
	ARCHIVE_NVP_DEFAULT(m_mipmapCount);
	ARCHIVE_NVP_DEFAULT(m_alphaCutoff);
	ARCHIVE_NVP_DEFAULT(m_format);
}
ASSET_ARCHIVE_END