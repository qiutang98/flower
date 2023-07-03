#pragma once

#include <util/util.h>
#include <rhi/rhi.h>

#include "asset_archive.h"
#include <lz4.h>


namespace engine
{
	extern bool isEngineMetaAsset(const std::string& extension);
	extern bool isAssetTextureMeta(const std::string& extension);
	extern bool isAssetStaticMeshMeta(const std::string& extension);
	extern bool isAssetMaterialMeta(const std::string& extension);
	extern bool isAssetSceneMeta(const std::string& extension);
	extern bool isAssetPMXMeta(const std::string& extension);
	extern bool isAssetVMDMeta(const std::string& extension);
	extern bool isAssetWaveMeta(const std::string& extension);

	// Engine asset type.
	enum class EAssetType
	{
		Texture = 0, // Texture asset.
		StaticMesh,  // StaticMesh asset.
		Material,    // Material asset.
		Scene,       // Scene.
		PMX,         // Pmx file.
		VMD,         // Vmd file.
		Wave,
		Max,
	};

	inline static std::string buildRelativePathUtf8(
		const std::filesystem::path& projectRootPath, 
		const std::filesystem::path& savePath)
	{
		const std::u16string assetProjectRootPath = std::filesystem::absolute(projectRootPath).u16string();
		const std::u16string assetSavePath = std::filesystem::absolute(savePath).u16string();

		return utf8::utf16to8(assetSavePath.substr(assetProjectRootPath.size()));
	}

	// Asset interface.
	class AssetInterface : public std::enable_shared_from_this<AssetInterface>
	{
	public:
		AssetInterface() = default;

		AssetInterface(const std::string& assetNameUtf8, const std::string& assetRelativeRootProjectPathUtf8)
			: m_assetNameUtf8(assetNameUtf8)
			, m_assetRelativePathUtf8(assetRelativeRootProjectPathUtf8)
		{

		}

		// Draw asset config pannel.
		virtual bool drawAssetConfig() { return false; };

		virtual EAssetType getType() const { return EAssetType::Max; }
		virtual const char* getSuffix() const = 0;

		bool saveAction();
		bool savePathUnvalid() const { return m_assetNameUtf8.empty() || m_assetRelativePathUtf8.empty(); }

		const UUID& getUUID() const { return m_uuid; }
		void setUUID(const UUID& uuid) { m_uuid = uuid; }

		const std::string& getNameUtf8() const { return m_assetNameUtf8; }
		const std::string& getRelativePathUtf8() const { return m_assetRelativePathUtf8; }

		void setNameUtf8(const std::string& in) { m_assetNameUtf8 = in; }
		void setRelativePathUtf8(const std::string& in) { m_assetRelativePathUtf8 = in; }

		uint32_t getSnapshotWidth() const { return m_widthSnapShot; }
		uint32_t getSnapshotHeight() const { return m_heightSnapShot; }

		void buildSnapshot(uint32_t width, uint32_t height, const uint8_t* buffer);

		inline bool existSnapshot() const
		{
			return !m_snapshotData.empty() && m_widthSnapShot && m_heightSnapShot;
		}

		const auto& getSnapshotData() const { return m_snapshotData; }

		const UUID& getSnapshotUUID() const { return m_snapshotUUID; }

		std::shared_ptr<GPUImageAsset> getOrCreateLRUSnapShot(VulkanContext* ct);

		// This scene already edit? need save?
		bool isDirty() const { return m_bDirty; }

		// Set scene dirty state.
		bool setDirty(bool bDirty = true);

		// Get shared ptr.
		template<typename T>
		std::shared_ptr<T> getptr()
		{
			static_assert(std::is_base_of_v<AssetInterface, T>, "T must derive from AssetInterface!");
			return std::static_pointer_cast<T>(AssetInterface::shared_from_this());
		}

		std::filesystem::path getSavePath() const;

	protected:
		virtual bool saveActionImpl() 
		{
			LOG_WARN("Unimplement save path for asset type, fix me!");
			return false; 
		}

		inline static void quantifySnapshotDim(uint32_t& widthSnapShot, uint32_t& heightSnapShot, uint32_t texWidth, uint32_t texHeight)
		{
			if (texWidth >= kMaxSnapshotDim || texHeight >= kMaxSnapshotDim)
			{
				if (texWidth > texHeight)
				{
					widthSnapShot = kMaxSnapshotDim;
					heightSnapShot = texHeight / (texWidth / kMaxSnapshotDim);
				}
				else if (texHeight > texWidth)
				{
					heightSnapShot = kMaxSnapshotDim;
					widthSnapShot = texWidth / (texHeight / kMaxSnapshotDim);
				}
				else
				{
					widthSnapShot = kMaxSnapshotDim;
					heightSnapShot = kMaxSnapshotDim;
				}
			}
			else
			{
				heightSnapShot = texHeight;
				widthSnapShot = texWidth;
			}
		}

		inline static uint32_t kMaxSnapshotDim = 128u;

	protected:
		// Is asset dirty?
		bool m_bDirty = false;




	protected:
		// Serialize field.
		ARCHIVE_DECLARE;

		// Asset uuid.
		UUID m_uuid = buildUUID();

		// This asset name, exclusive suffix, utf8 encode.
		std::string m_assetNameUtf8;

		// This asset relative project root path, utf8 encode.
		std::string m_assetRelativePathUtf8;

		// Asset snap shot
		UUID m_snapshotUUID = buildUUID();
		uint32_t m_widthSnapShot = 0;
		uint32_t m_heightSnapShot = 0;
		std::vector<uint8_t> m_snapshotData = {};
	};

	// Load from asset header snapshot data, no compress, cache in lru map.
	struct SnapshotAssetTextureLoadTask : public AssetTextureLoadTask
	{
		explicit SnapshotAssetTextureLoadTask(std::shared_ptr<AssetInterface> inAsset)
			: cacheAsset(inAsset)
		{

		}

		std::shared_ptr<AssetInterface> cacheAsset;

		virtual void uploadFunction(
			uint32_t stageBufferOffset,
			void* bufferPtrStart, 
			RHICommandBufferBase& commandBuffer, 
			VulkanBuffer& stageBuffer) override;

		static std::shared_ptr<SnapshotAssetTextureLoadTask> build(VulkanContext* context, std::shared_ptr<AssetInterface> asset);
	};

	struct AssetCompressionHelper
	{
		int originalSize;
		int compressionSize;

		template<class Archive> void serialize(Archive& archive)
		{
			archive(originalSize, compressionSize);
		}
	};

	static inline bool loadAssetBinaryWithDecompression(std::vector<uint8_t>& out, const std::filesystem::path& rawSavePath)
	{
		if (!std::filesystem::exists(rawSavePath))
		{
			LOG_ERROR("Binary data {} miss!", utf8::utf16to8(rawSavePath.u16string()));
			return false;
		}

		std::ifstream is(rawSavePath, std::ios::binary);
		cereal::BinaryInputArchive archive(is);

		AssetCompressionHelper sizeHelper;

		std::vector<uint8_t> compressionData;
		archive(sizeHelper, compressionData);

		// Resize to src data.
		out.resize(sizeHelper.originalSize);

		LZ4_decompress_safe((const char*)compressionData.data(), (char*)out.data(), sizeHelper.compressionSize, sizeHelper.originalSize);
		return true;
	}

	static inline bool saveAssetBinaryWithCompression(const uint8_t* out, int size, const std::filesystem::path& savePath, const char* suffix)
	{
		std::filesystem::path rawSavePath = savePath;
		rawSavePath += suffix;

		if (std::filesystem::exists(rawSavePath))
		{
			LOG_ERROR("Binary data {} already exist, make sure never import save resource at same folder!", utf8::utf16to8(rawSavePath.u16string()));
			return false;
		}

		// Save to disk.
		std::ofstream os(rawSavePath, std::ios::binary);
		cereal::BinaryOutputArchive archive(os);

		// LZ4 compression.
		std::vector<uint8_t> compressionData;

		// Compress and shrink.
		auto compressStaging = LZ4_compressBound(size);
		compressionData.resize(compressStaging);
		auto compressedSize = LZ4_compress_default((const char*)out, (char*)compressionData.data(), size, compressStaging);
		compressionData.resize(compressedSize);

		AssetCompressionHelper sizeHelper
		{
			.originalSize = size,
			.compressionSize = compressedSize,
		};

		archive(sizeHelper, compressionData);
		return true;
	}

	static inline bool saveAssetBinaryWithCompression(const std::vector<uint8_t>& out, const std::filesystem::path& savePath, const char* suffix)
	{
		return saveAssetBinaryWithCompression(out.data(), (int)out.size(), savePath, suffix);
	}



	template<typename T>
	inline bool loadAsset(T& out, const std::filesystem::path& savePath)
	{
		if (!std::filesystem::exists(savePath))
		{
			LOG_ERROR("Asset data {} miss!", utf8::utf16to8(savePath.u16string()));
			return false;
		}

		AssetCompressionHelper sizeHelper;
		std::vector<char> compressionData;
		{
			std::ifstream is(savePath, std::ios::binary);
			cereal::BinaryInputArchive archive(is);
			archive(sizeHelper, compressionData);
		}

		std::vector<char> decompressionData(sizeHelper.originalSize);
		CHECK(sizeHelper.compressionSize == compressionData.size());
		int decompressSize = LZ4_decompress_safe(compressionData.data(), decompressionData.data(), sizeHelper.compressionSize, sizeHelper.originalSize);
		CHECK(decompressSize == sizeHelper.originalSize);

		// Exist copy-construct.
		{
			std::string str(decompressionData.data(), decompressionData.size());
			std::stringstream ss;
			ss << std::move(str);
			cereal::BinaryInputArchive archive(ss);
			archive(out);
		}
		
		return true;
	}



	// Save as unique_ptr
	template<typename T>
	inline bool saveAsset(const T& in, const std::filesystem::path& savePath, const char* suffix, bool bRequireNoExist = true)
	{
		std::filesystem::path rawSavePath = savePath;
		rawSavePath += suffix;

		if (bRequireNoExist && std::filesystem::exists(rawSavePath))
		{
			LOG_ERROR("Meta data {} already exist, make sure never import save resource at same folder!", utf8::utf16to8(rawSavePath.u16string()));
			return false;
		}
		
		AssetCompressionHelper sizeHelper;
		std::string originalData;
		// Exist Copy construct.
		{
			std::stringstream ss;
			cereal::BinaryOutputArchive archive(ss);
			archive(in);

			originalData = std::move(ss.str());
		}

		std::vector<char> compressedData(LZ4_compressBound((int)originalData.size()));
		sizeHelper.compressionSize = LZ4_compress_default(originalData.c_str(), compressedData.data(), (int)originalData.size(), (int)compressedData.size());

		compressedData.resize(sizeHelper.compressionSize);
		sizeHelper.originalSize = (int)originalData.size();

		{
			std::ofstream os(rawSavePath, std::ios::binary);
			cereal::BinaryOutputArchive archive(os);
			archive(sizeHelper, compressedData);
		}

		return true;
	}

	template<typename T>
	inline bool saveAssetMeta(const T& in, const std::filesystem::path& savePath, const char* suffix, bool bRequireNoExist = true)
	{
		// See insertAsset in asset system.
		static_assert(std::is_base_of_v<AssetInterface, T>, "T must derived from AssetInterface.");

		// Exist one copy?
		std::shared_ptr<AssetInterface> save = std::make_shared<T>(in);
		return saveAsset(save, savePath, suffix, bRequireNoExist);
	}
}

ASSET_ARCHIVE_IMPL(AssetInterface)
{
	ARCHIVE_NVP_DEFAULT(m_assetNameUtf8);
	ARCHIVE_NVP_DEFAULT(m_assetRelativePathUtf8);
	ARCHIVE_NVP_DEFAULT(m_widthSnapShot);
	ARCHIVE_NVP_DEFAULT(m_heightSnapShot);
	ARCHIVE_NVP_DEFAULT(m_snapshotData);
	ARCHIVE_NVP_DEFAULT(m_uuid);
	ARCHIVE_NVP_DEFAULT(m_snapshotUUID);
}
ASSET_ARCHIVE_END