#pragma once

#include "../utils/utils.h"
#include <engine/graphics/resource.h>
#include <fstream>
#include <lz4.h>
#define CEREAL_THREAD_SAFE 1

#include <cereal/access.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/array.hpp>
#include <cereal/cereal.hpp> 
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/unordered_set.hpp>
#include <cereal/types/list.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/set.hpp>
#include <cereal/types/map.hpp>

#include <rttr/type.h>
#include <rttr/registration_friend.h>

#include <nlohmann/json.hpp>
#include <tinygltf/tiny_gltf.h>
#include <profile/profile.h>
namespace glm
{
	template<class Archive> void serialize(Archive& archive, glm::vec2&  v) { archive(v.x, v.y);               }
	template<class Archive> void serialize(Archive& archive, glm::vec3&  v) { archive(v.x, v.y, v.z);          }
	template<class Archive> void serialize(Archive& archive, glm::vec4&  v) { archive(v.x, v.y, v.z, v.w);     }
	template<class Archive> void serialize(Archive& archive, glm::ivec2& v) { archive(v.x, v.y);               }
	template<class Archive> void serialize(Archive& archive, glm::ivec3& v) { archive(v.x, v.y, v.z);          }
	template<class Archive> void serialize(Archive& archive, glm::ivec4& v) { archive(v.x, v.y, v.z, v.w);     }
	template<class Archive> void serialize(Archive& archive, glm::uvec2& v) { archive(v.x, v.y);               }
	template<class Archive> void serialize(Archive& archive, glm::uvec3& v) { archive(v.x, v.y, v.z);          }
	template<class Archive> void serialize(Archive& archive, glm::uvec4& v) { archive(v.x, v.y, v.z, v.w);     }
	template<class Archive> void serialize(Archive& archive, glm::dvec2& v) { archive(v.x, v.y);               }
	template<class Archive> void serialize(Archive& archive, glm::dvec3& v) { archive(v.x, v.y, v.z);          }
	template<class Archive> void serialize(Archive& archive, glm::dvec4& v) { archive(v.x, v.y, v.z, v.w);     }
	template<class Archive> void serialize(Archive& archive, glm::mat2&  m) { archive(m[0], m[1]);             }
	template<class Archive> void serialize(Archive& archive, glm::dmat2& m) { archive(m[0], m[1]);             }
	template<class Archive> void serialize(Archive& archive, glm::mat3&  m) { archive(m[0], m[1], m[2]);       }
	template<class Archive> void serialize(Archive& archive, glm::mat4&  m) { archive(m[0], m[1], m[2], m[3]); }
	template<class Archive> void serialize(Archive& archive, glm::dmat4& m) { archive(m[0], m[1], m[2], m[3]); }
	template<class Archive> void serialize(Archive& archive, glm::quat&  q) { archive(q.x, q.y, q.z, q.w);     }
	template<class Archive> void serialize(Archive& archive, glm::dquat& q) { archive(q.x, q.y, q.z, q.w);     }
}

namespace engine
{
	extern const uint32_t kAssetVersion;
}

#define ARCHIVE_DECLARE                                                                  \
	friend class cereal::access;                                                         \
	template<class Archive>                                                              \
	void serialize(Archive& archive, std::uint32_t const version);

#define ARCHIVE_NVP_DEFAULT(Member) archive(cereal::make_nvp(#Member, Member))

// Version and type registry.
#define ASSET_ARCHIVE_IMPL_BASIC(AssetNameXX, Version)                                   \
	CEREAL_CLASS_VERSION(engine::AssetNameXX, Version);                                  \
	CEREAL_REGISTER_TYPE_WITH_NAME(engine::AssetNameXX, "engine::"#AssetNameXX);

// Virtual children class.
#define registerClassMemberInherit(AssetNameXX, AssetNamePP)                             \
	ASSET_ARCHIVE_IMPL_BASIC(AssetNameXX, engine::kAssetVersion);                        \
	CEREAL_REGISTER_POLYMORPHIC_RELATION(engine::AssetNamePP, engine::AssetNameXX)       \
	template<class Archive>                                                              \
	void engine::AssetNameXX::serialize(Archive& archive, std::uint32_t const version) { \
	archive(cereal::base_class<engine::AssetNamePP>(this));

// Baisc class.
#define registerClassMember(AssetNameXX)                                                 \
	ASSET_ARCHIVE_IMPL_BASIC(AssetNameXX, engine::kAssetVersion);                        \
	template<class Archive>                                                              \
	void engine::AssetNameXX::serialize(Archive& archive, std::uint32_t const version)

#define registerPODClassMember(AssetNameXX)                                              \
	CEREAL_CLASS_VERSION(engine::AssetNameXX, engine::kAssetVersion);                    \
	template<class Archive>                                                              \
	void engine::AssetNameXX::serialize(Archive& archive, std::uint32_t const version)

#define REGISTER_BODY_DECLARE(...)  \
	ARCHIVE_DECLARE             \
	RTTR_ENABLE(__VA_ARGS__);   \
	RTTR_REGISTRATION_FRIEND();

#define ARCHIVE_ENUM_CLASS(value)   \
	{ size_t enum__type__##value = (size_t)value; \
	ARCHIVE_NVP_DEFAULT(enum__type__##value); \
	value = (decltype(value))(enum__type__##value); }

namespace engine
{

	struct AssetImportConfigInterface
	{
		using ImportAssetPath = std::pair<std::filesystem::path, std::filesystem::path>;
		ImportAssetPath path;
	};

	enum class EAssetType
	{
		darkscene = 0,
		darktexture,
		darkstaticmesh,
		darkmaterial,

		max
	};

	struct AssetReflectionInfo
	{
		using ImportConfigPtr = std::shared_ptr<AssetImportConfigInterface>;

		std::string name;
		std::string icon;
		std::string decoratedName;

		struct ImportConfig
		{
			bool bImportable;
			std::string importRawAssetExtension;
			std::function<ImportConfigPtr()> buildAssetImportConfig = nullptr;
			std::function<void(ImportConfigPtr)> drawAssetImportConfig = nullptr;
			std::function<bool(ImportConfigPtr)> importAssetFromConfigThreadSafe = nullptr;
		} importConfig;
	};

	class AssetSaveInfo
	{
		ARCHIVE_DECLARE;

	public:
		static const u8str kTempFolderStartChar;
		static const u8str kBuiltinFileStartChar;

		AssetSaveInfo() = default;
		explicit AssetSaveInfo(const u8str& name, const u8str& storeFolder);

		static AssetSaveInfo buildTemp(const u8str& name);
		static AssetSaveInfo buildRelativeProject(const std::filesystem::path& savePath);
		
	public:
		const UUID getUUID() const 
		{
			if (isBuiltin())
			{
				return m_name;
			}
			return m_storePath; 
		}

		const std::u16string getStorePath() const 
		{ 
			return utf8::utf8to16(m_storePath); 
		}

		UUID getSnapshotUUID() const
		{
			if (isTemp())
			{
				// Temp info no exist snapshot.
				return getUUID();
			}

			const size_t hashID = std::hash<UUID>{}(getUUID() + "SnapShotImage");
			return std::to_string(hashID) + "_SnapShotImage";
		}

		UUID getBinUUID() const
		{
			if (isTemp())
			{
				// Temp info no exist bin.
				return getUUID();
			}

			const size_t hashID = std::hash<UUID>{}(getUUID() + "BinFile");
			return std::to_string(hashID) + "_BinFile";
		}

		const u8str& getStorePathU8() const 
		{
			return m_storePath;
		}

		const std::filesystem::path toPath() const;

		const u8str& getName() const { return m_name; }
		const u8str& getStoreFolder() const { return m_storeFolder; }

		void setName(const u8str& newValue);
		void setStoreFolder(const u8str& newValue);

		bool empty() const
		{
			return m_name.empty() || m_storePath.empty();
		}

		bool isTemp() const;
		bool isBuiltin() const;

		// Save info can use for new asset create path or not.
		bool canUseForCreateNewAsset() const;

		// This save info path already in disk.
		bool alreadyInDisk() const;


		auto operator<=>(const AssetSaveInfo&) const = default;

	private:
		void updateStorePath();

	private:
		// Asset name.
		u8str m_name = {};

		// Store folder relative to project asset folder.
		u8str m_storeFolder = {};

		// Store path relative to project asset folder.
		u8str m_storePath = {};
	};

	struct AssetSnapshot : NonCopyable
	{
		bool empty() const
		{
			return data.empty();
		}

		math::uvec2 dimension = {0, 0};
		std::vector<uint8_t> data = {};

		template<class Archive>
		void serialize(Archive& archive)
		{
			archive(dimension, data);
		}

	public:
		inline static uint32_t kMaxSnapshotDim = 128u;

		static void quantifySnapshotDim(
			uint32_t& widthSnapShot, 
			uint32_t& heightSnapShot, 
			uint32_t texWidth, 
			uint32_t texHeight)
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
	};

	class AssetInterface : public std::enable_shared_from_this<AssetInterface>
	{
		REGISTER_BODY_DECLARE();

	public:
		AssetInterface() = default;
		explicit AssetInterface(const AssetSaveInfo& saveInfo);
		virtual ~AssetInterface() = default;

		// ~AssetInterface virtual function.
		virtual EAssetType getType() const  = 0;
		virtual void onPostAssetConstruct() = 0; // Call back when call AssetManager::createAsset
		virtual VulkanImage* getSnapshotImage();
		// ~AssetInterface virtual function.

		// Get suffix of asset.
		std::string getSuffix() const;

		// Get asset store path.
		const std::u16string& getStorePath() const { return m_saveInfo.getStorePath(); }

		// Get asset store name.
		const u8str& getName() const { return m_saveInfo.getName(); }

		// Get asset store folder.
		const u8str& getStoreFolder() const { return m_saveInfo.getStoreFolder(); }

		bool isSavePathEmpty() const { return m_saveInfo.empty(); }

		// Save asset.
		bool save();

		void unload();

		// Asset is dirty or not.
		bool isDirty() const { return m_bDirty; }

		// Mark asset is dirty.
		bool markDirty();

		// Get shared ptr.
		template<typename T> std::shared_ptr<T> getptr()
		{
			static_assert(std::is_base_of_v<AssetInterface, T>, "T must derive from AssetInterface!");
			return std::static_pointer_cast<T>(AssetInterface::shared_from_this());
		}

		// Get asset save path.
		std::filesystem::path getSavePath() const;

		const AssetSaveInfo& getSaveInfo() const { return m_saveInfo; }

		void discardChanged();

		bool changeSaveInfo(const AssetSaveInfo& newInfo);

		UUID getSnapshotUUID() const;
		std::filesystem::path getSnapshotPath() const;

		std::filesystem::path getRawAssetPath() const;

		UUID getBinUUID() const;
		std::filesystem::path getBinPath() const;

	protected:
		// ~AssetInterface virtual function.
		virtual bool saveImpl() = 0;

		virtual void unloadImpl() = 0;
		// ~AssetInterface virtual function.

	private:
		// Asset is dirty or not.
		bool m_bDirty = false;

	protected:
		AssetSaveInfo m_saveInfo = { };

		// Raw asset path relative to asset folder.
		u8str m_rawAssetPath = {};
	};

	struct AssetCompressionHelper
	{
		int originalSize;
		int compressionSize;

		template<class Archive>
		void serialize(Archive& archive)
		{
			archive(originalSize, compressionSize);
		}
	};

	extern bool loadAssetBinaryWithDecompression(std::vector<uint8_t>& out, const std::filesystem::path& rawSavePath);
	extern bool saveAssetBinaryWithCompression(const uint8_t* out, int size, const std::filesystem::path& savePath, const char* suffix);

	inline bool saveAssetBinaryWithCompression(const std::vector<uint8_t>& out, const std::filesystem::path& savePath, const char* suffix)
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

		// Exist copsaveAssety-construct.
		{
			std::string str(decompressionData.data(), decompressionData.size());
			std::stringstream ss;
			ss << std::move(str);
			cereal::BinaryInputArchive archive(ss);
			archive(out);
		}

		return true;
	}

	template<typename T>
	inline bool saveAsset(const T& in, const std::filesystem::path& savePath, bool bRequireNoExist = true)
	{
		std::filesystem::path rawSavePath = savePath;
		if (bRequireNoExist && std::filesystem::exists(rawSavePath))
		{
			LOG_ERROR("Meta data {} already exist, make sure never import save resource at same folder!",
				utf8::utf16to8(rawSavePath.u16string()));
			return false;
		}

		AssetCompressionHelper sizeHelper;
		std::string originalData;
		{
			std::stringstream ss;
			cereal::BinaryOutputArchive archive(ss);
			archive(in);
			originalData = std::move(ss.str());
		}

		std::vector<char> compressedData(LZ4_compressBound((int)originalData.size()));
		sizeHelper.compressionSize = LZ4_compress_default(
			originalData.c_str(),
			compressedData.data(),
			(int)originalData.size(),
			(int)compressedData.size());

		compressedData.resize(sizeHelper.compressionSize);
		sizeHelper.originalSize = (int)originalData.size();
		{
			std::ofstream os(rawSavePath, std::ios::binary);
			cereal::BinaryOutputArchive archive(os);
			archive(sizeHelper, compressedData);
		}
		return true;
	}

	struct StaticMeshRenderBounds
	{
		ARCHIVE_DECLARE;

		// AABB min = origin - extents.
		// AABB max = origin + extents.
		// AABB center = origin.
		math::vec3 origin;
		math::vec3 extents;

		float radius;
	};

	struct StaticMeshSubMesh
	{
		ARCHIVE_DECLARE;

		uint32_t indicesStart = 0;
		uint32_t indicesCount = 0;

		// Material of this submesh.
		UUID material = {};
		StaticMeshRenderBounds bounds = {};
	};

	// Standard index type in this engine.
	using VertexIndexType = uint32_t;

	// No sure which vertex layout is better.
	// We default use seperate method instead of interleave.
	// https://frostbite-wp-prd.s3.amazonaws.com/wp-content/uploads/2016/03/29204330/GDC_2016_Compute.pdf 
	// https://developer.android.com/games/optimize/vertex-data-management?hl=zh-tw
	using VertexPosition = math::vec3;
	static_assert(sizeof(VertexPosition) == sizeof(float) * 3);

	using VertexNormal = math::vec3;
	static_assert(sizeof(VertexNormal) == sizeof(float) * 3);

	using VertexTangent = math::vec4;
	static_assert(sizeof(VertexTangent) == sizeof(float) * 4);

	using VertexUv0 = math::vec2;
	static_assert(sizeof(VertexUv0) == sizeof(float) * 2);

	struct StaticMeshBin
	{
		ARCHIVE_DECLARE;

		std::vector<VertexPosition> positions;
		std::vector<VertexNormal> normals;
		std::vector<VertexTangent> tangents;
		std::vector<VertexUv0> uv0s;
		std::vector<VertexIndexType> indices;
	};
}