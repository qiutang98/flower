#pragma once

#include "asset_common.h"
#include "asset_archive.h"
#include <util/shader_struct.h>

namespace engine
{
	struct StaticMeshBin
	{

		std::vector<VertexPosition>   positions;
		std::vector<VertexNormal> normals;
		std::vector<VertexTangent> tangents;
		std::vector<VertexUv0> uv0s;
		std::vector<VertexIndexType> indices;

		template<class Archive> void serialize(Archive& archive)
		{
			archive(normals, tangents, uv0s, positions, indices);
		}
	};

	class AssetStaticMesh : public AssetInterface
	{
	public:
		struct ImportConfig
		{
			
		};

		AssetStaticMesh() = default;
		AssetStaticMesh(const std::string& assetNameUtf8, const std::string& assetRelativeRootProjectPathUtf8);

		virtual EAssetType getType() const override { return EAssetType::StaticMesh; }
		virtual const char* getSuffix() const { return ".staticmesh"; }


		const auto& getSubMeshes() const { return m_subMeshes; }
		size_t getVerticesCount() const { return m_verticesCount; }
		size_t getIndicesCount() const { return m_indicesCount; }

		static bool buildFromConfigs(
			const ImportConfig& config,
			const std::filesystem::path& projectRootPath,
			const std::filesystem::path& savePath,
			const std::filesystem::path& srcPath
		);
	protected:
		// Serialize field.
		ARCHIVE_DECLARE;

		std::vector<StaticMeshSubMesh> m_subMeshes = {};
		size_t m_indicesCount;
		size_t m_verticesCount;

	};

	struct AssetStaticMeshLoadFromCacheTask : public AssetStaticMeshLoadTask
	{
		AssetStaticMeshLoadFromCacheTask() { }

		std::shared_ptr<AssetStaticMesh> cachePtr;
		virtual void uploadFunction(uint32_t stageBufferOffset, void* bufferPtrStart, RHICommandBufferBase& commandBuffer, VulkanBuffer& stageBuffer) override;
		static std::shared_ptr<AssetStaticMeshLoadFromCacheTask> build(VulkanContext* context, std::shared_ptr<AssetStaticMesh> meta);
	};
}

ASSET_ARCHIVE_IMPL_INHERIT(AssetStaticMesh, AssetInterface)
{
	ARCHIVE_NVP_DEFAULT(m_subMeshes);
	ARCHIVE_NVP_DEFAULT(m_indicesCount);
	ARCHIVE_NVP_DEFAULT(m_verticesCount);
}
ASSET_ARCHIVE_END