#pragma once

#include "asset.h"
#include "asset_common.h"
#include "../graphics/gpu_asset.h"
#include <common_header.h>

namespace engine
{
	struct AssetStaticMeshImportConfig : public AssetImportConfigInterface
	{
		bool bImportMaterial = false;
	};

	class AssetStaticMesh : public AssetInterface
	{
		REGISTER_BODY_DECLARE(AssetInterface);

		friend class AssimpStaticMeshImporter;
		friend bool importStaticMeshFromConfigThreadSafe(std::shared_ptr<AssetImportConfigInterface> ptr);

	public:
		AssetStaticMesh() = default;
		virtual ~AssetStaticMesh() = default;

		explicit AssetStaticMesh(const AssetSaveInfo& saveInfo);

		// ~AssetInterface virtual function.
		virtual EAssetType getType() const override { return EAssetType::darkstaticmesh; }
		virtual void onPostAssetConstruct() override;
		// ~AssetInterface virtual function.

		// ~Asset import reflection functions.
		static const AssetReflectionInfo& uiGetAssetReflectionInfo();
		const static AssetStaticMesh* getCDO();
		// ~Asset import reflection functions.

	protected:
		// ~AssetInterface virtual function.
		virtual bool saveImpl() override;
		virtual void unloadImpl() override;
		// ~AssetInterface virtual function.

	public:
		const auto& getSubMeshes() const { return m_subMeshes; }
		size_t getVerticesCount() const { return m_verticesCount; }
		size_t getIndicesCount() const { return m_indicesCount; }

		static bool isStaticMesh(const char* ext)
		{
			if (ext == getCDO()->getSuffix())
			{
				return true;
			}
			return false;
		}

		std::shared_ptr<GPUStaticMeshAsset> getGPUAsset();

		const vec3& getMinPosition() const { return m_minPosition; }
		const vec3& getMaxPosition() const { return m_maxPosition; }

	protected:
		std::weak_ptr<GPUStaticMeshAsset> m_gpuWeakPtr = {};


	private:
		std::vector<StaticMeshSubMesh> m_subMeshes = {};
		size_t m_indicesCount;
		size_t m_verticesCount;

		// AABB bounds.
		math::vec3 m_minPosition = {};
		math::vec3 m_maxPosition = {};
	};

	struct AssetStaticMeshLoadFromCacheTask : public AssetStaticMeshLoadTask
	{
	public:
		virtual void uploadFunction(
			uint32_t stageBufferOffset, 
			void* bufferPtrStart, 
			RHICommandBufferBase& commandBuffer, 
			VulkanBuffer& stageBuffer) override;

		static std::shared_ptr<AssetStaticMeshLoadFromCacheTask> 
			build(std::shared_ptr<AssetStaticMesh> meta);

	public:
		std::shared_ptr<AssetStaticMesh> cachePtr;
	};
}