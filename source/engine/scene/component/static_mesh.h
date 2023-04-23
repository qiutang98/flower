#pragma once
#include "../component.h"
#include <rhi/gpu_asset.h>
#include <rhi/rhi.h>
#include <asset/asset_material.h>
#include <asset/asset_staticmesh.h>
namespace engine
{
	class SceneNode;

	struct MaterialCache
	{
		StandardPBRMaterialHandle handle;
		std::shared_ptr<StandardPBRMaterial> asset;
	};

	class StaticMeshComponent : public Component
	{
	public:
		StaticMeshComponent() = default;
		virtual ~StaticMeshComponent();

		StaticMeshComponent(std::shared_ptr<SceneNode> sceneNode)
			: Component(sceneNode)
		{

		}

		virtual void tick(const RuntimeModuleTickData& tickData) override;

		// Return is move this frame, which toggle tlas rebuild.
		void renderObjectCollect(std::vector<GPUStaticMeshPerObjectData>& collector);

		bool setMesh(const UUID& in, const std::string& staticMeshAssetRelativeRoot, bool bEngineAsset);

		const UUID& getMeshUUID() const { return m_staticMeshUUID; }
		const std::string& getMeshAssetRelativeRoot() const { return m_staticMeshAssetRelativeRoot; }
		const bool isUsingEngineAsset() const { return m_bEngineAsset; }

		uint32_t getVerticesCount() const { return uint32_t(m_cacheGPUMeshAsset->getVerticesCount()); }
		uint32_t getIndicesCount() const { return uint32_t(m_cacheGPUMeshAsset->getIndicesCount()); }
		uint32_t getSubmeshCount() const { return uint32_t(m_perobjectCache.cachePerObjectData.size()); }

		bool isGPUMeshAssetExist() const { return m_cacheGPUMeshAsset != nullptr; }


	private:
		using MaterialUUID = UUID;

		struct PerObjectCache
		{
			// SOA for insert and multi thread.
			std::vector<GPUStaticMeshPerObjectData> cachePerObjectData;
			std::vector<MaterialUUID> cacheMaterialId;

			void clear()
			{
				cachePerObjectData.clear();
				cacheMaterialId.clear();
			}
			void resize(size_t i)
			{
				cachePerObjectData.resize(i);
				cacheMaterialId.resize(i);
			}
		};



	
		// Clear static mesh object info cache.
		void clearCache();

		// Update cache object collect info.
		void updateObjectCollectInfo(const RuntimeModuleTickData* tickData);

		// Load mesh asset by uuid.
		void loadAssetByUUID();

		// async load state prepare.
		void asyncLoadStateHandle();

		// Update cache materials.
		void updateMaterials();

	protected:
		// Mesh already replace?
		bool m_bMeshReplace = true; 

		// Mesh load ready?
		bool m_bMeshReady = false; 

		// Cache gpu mesh asset.
		std::shared_ptr<GPUStaticMeshAsset> m_cacheGPUMeshAsset;

		// This is optional, some asset no exist asset, we store mesh info in gpu mesh asset directly.
		std::weak_ptr<AssetStaticMesh> m_cacheStaticMeshAsset;

		// Cache perobject info.
		PerObjectCache m_perobjectCache;

		// Cache perobject material.
		std::map<MaterialUUID, MaterialCache> m_cachePerObjectMaterials;

	protected:
		ARCHIVE_DECLARE;

		bool m_bEngineAsset = false;
		UUID m_staticMeshUUID = {};
		std::string m_staticMeshAssetRelativeRoot = {};
	};
}


