#pragma once
#include "../component.h"
#include <asset/asset_staticmesh.h>
#include "../../graphics/context.h"
#include <asset/asset_material.h>

namespace engine
{
	class RenderScene;

	struct MaterialCache
	{
		BSDFMaterialTextureHandle    handle;
		std::shared_ptr<AssetMaterial> asset;
	};

	class StaticMeshComponent : public RenderableComponent
	{
		REGISTER_BODY_DECLARE(RenderableComponent);
	public:
		StaticMeshComponent() = default;
		StaticMeshComponent(std::shared_ptr<SceneNode> sceneNode) : RenderableComponent(sceneNode) { }
		virtual ~StaticMeshComponent() = default;

		virtual bool uiDrawComponent() override;
		static const UIComponentReflectionDetailed& uiComponentReflection();

		virtual void tick(const RuntimeModuleTickData& tickData) override;

	public:
		bool setAssetUUID(const UUID& in);
		UUID getAssetUUID() const { return m_assetUUID; }

		uint32_t getSubmeshCount()  const;
		uint32_t getVerticesCount() const;
		uint32_t getIndicesCount()  const;

		void collectRenderObject(RenderScene& renderScene);

	private:
		void clearCache();
		void buildCacheSync();

		void updateMaterials();

	protected:



		using MaterialUUID = UUID;

		// Cache perobject info.
		struct MeshInfoCache
		{
			bool bNewlyCreated = true;

			std::weak_ptr<AssetStaticMesh> assetWeakPtr;
			std::shared_ptr<GPUStaticMeshAsset> cacheMeshGPU;
			std::map<MaterialUUID, MaterialCache> cachePerObjectMaterials;

			// SOA for insert and multi thread.
			std::vector<PerObjectInfo> cachePerObjectData;
			std::vector<VkAccelerationStructureInstanceKHR> cachePerObjectAs;
			std::vector<MaterialUUID> cacheMaterialId;

			void clear()
			{
				cachePerObjectData.clear();
				cacheMaterialId.clear();
				cachePerObjectAs.clear();

				assetWeakPtr = {};
				cacheMeshGPU = nullptr;
				cachePerObjectMaterials.clear();

				bNewlyCreated = true;
			}

			bool empty()
			{
				return 
					cachePerObjectData.empty() &&
					cachePerObjectAs.empty() &&
					cacheMaterialId.empty();
			}

			void resize(size_t i)
			{
				cachePerObjectData.resize(i);
				cacheMaterialId.resize(i);

				cachePerObjectAs.resize(i);
			}

		} m_meshCache;

	protected:
		UUID m_assetUUID = {};
	};
}