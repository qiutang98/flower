#pragma once
#include "../Component.h"
#include "../../Renderer/MeshMisc.h"
#include "../../Renderer/Parameters.h"
#include "../../AssetSystem/MaterialManager.h"

namespace Flower
{
	class SceneNode;
	class GPUMeshAsset;
	class StaticMeshAssetHeader;
	class StandardPBRMaterialHeader;

	class StaticMeshComponent : public Component
	{
		friend class cereal::access;

	public:
		using MaterialHeaderUUID = UUID;

		// SOA for insert and multi thread.
		struct PerObjectCache
		{
			// Cache perobject data.
			std::vector<GPUPerObjectData> m_cachePerObjectData;

			// Cache perobject data id.
			std::vector<MaterialHeaderUUID> m_cachePerObjectDataId;

			void clear()
			{
				m_cachePerObjectData.clear();
				m_cachePerObjectDataId.clear();
			}
			void resize(size_t i)
			{
				m_cachePerObjectData.resize(i);
				m_cachePerObjectDataId.resize(i);
			}
		};

		struct MaterialCache
		{
			StandardPBRTexturesHandle handle;
			std::shared_ptr<StandardPBRMaterialHeader> header;
		};

	protected:
		// Mesh already replace?
		bool m_bMeshReplace = true;

		// Mesh load ready?
		bool m_bMeshReady = false;

		// Cache gpu mesh asset.
		std::shared_ptr<GPUMeshAsset> m_cacheGPUMeshAsset;

		// header is optional, some asset no exist header, we store mesh info in gpu mesh asset directly.
		std::shared_ptr<StaticMeshAssetHeader> m_cacheStaticAssetHeader = nullptr;

		// Cache perobject info.
		PerObjectCache m_perobjectCache;

		// Cache perobject material.
		std::map<MaterialHeaderUUID, MaterialCache> m_cachePerObjectMaterials;

#pragma region SerializeField
	////////////////////////////// Serialize area //////////////////////////////
	protected:
		template<class Archive>
		void serialize(Archive& archive, std::uint32_t const version);

		// Asset uuid.
		UUID m_staticMeshUUID = {};

	////////////////////////////// Serialize area //////////////////////////////
#pragma endregion SerializeField

	public:
		StaticMeshComponent();
		virtual ~StaticMeshComponent();

		StaticMeshComponent(std::shared_ptr<SceneNode> sceneNode)
			: Component(sceneNode)
		{

		}

		void setMeshUUID(const Flower::UUID& in);

		const UUID& getUUID() const
		{
			return m_staticMeshUUID;
		}

		bool isMeshAlreadySet() const
		{
			return !m_staticMeshUUID.empty();
		}

		uint32_t getVerticesCount() const;
		uint32_t getIndicesCount() const;
		uint32_t getSubmeshCount() const;

		const std::string& getMeshAssetName() const;

		virtual void tick(const RuntimeModuleTickData& tickData) override;

		void renderObjectCollect(std::vector<GPUPerObjectData>& collector);

		bool canReplaceMesh() const;

	private:
		bool setUUID(const Flower::UUID& in);

		void updateObjectCollectInfo(const RuntimeModuleTickData* tickData);

		void loadAssetByUUID();

		void clearCache();

		void asyncLoadStateHandle();

		void updateMaterials();
	};

}
