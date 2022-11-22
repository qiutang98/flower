#pragma once
#include "../Component.h"
#include "../../Renderer/MeshMisc.h"
#include "../../Renderer/Parameters.h"

namespace Flower
{
	class SceneNode;
	class StaticMeshComponent;
	class GPUMeshAsset;
	class StaticMeshGPUProxy;
	class StaticMeshAssetHeader;

	class StaticMeshGPUProxy
	{
		friend class cereal::access;

		friend StaticMeshComponent;

	public:
		StaticMeshGPUProxy(StaticMeshComponent* in)
			: m_staticMeshComp(in)
		{

		}

	private:
		StaticMeshComponent* m_staticMeshComp;

		// Asset uuid.
		UUID m_staticMeshUUID = {};

		bool m_bMeshReplace = true;
		bool m_bMeshReady = false;
		std::shared_ptr<GPUMeshAsset> m_cacheGPUMeshAsset;

		// header is optional, some asset no exist header, we store mesh info in gpu mesh asset directly.

		std::shared_ptr<StaticMeshAssetHeader> m_cacheStaticAssetHeader = nullptr;
		std::vector<GPUPerObjectData> m_cachePerObjectData;

		std::vector<std::shared_ptr<CPUStaticMeshStandardPBRMaterial>> m_cachePerObjectMaterials;

	public:
		void updateObjectCollectInfo();
		void renderObjectCollect(std::vector<GPUPerObjectData>& collector);
		bool setUUID(const Flower::UUID& in);

		bool canReplaceMesh() const;
	};

	class StaticMeshComponent : public Component
	{
	private:
		std::unique_ptr<StaticMeshGPUProxy> m_gpuProxy = nullptr;

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
			return m_gpuProxy->m_staticMeshUUID;
		}

		bool isMeshAlreadySet() const
		{
			return !m_gpuProxy->m_staticMeshUUID.empty();
		}

		uint32_t getVerticesCount() const;
		uint32_t getIndicesCount() const;
		uint32_t getSubmeshCount() const;

		const std::string& getMeshAssetName() const;

	public:
		virtual void tick(const RuntimeModuleTickData& tickData) override;

		void renderObjectCollect(std::vector<GPUPerObjectData>& collector);
	};

}