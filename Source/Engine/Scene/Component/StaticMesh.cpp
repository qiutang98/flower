#include "Pch.h"
#include "StaticMesh.h"
#include "Scene/SceneNode.h"
#include "Scene/Scene.h"
#include "../../MeshTool/MeshToolCommon.h"
#include "../../AssetSystem/MeshManager.h"

namespace Flower
{
	void StaticMeshGPUProxy::updateObjectCollectInfo()
	{
		if (m_staticMeshUUID.empty())
		{
			// No set mesh, return.
			return;
		}

		// Pre-return if no mesh replace and no mesh loading state change.
		if (!m_bMeshReplace && m_bMeshReady)
		{
			return;
		}

		// Update mesh loading state.
		// If mesh replace, update proxys.
		// If mesh loading state change, upadte proxy.
		m_bMeshReady = m_cacheGPUMeshAsset->isAssetReady();
		if (m_bMeshReplace || m_bMeshReady)
		{
			GPUMeshAsset* asset = m_cacheGPUMeshAsset->getReadyAsset();
			m_cachePerObjectData.clear();

			GPUPerObjectData object{};
			object.verticesArrayId = asset->getVerticesBindlessIndex();
			object.indicesArrayId = asset->getIndicesBindlessIndex();
			object.material = GPUStaticMeshStandardPBRMaterial::buildDeafult();

			if (m_cacheStaticAssetHeader)
			{
				const auto& submeshes = m_cacheStaticAssetHeader->getSubMeshes();
				std::vector<std::shared_ptr<CPUStaticMeshStandardPBRMaterial>> tempCacheMaterials;
				for (const auto& submesh : submeshes)
				{
					object.indexStartPosition = submesh.indexStartPosition;
					object.indexCount = submesh.indexCount;
					object.sphereBounds = glm::vec4(submesh.renderBounds.origin, submesh.renderBounds.radius);
					object.extents = glm::vec4(submesh.renderBounds.extents, 1.0f);

					std::shared_ptr<CPUStaticMeshStandardPBRMaterial> cpuMaterials = 
						std::make_shared<CPUStaticMeshStandardPBRMaterial>();

					m_bMeshReady &= cpuMaterials->buildWithMaterialUUID(submesh.material);

					object.material = cpuMaterials->buildGPU();

					m_cachePerObjectData.push_back(object);
					tempCacheMaterials.push_back(cpuMaterials);
				}

				if (m_bMeshReady)
				{
					m_cachePerObjectMaterials = tempCacheMaterials;
				}
				else
				{
					m_cachePerObjectMaterials.insert(m_cachePerObjectMaterials.end(), tempCacheMaterials.begin(), tempCacheMaterials.end());
				}
			}
			else
			{
				object.indexStartPosition = 0;
				object.indexCount = asset->getIndicesCount();
				object.sphereBounds = BuildInSphereBounds;
				object.extents = BuildInExtent;
				m_cachePerObjectData.push_back(object);
			}
		}
		m_bMeshReplace = false;
	}

	void StaticMeshGPUProxy::renderObjectCollect(std::vector<GPUPerObjectData>& collector)
	{
		glm::mat4 modelMatrix = m_staticMeshComp->getNode()->getTransform()->getWorldMatrix();
		glm::mat4 modelMatrixPrev = m_staticMeshComp->getNode()->getTransform()->getPrevWorldMatrix();

		for (auto& object : m_cachePerObjectData)
		{
			object.modelMatrix = modelMatrix;
			object.modelMatrixPrev = modelMatrixPrev;
			object.bObjectMove = (modelMatrix == modelMatrixPrev) ? 0 : 1;
		}
		collector.insert(collector.end(), m_cachePerObjectData.begin(), m_cachePerObjectData.end());
	}

	bool StaticMeshGPUProxy::canReplaceMesh() const
	{
		return !GpuUploader::get()->busy();
	}

	bool StaticMeshGPUProxy::setUUID(const Flower::UUID& in)
	{
		// When load ready and uuid change, enable replace.
		if (in != m_staticMeshUUID && canReplaceMesh()) 
		{
			// Mesh id set, require from Meshmanager.
			m_staticMeshUUID = in;
			
			// Asset header is optional.
			m_cacheStaticAssetHeader = AssetRegistryManager::get()->getHeaderMap().contains(m_staticMeshUUID) ?
				std::dynamic_pointer_cast<StaticMeshAssetHeader>(AssetRegistryManager::get()->getHeaderMap().at(m_staticMeshUUID)) :
				nullptr;

			// When static mesh replace, toggle once asset system gpu lru cache clear.
			GEngine->getRuntimeModule<AssetSystem>()->markCallGPULRUCacheShrink();
		
	
			// Get gpu asset.
			m_cacheGPUMeshAsset = MeshManager::get()->getOrCreateLRUMesh(m_staticMeshUUID);
			m_bMeshReplace = true;
			m_bMeshReady = m_cacheGPUMeshAsset->isAssetReady();
			updateObjectCollectInfo();

			

			// Mesh replace.
			return true;
		}

		// No replace mesh, just return false.
		return false;
	}

	StaticMeshComponent::StaticMeshComponent()
	{
		m_gpuProxy = std::make_unique<StaticMeshGPUProxy>(this);
	}

	StaticMeshComponent::~StaticMeshComponent()
	{
		// When static mesh destroy, toggle once asset system gpu lru cache clear.
		GEngine->getRuntimeModule<AssetSystem>()->markCallGPULRUCacheShrink();
	}

	void StaticMeshComponent::setMeshUUID(const Flower::UUID& in)
	{
		if (m_gpuProxy->setUUID(in))
		{
			m_node.lock()->getScene()->setDirty(true);
		}
	}

	void StaticMeshComponent::tick(const RuntimeModuleTickData& tickData)
	{
		m_gpuProxy->updateObjectCollectInfo();
	}

	void StaticMeshComponent::renderObjectCollect(std::vector<GPUPerObjectData>& collector)
	{
		m_gpuProxy->renderObjectCollect(collector);
	}

	uint32_t StaticMeshComponent::getVerticesCount() const
	{
		return uint32_t(m_gpuProxy->m_cacheGPUMeshAsset->getVerticesCount());
	}

	uint32_t StaticMeshComponent::getIndicesCount() const
	{
		return uint32_t(m_gpuProxy->m_cacheGPUMeshAsset->getIndicesCount());
	}

	uint32_t StaticMeshComponent::getSubmeshCount() const
	{
		return uint32_t(m_gpuProxy->m_cachePerObjectData.size());
	}

	const std::string& StaticMeshComponent::getMeshAssetName() const
	{
		static const std::string engineName = "EngineMesh";
		if (m_gpuProxy->m_cacheStaticAssetHeader)
		{
			return m_gpuProxy->m_cacheStaticAssetHeader->getName();
		}
		else
		{
			return engineName;
		}
	}
}

