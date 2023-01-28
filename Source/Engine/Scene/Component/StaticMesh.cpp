#include "Pch.h"
#include "StaticMesh.h"
#include "Scene/SceneNode.h"
#include "Scene/Scene.h"
#include "../../AssetSystem/MeshManager.h"
#include "../../AssetSystem/MaterialManager.h"
#include <execution>

namespace Flower
{
	constexpr size_t kMinSubMeshNumStartParallel = 100;

	StaticMeshComponent::StaticMeshComponent()
	{

	}

	StaticMeshComponent::~StaticMeshComponent()
	{
		// When static mesh destroy, toggle once asset system gpu lru cache clear.
		GEngine->getRuntimeModule<AssetSystem>()->markCallGPULRUCacheShrink();
	}

	void StaticMeshComponent::tick(const RuntimeModuleTickData& tickData)
	{
		if (m_cacheGPUMeshAsset == nullptr)
		{
			auto tempStore = m_staticMeshUUID;
			m_staticMeshUUID = {};
			setUUID(tempStore);
		}
		else
		{
			updateObjectCollectInfo(&tickData);
		}

	}

	void StaticMeshComponent::asyncLoadStateHandle()
	{
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

			// Now static mesh replace or load ready just now, we try to build it's BLAS if unbuild.
			if (RHI::bSupportRayTrace)
			{
				asset->getOrBuilddBLAS();
			}

			m_perobjectCache.clear();
			m_cachePerObjectMaterials.clear();

			// Collect object.
			if (m_cacheStaticAssetHeader)
			{
				const auto& submeshes = m_cacheStaticAssetHeader->getSubMeshes();

				m_perobjectCache.resize(submeshes.size());

				for (size_t i = 0; i < submeshes.size(); i++)
				{
					const auto& submesh = submeshes[i];
					auto& cacheObject = m_perobjectCache.m_cachePerObjectData.at(i);
					auto& cacheObjectId = m_perobjectCache.m_cachePerObjectDataId.at(i);

					cacheObject.verticesArrayId = asset->getVerticesBindlessIndex();
					cacheObject.indicesArrayId = asset->getIndicesBindlessIndex();
					cacheObject.indexStartPosition = submesh.indexStartPosition;
					cacheObject.indexCount = submesh.indexCount;
					cacheObject.sphereBounds = glm::vec4(submesh.renderBounds.origin, submesh.renderBounds.radius);
					cacheObject.extents = glm::vec4(submesh.renderBounds.extents, 1.0f);

					auto material = std::dynamic_pointer_cast<StandardPBRMaterialHeader>(AssetRegistryManager::get()->getHeaderMap().at(submesh.material));
					cacheObjectId = material->getHeaderUUID(); // Store Material header UUID.

					// Insert material if no exist cache.
					auto& cacheMaterialPair = m_cachePerObjectMaterials[cacheObjectId];

					cacheMaterialPair.handle = material->buildCache();
					cacheMaterialPair.header = material;

					// init material.
					cacheObject.material = material->getGPUOnly();
				}
			}
			else
			{
				GPUPerObjectData object{};
				object.verticesArrayId = asset->getVerticesBindlessIndex();
				object.indicesArrayId = asset->getIndicesBindlessIndex();
				object.material = GPUStaticMeshStandardPBRMaterial::buildDeafult();

				// Default mesh, use fallback.
				static const glm::vec4 kBuildInSphereBounds = { 0.0f, 0.0f, 0.0f, 2.0f };
				static const glm::vec4 kBuildInExtent       = glm::vec4{ 1.0f, 1.0f, 1.0f, 0.0f };
				object.indexStartPosition = 0;
				object.indexCount         = asset->getIndicesCount();
				object.sphereBounds       = kBuildInSphereBounds;
				object.extents            = kBuildInExtent;

				m_perobjectCache.resize(1);
				m_perobjectCache.m_cachePerObjectData[0] = std::move(object);
			}
		}

		// We handle once mesh replace event.
		m_bMeshReplace = false;
	}

	void StaticMeshComponent::updateMaterials()
	{
		if (!m_cacheStaticAssetHeader)
		{
			return;
		}

		for (auto& materialPair : m_cachePerObjectMaterials)
		{
			auto& material = materialPair.second.header;
			material->getAndTryBuildGPU();
		}

		if (m_perobjectCache.m_cachePerObjectData.size() > kMinSubMeshNumStartParallel)
		{
			const auto loop = [this](const size_t loopStart, const size_t loopEnd)
			{
				for (size_t i = loopStart; i < loopEnd; ++i)
				{
					auto& object = m_perobjectCache.m_cachePerObjectData[i];
					const auto& id = m_perobjectCache.m_cachePerObjectDataId[i];
					const auto& material = m_cachePerObjectMaterials.at(id).header;

					object.material = material->getGPUOnly();
				}
			};
			GThreadPool::get()->parallelizeLoop(0, m_perobjectCache.m_cachePerObjectData.size(), loop).wait();
		}
		else
		{
			for (size_t i = 0; i < m_perobjectCache.m_cachePerObjectData.size(); i++)
			{
				auto& object = m_perobjectCache.m_cachePerObjectData[i];
				const auto& id = m_perobjectCache.m_cachePerObjectDataId[i];
				const auto& material = m_cachePerObjectMaterials.at(id).header;

				object.material = material->getGPUOnly();
			}
		}
	}

	void StaticMeshComponent::updateObjectCollectInfo(const RuntimeModuleTickData* tickData)
	{
		if (m_staticMeshUUID.empty())
		{
			// No set mesh, return.
			return;
		}

		// If asset no load yet, add load task.
		if (m_cacheGPUMeshAsset == nullptr)
		{
			loadAssetByUUID();
		}

		if (tickData && (tickData->tickCount % 11 == 0)) // Try update after some frame.
		{
			// Update load state change cases.
			asyncLoadStateHandle();

			updateMaterials();
		}
	}

	void StaticMeshComponent::loadAssetByUUID()
	{
		m_cacheGPUMeshAsset = MeshManager::get()->getOrCreateLRUMesh(m_staticMeshUUID);
		m_bMeshReplace = true;
		m_bMeshReady = m_cacheGPUMeshAsset->isAssetReady();
	}

	void StaticMeshComponent::clearCache()
	{
		m_bMeshReplace = true;
		m_bMeshReady = false;
		m_cacheGPUMeshAsset = nullptr;
		m_cacheStaticAssetHeader = nullptr;
		m_perobjectCache.clear();
		m_cachePerObjectMaterials.clear();
	}

	void StaticMeshComponent::renderObjectCollect(std::vector<GPUPerObjectData>& collector)
	{
		glm::mat4 modelMatrix = getNode()->getTransform()->getWorldMatrix();
		glm::mat4 modelMatrixPrev = getNode()->getTransform()->getPrevWorldMatrix();

		auto updateObject = [&](GPUPerObjectData& object)
		{
			object.modelMatrix     = modelMatrix;
			object.modelMatrixPrev = modelMatrixPrev;
			object.bObjectMove     = (modelMatrix == modelMatrixPrev) ? 0 : 1;
		};

		if (m_perobjectCache.m_cachePerObjectData.size() > kMinSubMeshNumStartParallel)
		{
			const auto loop = [&updateObject, this](const size_t loopStart, const size_t loopEnd)
			{
				for (size_t i = loopStart; i < loopEnd; ++i)
				{
					updateObject(m_perobjectCache.m_cachePerObjectData[i]);
				}
			};
			GThreadPool::get()->parallelizeLoop(0, m_perobjectCache.m_cachePerObjectData.size(), loop).wait();
		}
		else
		{
			for (auto& object : m_perobjectCache.m_cachePerObjectData)
			{
				updateObject(object);
			}
		}

		collector.insert(collector.end(), 
			m_perobjectCache.m_cachePerObjectData.begin(), 
			m_perobjectCache.m_cachePerObjectData.end());
	}

	bool StaticMeshComponent::canReplaceMesh() const
	{
		return !GpuUploader::get()->busy();
	}

	bool StaticMeshComponent::setUUID(const Flower::UUID& in)
	{
		// When load ready and uuid change, enable replace.
		if (in != m_staticMeshUUID && canReplaceMesh()) 
		{
			// Mesh id set, require from Meshmanager.
			m_staticMeshUUID = in;

			clearCache();
			
			// Asset header is optional.
			m_cacheStaticAssetHeader = AssetRegistryManager::get()->getHeaderMap().contains(m_staticMeshUUID) ?
				std::dynamic_pointer_cast<StaticMeshAssetHeader>(AssetRegistryManager::get()->getHeaderMap().at(m_staticMeshUUID)) :
				nullptr;

			// When static mesh replace, toggle once asset system gpu lru cache clear.
			GEngine->getRuntimeModule<AssetSystem>()->markCallGPULRUCacheShrink();
			

			// Get gpu asset.
			updateObjectCollectInfo(nullptr);

			// Mesh replace.
			return true;
		}

		// No replace mesh, just return false.
		return false;
	}

	void StaticMeshComponent::setMeshUUID(const Flower::UUID& in)
	{
		if (setUUID(in))
		{
			m_node.lock()->getScene()->setDirty(true);
		}
	}


	uint32_t StaticMeshComponent::getVerticesCount() const
	{
		return uint32_t(m_cacheGPUMeshAsset->getVerticesCount());
	}

	uint32_t StaticMeshComponent::getIndicesCount() const
	{
		return uint32_t(m_cacheGPUMeshAsset->getIndicesCount());
	}

	uint32_t StaticMeshComponent::getSubmeshCount() const
	{
		return uint32_t(m_perobjectCache.m_cachePerObjectData.size());
	}

	const std::string& StaticMeshComponent::getMeshAssetName() const
	{
		static const std::string engineName = "EngineMesh";
		if (m_cacheStaticAssetHeader)
		{
			return m_cacheStaticAssetHeader->getName();
		}
		else
		{
			return engineName;
		}
	}
}

