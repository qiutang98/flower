#include "Pch.h"
#include "MeshManager.h"
#include "TextureManager.h"
#include "MaterialManager.h"
#include "AssetSystem.h"
#include "AssetRegistry.h"

#include <nlohmann/json.hpp>
#include <stb/stb_image_write.h>
#include <stb/stb_image.h>


#pragma warning(disable: 4006)

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/GltfMaterial.h>

namespace Flower
{
	const UUID EngineMeshes::GBoxUUID = "12a68c4e-8352-4d97-a914-a0f4f4d1fd28";

	std::weak_ptr<GPUMeshAsset> EngineMeshes::GBoxPtrRef = {};
	struct AssimpModelProcess
	{
	public:
		std::filesystem::path folderPath;

		std::vector<StaticMeshSubMesh> m_subMeshInfos{};

		std::vector<StaticMeshVertex> m_vertices{};
		std::vector<VertexIndexType> m_indices{};

		// .................tex path...tex uuid........
		std::unordered_map<std::string, UUID> m_texPathUUIDMap{ };

		explicit AssimpModelProcess(const std::filesystem::path& in)
			: folderPath(in)
		{

		}

		StaticMeshSubMesh processMesh(aiMesh* mesh, const aiScene* scene, std::shared_ptr<RegistryEntry> materialFolderEntry, std::shared_ptr<RegistryEntry> texFolderEntry)
		{
			auto* assetSystem = GEngine->getRuntimeModule<AssetSystem>();

			StaticMeshSubMesh subMeshInfo{};
			subMeshInfo.indexStartPosition = (uint32_t)m_indices.size();
			uint32_t indexOffset = (uint32_t)m_vertices.size();

			std::vector<StaticMeshVertex> vertices{};
			std::vector<VertexIndexType> indices{};

			for (unsigned int i = 0; i < mesh->mNumVertices; i++)
			{
				StaticMeshVertex vertex;

				glm::vec3 vector{};
				vector.x = mesh->mVertices[i].x;
				vector.y = mesh->mVertices[i].y;
				vector.z = mesh->mVertices[i].z;
				vertex.position = vector;

				vector.x = mesh->mNormals[i].x;
				vector.y = mesh->mNormals[i].y;
				vector.z = mesh->mNormals[i].z;
				vertex.normal = vector;

				if (mesh->mTextureCoords[0])
				{
					glm::vec2 vec{};
					vec.x = mesh->mTextureCoords[0][i].x;
					vec.y = mesh->mTextureCoords[0][i].y;
					vertex.uv0 = vec;
				}
				else
				{
					vertex.uv0 = glm::vec2(0.0f, 0.0f);
				}

				glm::vec4 tangentVec{};
				tangentVec.x = mesh->mTangents[i].x;
				tangentVec.y = mesh->mTangents[i].y;
				tangentVec.z = mesh->mTangents[i].z;

				// Tangent vector.
				vector.x = mesh->mTangents[i].x;
				vector.y = mesh->mTangents[i].y;
				vector.z = mesh->mTangents[i].z;

				glm::vec3 bitangent{};
				bitangent.x = mesh->mBitangents[i].x;
				bitangent.y = mesh->mBitangents[i].y;
				bitangent.z = mesh->mBitangents[i].z;

				// Tangent sign process.
				tangentVec.w = glm::sign(glm::dot(glm::normalize(bitangent), glm::normalize(glm::cross(vertex.normal, vector))));
				vertex.tangent = tangentVec;
				vertices.push_back(vertex);
			}

			for (unsigned int i = 0; i < mesh->mNumFaces; i++)
			{
				aiFace face = mesh->mFaces[i];
				for (unsigned int j = 0; j < face.mNumIndices; j++)
				{
					indices.push_back(indexOffset + face.mIndices[j]);
				}
			}

			m_vertices.insert(m_vertices.end(), vertices.begin(), vertices.end());
			m_indices.insert(m_indices.end(), indices.begin(), indices.end());

			subMeshInfo.indexCount = (uint32_t)indices.size();

			// aabb bounds process.
			auto aabbMax = mesh->mAABB.mMax;
			auto aabbMin = mesh->mAABB.mMin;
			auto aabbExt = (aabbMax - aabbMin) * 0.5f;
			auto aabbCenter = aabbExt + aabbMin;
			subMeshInfo.renderBounds.extents[0] = aabbExt.x;
			subMeshInfo.renderBounds.extents[1] = aabbExt.y;
			subMeshInfo.renderBounds.extents[2] = aabbExt.z;
			subMeshInfo.renderBounds.origin[0] = aabbCenter.x;
			subMeshInfo.renderBounds.origin[1] = aabbCenter.y;
			subMeshInfo.renderBounds.origin[2] = aabbCenter.z;
			subMeshInfo.renderBounds.radius = glm::distance(
				glm::vec3(aabbMax.x, aabbMax.y, aabbMax.z),
				glm::vec3(aabbCenter.x, aabbCenter.y, aabbCenter.z)
			);

			// standard pbr texture prepare.
			aiString baseColorTextures{};
			aiString normalTextures{};
			aiString specularTextures{};
			aiString aoTextures{};
			aiString emissiveTextures{};

			auto tryFetechTexture = [&](const char* pathIn, std::string& OutId, bool bSrgb, float cutoff)
			{
				const auto path = (folderPath / pathIn).string();
				if (m_texPathUUIDMap.contains(path))
				{
					OutId = m_texPathUUIDMap[path];
				}
				else
				{
					OutId = assetSystem->importAsset(path, EAssetType::Texture, texFolderEntry, ImportOptions
						{
							.texOptions = ImportTextureOptions
							{
								.bSrgb = bSrgb,
								.bBuildMipmap = true,
								.cutoff = cutoff,
							}
						});
					m_texPathUUIDMap[path] = OutId;
				}
			};

			if (mesh->mMaterialIndex >= 0)
			{
				aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
				static const std::string materialName = "_mat";

				// Create new material.
				auto newMaterial = std::make_shared<StandardPBRMaterialHeader>((material->GetName().C_Str() + materialName).c_str());

				{
					// register in map.
					AssetRegistryManager::get()->registerAssetMap(newMaterial, EAssetType::Material);

					// add new entry file for materials folder entry.
					std::shared_ptr<RegistryEntry> newRegistry = std::make_shared<RegistryEntry>(newMaterial->getHeaderUUID(), newMaterial->getName());
					AssetRegistryManager::get()->addChild(materialFolderEntry, newRegistry, true);
				}

				if (material->GetTextureCount(aiTextureType_DIFFUSE) > 0)
				{
					material->GetTexture(aiTextureType_DIFFUSE, 0, &baseColorTextures);
					tryFetechTexture(baseColorTextures.C_Str(), newMaterial->baseColorTexture, true, 0.5f); // SRGB
				}

				if (material->GetTextureCount(aiTextureType_HEIGHT) > 0)
				{
					material->GetTexture(aiTextureType_HEIGHT, 0, &normalTextures);
					tryFetechTexture(normalTextures.C_Str(), newMaterial->normalTexture, false, 1.0f); // LINEAR
				}

				if (material->GetTextureCount(aiTextureType_SPECULAR) > 0)
				{
					material->GetTexture(aiTextureType_SPECULAR, 0, &specularTextures);
					tryFetechTexture(specularTextures.C_Str(), newMaterial->specularTexture, false, 1.0f); // Linear
				}

				// AO
				if (material->GetTextureCount(aiTextureType_AMBIENT) > 0)
				{
					material->GetTexture(aiTextureType_AMBIENT, 0, &aoTextures);
					tryFetechTexture(aoTextures.C_Str(), newMaterial->aoTexture, false, 1.0f); // Linear
				}

				if (material->GetTextureCount(aiTextureType_EMISSIVE) > 0)
				{
					material->GetTexture(aiTextureType_EMISSIVE, 0, &emissiveTextures);
					tryFetechTexture(emissiveTextures.C_Str(), newMaterial->emissiveTexture, true, 1.0f); // SRGB
				}

				subMeshInfo.material = newMaterial->getHeaderUUID();

				AssetRegistryManager::get()->markDirty();
			}
			else // no material found, keep empty.
			{
				subMeshInfo.material = {};
			}

			return subMeshInfo;
		}

		void processNode(aiNode* node, const aiScene* scene, std::shared_ptr<RegistryEntry> materialFolderEntry, std::shared_ptr<RegistryEntry> texFolderEntry)
		{
			for (unsigned int i = 0; i < node->mNumMeshes; i++)
			{
				aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
				m_subMeshInfos.push_back(processMesh(mesh, scene, materialFolderEntry, texFolderEntry));
			}

			for (unsigned int i = 0; i < node->mNumChildren; i++)
			{
				processNode(node->mChildren[i], scene, materialFolderEntry, texFolderEntry);
			}
		}
	};

	bool StaticMeshAssetHeader::initFromRawStaticMesh(const std::filesystem::path& rawPath, std::shared_ptr<RegistryEntry> parentEntry)
	{
		Assimp::Importer importer;
		const aiScene* scene = importer.ReadFile(rawPath.string(),
			aiProcessPreset_TargetRealtime_Fast | aiProcess_FlipUVs | aiProcess_GenBoundingBoxes);

		if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
		{
			LOG_ERROR("ERROR::ASSIMP::{0}", importer.GetErrorString());
			return false;
		}

		setCacheBinData(std::make_shared<StaticMeshAssetBin>(rawPath.filename().string()));
		auto processingMeshBin = getBinData<StaticMeshAssetBin>();

		if (parentEntry == nullptr)
		{
			parentEntry = AssetRegistryManager::get()->getRoot();
		}

		std::shared_ptr<RegistryEntry> meshFolderRegistry = std::make_shared<RegistryEntry>("", rawPath.stem().string() + "_Misc");
		AssetRegistryManager::get()->addChild(parentEntry, meshFolderRegistry, true);

		std::shared_ptr<RegistryEntry> texFolderRegistry = std::make_shared<RegistryEntry>("", "Texture");
		AssetRegistryManager::get()->addChild(meshFolderRegistry, texFolderRegistry, true);

		std::shared_ptr<RegistryEntry> materialFolderRegistry = std::make_shared<RegistryEntry>("", "Material");
		AssetRegistryManager::get()->addChild(meshFolderRegistry, materialFolderRegistry, true);

		AssimpModelProcess processor(rawPath.parent_path());
		processor.processNode(scene->mRootNode, scene, materialFolderRegistry, texFolderRegistry);

		m_subMeshes = processor.m_subMeshInfos;
		processingMeshBin->m_vertices = processor.m_vertices;
		processingMeshBin->m_indices = processor.m_indices;

		m_indicesCount = processor.m_indices.size();
		m_verticesCount = processor.m_vertices.size();

		return true;
	}

	static std::string getRuntimeUniqueMeshAssetName(const std::string& in)
	{
		static size_t GRuntimeId = 0;
		GRuntimeId++;
		return "Flower_MeshAssetId:" + std::to_string(GRuntimeId) + in;
	}

	uint32_t indexTypeToSize(VkIndexType type)
	{
		switch (type)
		{
		case VK_INDEX_TYPE_UINT16:    return sizeof(uint16_t);
		case VK_INDEX_TYPE_UINT32:    return sizeof(uint32_t);
		case VK_INDEX_TYPE_UINT8_EXT: return sizeof(uint8_t);
		default:                      CHECK_ENTRY();
		}

		return 0;
	}

	GPUMeshAsset::GPUMeshAsset(
		bool bPersistent,
		GPUMeshAsset* fallback,
		const std::string& name,
		VkDeviceSize vertexSize,
		size_t singleVertexSize,
		VkDeviceSize indexSize,
		VkIndexType indexType)
		: LRUAssetInterface(fallback, bPersistent)
		, m_name(name)
	{
		CHECK(m_vertexBuffer == nullptr && "You must ensure mesh asset only init once.");
		CHECK(m_indexBuffer == nullptr && "You must ensure mesh asset only init once.");

		// Mesh info also support Ray trace info.
		auto bufferFlagBasic = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		VmaAllocationCreateFlags bufferFlagVMA = {};
		if (RHI::bSupportRayTrace)
		{
			bufferFlagBasic |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
			bufferFlagVMA = {};
		}

		m_vertexBuffer = VulkanBuffer::create2(
			getRuntimeUniqueMeshAssetName(name).c_str(),
			bufferFlagBasic | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			bufferFlagVMA,
			vertexSize
		);

		m_indexBuffer = VulkanBuffer::create2(
			getRuntimeUniqueMeshAssetName(name).c_str(),
			bufferFlagBasic | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			bufferFlagVMA,
			indexSize
		);


		if (RHI::bSupportRayTrace)
		{
			m_bottomLevelAccelerateStructure = std::make_unique<AccelerateStructure>(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR);
		}

		m_indexType = indexType;
		m_singleIndexSize = sizeof(uint32_t);
		m_indexCount = uint32_t(indexSize) / indexTypeToSize(indexType);
		m_indexCountUint32Count = uint32_t(indexSize) / sizeof(uint32_t);

		m_singleVertexSize = uint32_t(singleVertexSize);
		m_vertexCount = uint32_t(vertexSize) / m_singleVertexSize;
		m_vertexFloat32Count = uint32_t(vertexSize) / sizeof(float);
	}

	GPUMeshAsset::GPUMeshAsset(bool bPersistent, GPUMeshAsset* fallback, const std::string& name)
		: LRUAssetInterface(fallback, bPersistent)
		, m_name(name)
	{

	}

	GPUMeshAsset::~GPUMeshAsset()
	{
		if (!m_bPersistent)
		{
			if (m_vertexBufferBindlessIndex != ~0)
			{
				MeshManager::get()->getBindlessVertexBuffers()->freeBindlessImpl(
					m_vertexBufferBindlessIndex,
					EngineMeshes::GBoxPtrRef.lock() ? EngineMeshes::GBoxPtrRef.lock()->m_vertexBuffer : nullptr);
			}
			if (m_indexBufferBindlessIndex != ~0)
			{
				MeshManager::get()->getBindlessIndexBuffers()->freeBindlessImpl(
					m_indexBufferBindlessIndex,
					EngineMeshes::GBoxPtrRef.lock() ? EngineMeshes::GBoxPtrRef.lock()->m_indexBuffer : nullptr);
			}
		}

		m_indexBuffer.reset();
		m_vertexBuffer.reset();
	}

	void GPUMeshAsset::prepareToUpload()
	{
		CHECK(m_vertexBufferBindlessIndex == ~0);
		CHECK(m_indexBufferBindlessIndex == ~0);
	}

	void GPUMeshAsset::finishUpload()
	{
		m_vertexBufferBindlessIndex =
			MeshManager::get()->getBindlessVertexBuffers()->updateBufferToBindlessDescriptorSet(
				m_vertexBuffer->getVkBuffer(), 0, m_vertexBuffer->getSize());

		m_indexBufferBindlessIndex =
			MeshManager::get()->getBindlessIndexBuffers()->updateBufferToBindlessDescriptorSet(
				m_indexBuffer->getVkBuffer(), 0, m_indexBuffer->getSize());

		CHECK(m_vertexBufferBindlessIndex != ~0);
		CHECK(m_indexBufferBindlessIndex != ~0);
	}

	void MeshContext::init()
	{
		m_lruCache = std::make_unique<LRUAssetCache<GPUMeshAsset>>(512, 256);

		m_vertexBindlessBuffer = std::make_unique<BindlessStorageBuffer>();
		m_indexBindlessBuffer = std::make_unique<BindlessStorageBuffer>();

		m_vertexBindlessBuffer->init();
		m_indexBindlessBuffer->init();
	}

	void MeshContext::release()
	{
		m_lruCache.reset();

		m_vertexBindlessBuffer->release();
		m_indexBindlessBuffer->release();
	}

	void StaticMeshRawDataLoadTask::finishCallback()
	{
		meshAssetGPU->setAsyncLoadState(false);
	}

	void StaticMeshRawDataLoadTask::uploadFunction(
		uint32_t stageBufferOffset,
		RHICommandBufferBase& commandBuffer,
		VulkanBuffer& stageBuffer)
	{
		CHECK(uploadSize() == uint32_t(cacheIndexData.size() + cacheVertexData.size()));
		uint32_t indexOffsetInSrcBuffer = stageBufferOffset;
		uint32_t vertexOffsetInSrcBuffer = indexOffsetInSrcBuffer + uint32_t(cacheIndexData.size());

		stageBuffer.map();
		memcpy((void*)((char*)stageBuffer.mapped + indexOffsetInSrcBuffer), cacheIndexData.data(), cacheIndexData.size());
		memcpy((void*)((char*)stageBuffer.mapped + vertexOffsetInSrcBuffer), cacheVertexData.data(), cacheVertexData.size());
		stageBuffer.unmap();

		meshAssetGPU->prepareToUpload();

		{
			VkBufferCopy regionIndex{};
			regionIndex.size = VkDeviceSize(cacheIndexData.size());
			regionIndex.srcOffset = indexOffsetInSrcBuffer;
			regionIndex.dstOffset = 0;
			vkCmdCopyBuffer(
				commandBuffer.cmd,
				stageBuffer,
				meshAssetGPU->getIndexBuffer().getVkBuffer(),
				1,
				&regionIndex);
		}

		{
			VkBufferCopy regionVertex{};
			regionVertex.size = VkDeviceSize(cacheVertexData.size());
			regionVertex.srcOffset = vertexOffsetInSrcBuffer;
			regionVertex.dstOffset = 0;
			vkCmdCopyBuffer(
				commandBuffer.cmd,
				stageBuffer,
				meshAssetGPU->getVertexBuffer().getVkBuffer(),
				1,
				&regionVertex);
		}

		// Also build low level acceleration for static mesh.
		if (RHI::bSupportRayTrace)
		{
			
		}

		meshAssetGPU->finishUpload();
	}

	std::shared_ptr<StaticMeshRawDataLoadTask> StaticMeshRawDataLoadTask::buildFromData(
		const std::string& name,
		const UUID& uuid,
		bool bPersistent,
		uint8_t* indices,
		size_t indexSize,
		VkIndexType indexType,
		uint8_t* vertices,
		size_t vertexSize,
		size_t singleVertexSize)
	{
		if (bPersistent)
		{
			CHECK(!MeshManager::get()->isAssetExist(uuid) && "Persistent asset has exist, don't register repeatly.");
		}

		auto newTask = std::make_shared<StaticMeshRawDataLoadTask>();
		newTask->cacheVertexData.resize(vertexSize);
		newTask->cacheIndexData.resize(indexSize);

		memcpy((void*)(newTask->cacheVertexData.data()), (void*)vertices, vertexSize);
		memcpy((void*)(newTask->cacheIndexData.data()), (void*)indices, indexSize);

		GPUMeshAsset* fallback = nullptr;
		if (!bPersistent)
		{
			fallback = MeshManager::get()->getMesh(EngineMeshes::GBoxUUID).get();
			CHECK(fallback && "Non persistent asset must exist one fallback mesh.");
		}

		auto newAsset = std::shared_ptr<GPUMeshAsset>(new GPUMeshAsset(
			bPersistent,
			fallback,
			name,
			vertexSize,
			singleVertexSize,
			indexSize,
			indexType));
		MeshManager::get()->insertGPUAsset(uuid, newAsset);

		newTask->meshAssetGPU = newAsset;

		return newTask;
	}


	void StaticMeshLoadTask::finishCallback()
	{
		meshAssetGPU->setAsyncLoadState(false);
	}

	void StaticMeshLoadTask::uploadFunction(
		uint32_t stageBufferOffset,
		RHICommandBufferBase& commandBuffer,
		VulkanBuffer& stageBuffer)
	{
		// Load bin data.
		auto meshBin = std::dynamic_pointer_cast<StaticMeshAssetBin>(cacheHeader->loadBinData());
		CHECK(meshBin != nullptr);

		const auto verticesSize = meshBin->getVertices().size() * sizeof(meshBin->getVertices()[0]);
		const auto indicesSize = meshBin->getIndices().size() * sizeof(meshBin->getIndices()[0]);

		CHECK(uploadSize() == uint32_t(indicesSize + verticesSize));
		uint32_t indexOffsetInSrcBuffer = stageBufferOffset;
		uint32_t vertexOffsetInSrcBuffer = indexOffsetInSrcBuffer + uint32_t(indicesSize);

		stageBuffer.map();
		memcpy((void*)((char*)stageBuffer.mapped + indexOffsetInSrcBuffer), meshBin->getIndices().data(), indicesSize);
		memcpy((void*)((char*)stageBuffer.mapped + vertexOffsetInSrcBuffer), meshBin->getVertices().data(), verticesSize);
		stageBuffer.unmap();

		meshAssetGPU->prepareToUpload();

		{
			VkBufferCopy regionIndex{};
			regionIndex.size = VkDeviceSize(indicesSize);
			regionIndex.srcOffset = indexOffsetInSrcBuffer;
			regionIndex.dstOffset = 0;
			vkCmdCopyBuffer(
				commandBuffer.cmd,
				stageBuffer,
				meshAssetGPU->getIndexBuffer().getVkBuffer(),
				1,
				&regionIndex);
		}

		{
			VkBufferCopy regionVertex{};
			regionVertex.size = VkDeviceSize(verticesSize);
			regionVertex.srcOffset = vertexOffsetInSrcBuffer;
			regionVertex.dstOffset = 0;
			vkCmdCopyBuffer(
				commandBuffer.cmd,
				stageBuffer,
				meshAssetGPU->getVertexBuffer().getVkBuffer(),
				1,
				&regionVertex);
		}

		meshAssetGPU->finishUpload();
	}

	std::shared_ptr<StaticMeshLoadTask> StaticMeshLoadTask::build(
		std::shared_ptr<RegistryEntry> registry,
		bool bPersistent)
	{
		CHECK(registry->isLeaf() && registry->isValid());

		auto meshHeader = std::dynamic_pointer_cast<StaticMeshAssetHeader>(registry->getHeader());
		CHECK(meshHeader != nullptr);
		GPUMeshAsset* fallback = nullptr;
		if (bPersistent)
		{
			CHECK(!MeshManager::get()->isAssetExist(meshHeader->getHeaderUUID()) && "Persistent asset has exist, don't register repeatly.");
		}
		else
		{
			fallback = MeshManager::get()->getMesh(EngineMeshes::GBoxUUID).get();
			CHECK(fallback && "Non persistent asset must exist one fallback mesh.");
		}

		auto newTask = std::make_shared<StaticMeshLoadTask>();

		const auto verticesSize = meshHeader->getVerticesCount() * sizeof(StaticMeshVertex);
		const auto indicesSize = meshHeader->getIndicesCount() * sizeof(uint32_t);

		auto newAsset = std::shared_ptr<GPUMeshAsset>(new GPUMeshAsset(
			bPersistent,
			fallback,
			registry->getName(),
			verticesSize,
			sizeof(StaticMeshVertex),
			indicesSize,
			VK_INDEX_TYPE_UINT32));

		MeshManager::get()->insertGPUAsset(meshHeader->getHeaderUUID(), newAsset);
		newTask->meshAssetGPU = newAsset;
		newTask->cacheHeader = meshHeader;

		return newTask;
	}

	std::shared_ptr<GPUMeshAsset> MeshContext::getOrCreateLRUMesh(const AssetHeaderUUID& id)
	{
		// No exist in lru cache, need load from disk.
		if (!m_lruCache->contain(id))
		{
			const auto& entryHeaderMap = AssetRegistryManager::get()->getEntryHeaderMap();
			const auto& entryMap = AssetRegistryManager::get()->getEntryMap();
			auto newTask = StaticMeshLoadTask::build(entryMap.at(entryHeaderMap.at(id)).lock(), false);
			GEngine->getRuntimeModule<AssetSystem>()->addUploadTask(newTask);
		}

		return getMesh(id);
	}

	std::shared_ptr<GPUMeshAsset> MeshContext::getOrCreateLRUMesh(std::shared_ptr<StaticMeshAssetHeader> header)
	{
		return getOrCreateLRUMesh(header->getHeaderUUID());
	}

	void MeshContext::shrinkLRU()
	{
		// Find unused asset and push to lazy destory component.
		size_t sizeReduce = m_lruCache->prune([&](std::shared_ptr<GPUMeshAsset> removedAsset) 
		{
			GEngine->getRuntimeModule<AssetSystem>()->addUnusedAsset(removedAsset);
		});

		LOG_INFO("Mesh manager reduce {0} mesh size.", sizeReduce);
	}
}