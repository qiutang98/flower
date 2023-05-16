#include "asset_staticmesh.h"
#include "asset_material.h"
#include "asset_texture.h"

#include <util/assimp_helper.h>
#include "asset_system.h"



namespace engine
{
	AssetStaticMesh::AssetStaticMesh(const std::string& assetNameUtf8, const std::string& assetRelativeRootProjectPathUtf8)
		: AssetInterface(assetNameUtf8, assetRelativeRootProjectPathUtf8)
	{

	}

	bool AssetStaticMesh::buildFromConfigs(
		const ImportConfig& config,
		const std::filesystem::path& projectRootPath,
		const std::filesystem::path& savePath,
		const std::filesystem::path& srcPath)
	{
		Assimp::Importer importer;
		const aiScene* scene = importer.ReadFile(srcPath.string(),
			aiProcessPreset_TargetRealtime_Fast | aiProcess_FlipUVs | aiProcess_GenBoundingBoxes);

		if (scene == nullptr)
		{
			LOG_ERROR("Mesh {} import fail.", utf8::utf16to8(srcPath.u16string()));
			return false;
		}

		std::string assetNameUtf8 = utf8::utf16to8(savePath.filename().u16string());

		if (std::filesystem::exists(savePath))
		{
			LOG_ERROR("Path {0} already exist, asset {1} import fail!", utf8::utf16to8(savePath.u16string()), assetNameUtf8);
			return false;
		}

		if (!std::filesystem::create_directory(savePath))
		{
			LOG_ERROR("Folder {0} create failed, asset {1} import fail!", utf8::utf16to8(savePath.u16string()), assetNameUtf8);
			return false;
		}

		const auto textureFolderPath = savePath / "textures";
		const auto materialFolderPath = savePath / "materials";

		std::filesystem::create_directory(textureFolderPath);
		std::filesystem::create_directory(materialFolderPath);

		AssimpStaticMeshImporter processor(srcPath, projectRootPath, materialFolderPath, textureFolderPath);
		processor.processNode(scene->mRootNode, scene);

		const auto meshFileSavePath = savePath / assetNameUtf8;



		// Save asset meta.
		{
			AssetStaticMesh meta(assetNameUtf8, buildRelativePathUtf8(projectRootPath, meshFileSavePath));
			meta.m_subMeshes = processor.getSubmeshInfo();
			meta.m_indicesCount = processor.getIndicesCount();
			meta.m_verticesCount = processor.getVerticesCount();

			saveAssetMeta<AssetStaticMesh>(meta, meshFileSavePath, ".staticmesh");
		}

		// Save static mesh binary file.
		{
			StaticMeshBin meshBin{};
			meshBin.indices = processor.moveIndices();
			meshBin.tangents = processor.moveTangents();
			meshBin.normals = processor.moveNormals();
			meshBin.uv0s = processor.moveUv0s();
			meshBin.positions = processor.movePositions();

			saveAsset(meshBin, meshFileSavePath, ".staticmeshbin");
		}

		return true;
	}

	void AssetStaticMeshLoadFromCacheTask::uploadFunction(
		uint32_t stageBufferOffset, 
		void* bufferPtrStart, 
		RHICommandBufferBase& commandBuffer, 
		VulkanBuffer& stageBuffer)
	{
		auto savePath = getAssetSystem()->getProjectRootPath();
		auto filePath = "\\." + cachePtr->getRelativePathUtf8() + ".staticmeshbin";
		savePath += filePath;

		StaticMeshBin meshBin{};
		loadAsset(meshBin, savePath);

		const auto tangentSize = meshBin.tangents.size() * sizeof(meshBin.tangents[0]);
		const auto normalSize = meshBin.normals.size() * sizeof(meshBin.normals[0]);
		const auto uv0Size = meshBin.uv0s.size() * sizeof(meshBin.uv0s[0]);
		const auto indicesSize  = meshBin.indices.size() * sizeof(meshBin.indices[0]);
		const auto positionsSize = meshBin.positions.size() * sizeof(meshBin.positions[0]);

		ASSERT(uploadSize() == uint32_t(indicesSize + tangentSize + normalSize + uv0Size + positionsSize), "Static mesh size un-match!");

		uint32_t indicesOffsetInSrcBuffer = 0;
		uint32_t tangentOffsetInSrcBuffer = (uint32_t)(indicesOffsetInSrcBuffer + indicesSize);
		uint32_t normalOffsetInSrcBuffer = (uint32_t)(tangentOffsetInSrcBuffer + tangentSize);
		uint32_t uv0OffsetInSrcBuffer = (uint32_t)(normalOffsetInSrcBuffer + normalSize);
		uint32_t positionsOffsetInSrcBuffer = (uint32_t)(uv0OffsetInSrcBuffer + uv0Size);

		memcpy((void*)((char*)bufferPtrStart + indicesOffsetInSrcBuffer), (const void*)meshBin.indices.data(), indicesSize);
		memcpy((void*)((char*)bufferPtrStart + tangentOffsetInSrcBuffer), (const void*)meshBin.tangents.data(), tangentSize);
		memcpy((void*)((char*)bufferPtrStart + normalOffsetInSrcBuffer), (const void*)meshBin.normals.data(), normalSize);
		memcpy((void*)((char*)bufferPtrStart + uv0OffsetInSrcBuffer), (const void*)meshBin.uv0s.data(), uv0Size);
		memcpy((void*)((char*)bufferPtrStart + positionsOffsetInSrcBuffer), (const void*)meshBin.positions.data(), positionsSize);

		{
			VkBufferCopy regionIndex{};
			regionIndex.size = indicesSize;
			regionIndex.srcOffset = stageBufferOffset + indicesOffsetInSrcBuffer;
			regionIndex.dstOffset = 0;
			vkCmdCopyBuffer(
				commandBuffer.cmd,
				stageBuffer,
				meshAssetGPU->getIndices()->getVkBuffer(),
				1,
				&regionIndex);
		}

		{
			VkBufferCopy regionVertex{};
			regionVertex.size = tangentSize;
			regionVertex.srcOffset = stageBufferOffset + tangentOffsetInSrcBuffer;
			regionVertex.dstOffset = 0;
			vkCmdCopyBuffer(
				commandBuffer.cmd,
				stageBuffer,
				meshAssetGPU->getTangents()->getVkBuffer(),
				1,
				&regionVertex);
		}
		{
			VkBufferCopy regionVertex{};
			regionVertex.size = normalSize;
			regionVertex.srcOffset = stageBufferOffset + normalOffsetInSrcBuffer;
			regionVertex.dstOffset = 0;
			vkCmdCopyBuffer(
				commandBuffer.cmd,
				stageBuffer,
				meshAssetGPU->getNormals()->getVkBuffer(),
				1,
				&regionVertex);
		}
		{
			VkBufferCopy regionVertex{};
			regionVertex.size = uv0Size;
			regionVertex.srcOffset = stageBufferOffset + uv0OffsetInSrcBuffer;
			regionVertex.dstOffset = 0;
			vkCmdCopyBuffer(
				commandBuffer.cmd,
				stageBuffer,
				meshAssetGPU->getUv0s()->getVkBuffer(),
				1,
				&regionVertex);
		}

		{
			VkBufferCopy regionVertex{};
			regionVertex.size = positionsSize;
			regionVertex.srcOffset = stageBufferOffset + positionsOffsetInSrcBuffer;
			regionVertex.dstOffset = 0;
			vkCmdCopyBuffer(
				commandBuffer.cmd,
				stageBuffer,
				meshAssetGPU->getPosition()->getVkBuffer(),
				1,
				&regionVertex);
		}
	}

	std::shared_ptr<AssetStaticMeshLoadFromCacheTask> AssetStaticMeshLoadFromCacheTask::build(VulkanContext* context, std::shared_ptr<AssetStaticMesh> meta)
	{
		auto fallback = context->getEngineStaticMeshBox();
		auto newTask = std::make_shared<AssetStaticMeshLoadFromCacheTask>();

		const VkDeviceSize tangentSize  = meta->getVerticesCount() * sizeof(VertexTangent);
		const VkDeviceSize normalSize = meta->getVerticesCount() * sizeof(VertexNormal);
		const VkDeviceSize uv0Size = meta->getVerticesCount() * sizeof(VertexUv0);
		const VkDeviceSize positionsSize = meta->getVerticesCount() * sizeof(VertexPosition);
		const VkDeviceSize indicesSize   = meta->getIndicesCount() * sizeof(VertexIndexType);

		auto newAsset = std::make_shared<GPUStaticMeshAsset>(
			context,
			meta->getUUID(),
			fallback.get(),
			meta->getRelativePathUtf8(),
			tangentSize,
			sizeof(VertexTangent),
			normalSize,
			sizeof(VertexNormal),
			uv0Size,
			sizeof(VertexUv0),
			positionsSize,
			sizeof(VertexPosition),
			indicesSize,
			sizeof(VertexIndexType)
		);

		context->insertLRUAsset(meta->getUUID(), newAsset);
		newTask->meshAssetGPU = newAsset;
		newTask->cachePtr = meta;
		return newTask;
	}

}