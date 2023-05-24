#include "gpu_asset.h"
#include "rhi.h"


#pragma warning(disable: 4006)

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/GltfMaterial.h>
#include <util/assimp_helper.h>

#include <asset/asset_texture.h>

namespace engine
{
	static std::string getRuntimeUniqueGPUAssetName(const std::string& in)
	{
		static size_t GRuntimeId = 0;
		GRuntimeId++;
		return std::format("GPUAssetId: {}. {}.", GRuntimeId, in);
	}

	GPUImageAsset::GPUImageAsset(
		VulkanContext* context,
		GPUImageAsset* fallback,
		VkFormat format,
		const std::string& name,
		uint32_t mipmapCount,
		uint32_t width,
		uint32_t height,
		uint32_t depth)
		: m_context(context), LRUAssetInterface(fallback)
	{
		CHECK(m_image == nullptr && "You must ensure image asset only init once.");

		VkImageCreateInfo info{};
		info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		info.flags = {};
		info.imageType = depth != 1 ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
		info.format = format;
		info.extent.width = width;
		info.extent.height = height;
		info.extent.depth = depth;
		info.arrayLayers = 1;
		info.mipLevels = mipmapCount;
		info.samples = VK_SAMPLE_COUNT_1_BIT;
		info.tiling = VK_IMAGE_TILING_OPTIMAL;
		info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		m_image = std::make_unique<VulkanImage>(m_context, getRuntimeUniqueGPUAssetName(name).c_str(), info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	}

	GPUImageAsset::~GPUImageAsset()
	{
		if (m_bindlessIndex != ~0)
		{
			m_context->getBindlessTexture().freeBindlessImpl(m_bindlessIndex, m_context->isReleaseing() ? nullptr : m_context->getEngineTextureWhite()->m_image.get());
		}

		m_image.reset();
	}

	void GPUImageAsset::prepareToUpload(RHICommandBufferBase& cmd, VkImageSubresourceRange range)
	{
		CHECK(m_bindlessIndex == ~0);
		m_image->transitionLayout(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, range);
	}

	void GPUImageAsset::finishUpload(RHICommandBufferBase& cmd, VkImageSubresourceRange range)
	{
		m_image->transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, range);
		m_bindlessIndex = m_context->getBindlessTexture().updateTextureToBindlessDescriptorSet(m_image->getOrCreateView(buildBasicImageSubresource(), m_image->getInfo().imageType == VK_IMAGE_TYPE_3D ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_2D));

		CHECK(m_bindlessIndex != ~0);
	}

	void RawAssetTextureLoadTask::uploadFunction(
		uint32_t stageBufferOffset, 
		void* bufferPtrStart, 
		RHICommandBufferBase& commandBuffer,
		VulkanBuffer& stageBuffer)
	{
		VkImageSubresourceRange rangeAllMips = buildBasicImageSubresource();
		rangeAllMips.levelCount = (uint32_t)cacheBin->mipmapDatas.size();;

		imageAssetGPU->prepareToUpload(commandBuffer, buildBasicImageSubresource());

		uint32_t bufferOffset = 0;
		uint32_t bufferSize = 0;

		VkBufferImageCopy region{};
		region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		region.imageSubresource.baseArrayLayer = 0;
		region.imageSubresource.layerCount = 1;
		region.imageOffset = { 0, 0, 0 };
		region.bufferRowLength = 0;
		region.bufferImageHeight = 0;

		std::vector<VkBufferImageCopy> copyRegions{};
		const auto& mipmapDatas = cacheBin->mipmapDatas;
		for (uint32_t level = 0; level < rangeAllMips.levelCount; level++)
		{
			const auto& currentMip = mipmapDatas.at(level);
			const uint32_t currentMipSize = (uint32_t)currentMip.size();

			uint32_t mipWidth  = std::max<uint32_t>(imageAssetGPU->getImage().getExtent().width  >> level, 1);
			uint32_t mipHeight = std::max<uint32_t>(imageAssetGPU->getImage().getExtent().height >> level, 1);
			uint32_t mipDepth  = std::max<uint32_t>(imageAssetGPU->getImage().getExtent().depth  >> level, 1);

			memcpy((void*)((char*)bufferPtrStart + bufferOffset), currentMip.data(), currentMipSize);

			region.bufferOffset = stageBufferOffset + bufferOffset;
			region.imageSubresource.mipLevel = level;
			region.imageExtent = { mipWidth, mipHeight, mipDepth };

			copyRegions.push_back(region);

			bufferOffset += currentMipSize;
			bufferSize += currentMipSize;
		}

		vkCmdCopyBufferToImage(commandBuffer.cmd, stageBuffer, imageAssetGPU->getImage().getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, (uint32_t)copyRegions.size(), copyRegions.data());

		imageAssetGPU->finishUpload(commandBuffer, rangeAllMips);
	}

	extern 	void buildMipmapDataRGBA8(
		stbi_uc* srcPixels,
		AssetTextureBin& outBinData,
		float alphaCutOff,
		uint32_t mipmapCount,
		bool bSRGB,
		uint32_t inWidth,
		uint32_t inHeight);

	RawAssetTextureLoadTask::RawAssetTextureLoadTask()
	{
		cacheBin = std::make_unique<AssetTextureBin>();
	}

	std::shared_ptr<RawAssetTextureLoadTask> RawAssetTextureLoadTask::buildTexture(
		bool bEngineTex,
		VulkanContext* context, const std::filesystem::path& path, const UUID& uuid, VkFormat format, bool bSRGB, bool bMipmap)
	{
		if (bEngineTex)
		{
			ASSERT(!context->isEngineAssetExist(uuid), "Persistent asset has exist, don't register repeatly.");
		}
		else
		{
			if (context->isLRUAssetExist(uuid))
			{
				return nullptr;
			}
		}


		int32_t texWidth, texHeight, texChannels;
		stbi_uc* pixels = stbi_load(path.string().c_str(), &texWidth, &texHeight, &texChannels, 4);

		if (!pixels)
		{
			LOG_ERROR("Fail to load image {0}.", path.string());
			return nullptr;
		}

		std::shared_ptr<RawAssetTextureLoadTask> newTask = std::make_shared<RawAssetTextureLoadTask>();

		const bool bPOT = isPOT(texWidth) && isPOT(texHeight);
		uint32_t mipmapCount = 1;
		if (bMipmap && bPOT)
		{
			uint32_t maxDim = math::max(texWidth, texHeight);
			mipmapCount = std::bit_width(maxDim);
		}

		buildMipmapDataRGBA8(pixels, *newTask->cacheBin, 1.0f, mipmapCount, bSRGB, texWidth, texHeight);

		auto newAsset = std::make_shared<GPUImageAsset>(
			context,
			bEngineTex ? nullptr : context->getEngineTextureWhite().get(),
			format,
			path.stem().string(), 
			mipmapCount,
			texWidth, 
			texHeight, 
			1);

		// New engine asset.
		if (bEngineTex)
		{
			context->insertEngineAsset(uuid, newAsset);
		}
		else
		{
			context->insertLRUAsset(uuid, newAsset);
		}

		newTask->imageAssetGPU = newAsset;

		stbi_image_free(pixels);
		return newTask;
	}

	std::shared_ptr<RawAssetTextureLoadTask> RawAssetTextureLoadTask::buildEngine3dTexture(
		VulkanContext* context, const std::filesystem::path& path, const UUID& uuid, VkFormat format,
		math::uvec3 dim)
	{
		ASSERT(!context->isEngineAssetExist(uuid), "Persistent asset has exist, don't register repeatly.");

		auto newAsset = std::make_shared<GPUImageAsset>(
			context,
			nullptr,
			format,
			path.string(),
			1, // Mipmap count.
			dim.x,
			dim.y,
			dim.z);
		context->insertEngineAsset(uuid, newAsset);

		std::shared_ptr<RawAssetTextureLoadTask> newTask = std::make_shared<RawAssetTextureLoadTask>();
		newTask->imageAssetGPU = newAsset;

		newTask->cacheBin->mipmapDatas.resize(1);
		newTask->cacheBin->mipmapDatas[0].resize(dim.x * dim.y * dim.z * 4 * 4);

		auto file = std::ifstream(path, std::ios::binary);

		file.seekg(0, std::ios::end);
		int length = (int)file.tellg();

		CHECK(length == newTask->cacheBin->mipmapDatas[0].size());
		file.seekg(0, std::ios::beg);
		file.read((char*)newTask->cacheBin->mipmapDatas[0].data(), length);

		return newTask;
	}

	std::shared_ptr<RawAssetTextureLoadTask> RawAssetTextureLoadTask::buildEngineFlatTexture(
		VulkanContext* context, const std::string& name, const UUID& uuid, const glm::uvec4& color, const glm::uvec3& size, VkFormat format)
	{
		ASSERT(!context->isEngineAssetExist(uuid), "Persistent asset has exist, don't register repeatly.");

		auto newAsset = std::make_shared<GPUImageAsset>(
			context,
			nullptr,
			format,
			name,
			1, // Mipmap count.
			size.x,
			size.y,
			size.z);

		// New engine asset.
		context->insertEngineAsset(uuid, newAsset);

		// Create new task.
		std::shared_ptr<RawAssetTextureLoadTask> newTask = std::make_shared<RawAssetTextureLoadTask>();
		newTask->imageAssetGPU = newAsset;

		// Prepare upload data.
		newTask->cacheBin->mipmapDatas.resize(1);
		newTask->cacheBin->mipmapDatas[0].resize(size.x * size.y * size.z * 4);
		auto& mip0 = newTask->cacheBin->mipmapDatas[0];

		for (size_t i = 0; i < mip0.size(); i += 4)
		{
			mip0[i + 0] = uint8_t(color.x);
			mip0[i + 1] = uint8_t(color.y);
			mip0[i + 2] = uint8_t(color.z);
			mip0[i + 3] = uint8_t(color.w);
		}

		// NOTE: GPU Memory align, which make small texture size min is 512 byte. And may size no equal.
		//       But at least one thing is guarantee is that cache data size must less than upload Size.
		CHECK(mip0.size() <= newTask->uploadSize());
		return newTask;
	}

	GPUStaticMeshAsset::GPUStaticMeshAsset(
		VulkanContext* context, 
		UUID assetId,
		GPUStaticMeshAsset* fallback, 
		const std::string& name, 
		VkDeviceSize tangentSize,
		VkDeviceSize tangentStripSize,
		VkDeviceSize normalSize,
		VkDeviceSize normalStripSize,
		VkDeviceSize uv0Size,
		VkDeviceSize uv0StripSize,
		VkDeviceSize positionsSize, 
		VkDeviceSize positionStripSize, 
		VkDeviceSize indicesSize, 
		VkDeviceSize indexStripSize)
		: m_context(context), LRUAssetInterface(fallback)
		, m_assetId(assetId)
		, m_tangentsSize(tangentSize)
		, m_tangentStripSize(tangentStripSize)
		, m_normalSize(normalSize)
		, m_normalStripSize(normalStripSize)
		, m_uv0Size(uv0Size)
		, m_uv0StripSize(uv0StripSize)
		, m_positionsSize(positionsSize)
		, m_positionStripSize(positionStripSize)
		, m_indicesSize(indicesSize)
		, m_indexStripSize(indexStripSize)
	{
		ASSERT(
			   m_tangents == nullptr 
			&& m_normals == nullptr
			&& m_uv0s == nullptr
			&& m_indices == nullptr 
			&& m_positions == nullptr
			 , "You must ensure mesh asset only init once.");

		// Bindless fetch, transfer copy.
		auto bufferFlagBasic = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		VmaAllocationCreateFlags bufferFlagVMA = {};
		if (m_context->getGraphicsCardState().bSupportRaytrace)
		{
			// Raytracing accelerate struct, random shader fetch by address.
			bufferFlagBasic |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
			bufferFlagVMA = {};
		}

		m_tangents = std::make_unique<VulkanBuffer>(
			m_context,
			getRuntimeUniqueGPUAssetName(name + "_tangents"),
			bufferFlagBasic | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			bufferFlagVMA,
			tangentSize
		);
		m_normals = std::make_unique<VulkanBuffer>(
			m_context,
			getRuntimeUniqueGPUAssetName(name + "_normals"),
			bufferFlagBasic | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			bufferFlagVMA,
			normalSize
		);
		m_uv0s = std::make_unique<VulkanBuffer>(
			m_context,
			getRuntimeUniqueGPUAssetName(name + "_uv0s"),
			bufferFlagBasic | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			bufferFlagVMA,
			uv0Size
		);

		m_positions = std::make_unique<VulkanBuffer>(
			m_context,
			getRuntimeUniqueGPUAssetName(name + "_positions"),
			bufferFlagBasic | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			bufferFlagVMA,
			positionsSize
		);

		ASSERT(
			(positionsSize / positionStripSize) == (tangentSize / tangentStripSize) &&
			(positionsSize / positionStripSize) == (normalSize / normalStripSize) &&
			(positionsSize / positionStripSize) == (uv0Size / uv0StripSize) 
			, "Vertices data no correct!");

		m_indices = std::make_unique<VulkanBuffer>(
			m_context,
			getRuntimeUniqueGPUAssetName(name + "_indices"),
			bufferFlagBasic | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
			bufferFlagVMA,
			indicesSize
		);

		m_tangentsBindless = m_context->getBindlessSSBOs().updateBufferToBindlessDescriptorSet(m_tangents->getVkBuffer(), 0, m_tangentsSize);
		m_normalsBindless = m_context->getBindlessSSBOs().updateBufferToBindlessDescriptorSet(m_normals->getVkBuffer(), 0, m_normalSize);
		m_uv0sBindless = m_context->getBindlessSSBOs().updateBufferToBindlessDescriptorSet(m_uv0s->getVkBuffer(), 0, m_uv0Size);
		m_indicesBindless = m_context->getBindlessSSBOs().updateBufferToBindlessDescriptorSet(m_indices->getVkBuffer(), 0, m_indicesSize);
		m_positionBindless = m_context->getBindlessSSBOs().updateBufferToBindlessDescriptorSet(m_positions->getVkBuffer(), 0, m_positionsSize);
	}

	GPUStaticMeshAsset::~GPUStaticMeshAsset()
	{
		if (m_indicesBindless != ~0)
		{
			m_context->getBindlessSSBOs().freeBindlessImpl(m_indicesBindless, m_context->isReleaseing() ? nullptr : m_context->getEngineStaticMeshBox()->getIndices());
		}
		if (m_tangentsBindless != ~0)
		{
			m_context->getBindlessSSBOs().freeBindlessImpl(m_tangentsBindless, m_context->isReleaseing() ? nullptr : m_context->getEngineStaticMeshBox()->getTangents());
		}
		if (m_normalsBindless != ~0)
		{
			m_context->getBindlessSSBOs().freeBindlessImpl(m_normalsBindless, m_context->isReleaseing() ? nullptr : m_context->getEngineStaticMeshBox()->getNormals());
		}
		if (m_uv0sBindless != ~0)
		{
			m_context->getBindlessSSBOs().freeBindlessImpl(m_uv0sBindless, m_context->isReleaseing() ? nullptr : m_context->getEngineStaticMeshBox()->getUv0s());
		}
		if (m_positionBindless != ~0)
		{
			m_context->getBindlessSSBOs().freeBindlessImpl(m_positionBindless, m_context->isReleaseing() ? nullptr : m_context->getEngineStaticMeshBox()->getPosition());
		}

		m_indicesBindless = ~0;
		m_tangentsBindless = ~0;
		m_uv0sBindless = ~0;
		m_normalsBindless = ~0;
		m_positionBindless = ~0;

		m_indices.reset();
		m_tangents.reset();
		m_normals.reset();
		m_uv0s.reset();
		m_positions.reset();

		m_blasBuilder.destroy();
	}

	BLASBuilder& GPUStaticMeshAsset::getOrBuilddBLAS()
	{
		if (!m_blasBuilder.isInit())
		{
			std::vector<StaticMeshSubMesh> submeshes;
			if (m_context->isEngineAssetExist(m_assetId))
			{
				// Engine asset. one sub mesh.
				submeshes.resize(1);
				submeshes[0].bounds = m_context->getEngineMeshRenderBounds(m_assetId);
				submeshes[0].indicesCount = getIndicesCount();
				submeshes[0].indicesStart = 0;
				submeshes[0].material = {};
			}
			else
			{
				auto asset = std::dynamic_pointer_cast<AssetStaticMesh>(getAssetSystem()->getAsset(m_assetId));
				submeshes = asset->getSubMeshes();
			}

			const uint32_t maxVertex = getVerticesCount();

			std::vector<BLASBuilder::BlasInput> allBlas(submeshes.size());
			for (size_t i = 0; i < submeshes.size(); i++)
			{
				const auto& submesh = submeshes[i];

				const uint32_t maxPrimitiveCount = submesh.indicesCount / 3;

				// Describe buffer as array of VertexObj.
				VkAccelerationStructureGeometryTrianglesDataKHR triangles{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR };
				triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;  // vec3 vertex position data.
				triangles.vertexData.deviceAddress = m_positions->getDeviceAddress();
				triangles.vertexStride = m_positionStripSize;
				triangles.indexType = VK_INDEX_TYPE_UINT32;
				triangles.indexData.deviceAddress = m_indices->getDeviceAddress();
				triangles.maxVertex = maxVertex;

				// Identify the above data as containing opaque triangles.
				VkAccelerationStructureGeometryKHR asGeom{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
				asGeom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
				asGeom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
				asGeom.geometry.triangles = triangles;

				VkAccelerationStructureBuildRangeInfoKHR offset{ };
				offset.firstVertex = 0; // No vertex offset, current all vertex buffer start from zero.
				offset.primitiveCount = maxPrimitiveCount;
				offset.primitiveOffset = submesh.indicesStart * sizeof(VertexIndexType);
				offset.transformOffset = 0;

				allBlas[i].asGeometry.emplace_back(asGeom);
				allBlas[i].asBuildOffsetInfo.emplace_back(offset);
			}
			m_blasBuilder.build(allBlas, 
				VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | 
				VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR);
		}

		return m_blasBuilder;
	}

	bool GPUStaticMeshAsset::isEngineAsset() const
	{
		return getContext()->isEngineAssetExist(m_assetId);
	}

	void AssetRawStaticMeshLoadTask::uploadFunction(
		uint32_t stageBufferOffset, 
		void* bufferPtrStart, RHICommandBufferBase& commandBuffer, VulkanBuffer& stageBuffer)
	{
		CHECK(uploadSize() == uint32_t(
			cacheIndices.size() + cacheTangents.size() + cacheNormals.size() + cacheUv0s.size() + cachePositions.size()));

		uint32_t indicesOffsetInSrcBuffer = 0;
		uint32_t tangentOffsetInSrcBuffer = indicesOffsetInSrcBuffer + uint32_t(cacheIndices.size());
		uint32_t normalOffsetInSrcBuffer = tangentOffsetInSrcBuffer + uint32_t(cacheTangents.size());
		uint32_t uv0OffsetInSrcBuffer = normalOffsetInSrcBuffer + uint32_t(cacheNormals.size());
		uint32_t positionsOffsetInSrcBuffer = uv0OffsetInSrcBuffer + uint32_t(cacheUv0s.size());

		memcpy((void*)((char*)bufferPtrStart + indicesOffsetInSrcBuffer), cacheIndices.data(), cacheIndices.size());
		memcpy((void*)((char*)bufferPtrStart + tangentOffsetInSrcBuffer), cacheTangents.data(), cacheTangents.size());
		memcpy((void*)((char*)bufferPtrStart + normalOffsetInSrcBuffer), cacheNormals.data(), cacheNormals.size());
		memcpy((void*)((char*)bufferPtrStart + uv0OffsetInSrcBuffer), cacheUv0s.data(), cacheUv0s.size());
		memcpy((void*)((char*)bufferPtrStart + positionsOffsetInSrcBuffer), cachePositions.data(), cachePositions.size());

		{
			VkBufferCopy regionIndex{};
			regionIndex.size = VkDeviceSize(cacheIndices.size());
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
			regionVertex.size = VkDeviceSize(cacheTangents.size());
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
			regionVertex.size = VkDeviceSize(cacheNormals.size());
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
			regionVertex.size = VkDeviceSize(cacheUv0s.size());
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
			regionVertex.size = VkDeviceSize(cachePositions.size());
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

	std::shared_ptr<AssetRawStaticMeshLoadTask> AssetRawStaticMeshLoadTask::buildFromPath(
		VulkanContext* context,
		const std::filesystem::path& path, 
		const UUID& uuid,
		StaticMeshRenderBounds& outBounds)
	{
		ASSERT(!context->isEngineAssetExist(uuid), "Build from path is persistent asset, only init once.");

		Assimp::Importer importer;
		const aiScene* scene = importer.ReadFile(path.string(),
			aiProcessPreset_TargetRealtime_Fast | aiProcess_FlipUVs | aiProcess_GenBoundingBoxes);
		if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
		{
			LOG_ERROR("Assimp import fail: {0}.", importer.GetErrorString());
			return nullptr;
		}

		AssimpStaticMeshImporter processor(path);
		processor.processNode(scene->mRootNode, scene);

		auto newTask = std::make_shared<AssetRawStaticMeshLoadTask>();

		newTask->cacheTangents.resize(processor.getVerticesCount() * sizeof(processor.getTangents()[0]));
		newTask->cacheNormals.resize(processor.getVerticesCount() * sizeof(processor.getNormals()[0]));
		newTask->cacheUv0s.resize(processor.getVerticesCount() * sizeof(processor.getUv0s()[0]));
		newTask->cachePositions.resize(processor.getVerticesCount() * sizeof(processor.getPositions()[0]));
		newTask->cacheIndices.resize(processor.getIndicesCount() * sizeof(processor.getIndices()[0]));

		{
			memcpy((void*)(newTask->cacheTangents.data()), (void*)processor.getTangents().data(), newTask->cacheTangents.size());
			memcpy((void*)(newTask->cacheNormals.data()), (void*)processor.getNormals().data(), newTask->cacheNormals.size());
			memcpy((void*)(newTask->cacheUv0s.data()), (void*)processor.getUv0s().data(), newTask->cacheUv0s.size());
			memcpy((void*)(newTask->cachePositions.data()), (void*)processor.getPositions().data(), newTask->cachePositions.size());
			memcpy((void*)(newTask->cacheIndices.data()), (void*)processor.getIndices().data(), newTask->cacheIndices.size());
		}


		ASSERT(processor.getSubmeshInfo().size() == 1, "Engine mesh only support one material and one submesh!");
		outBounds = processor.getSubmeshInfo()[0].bounds;

		auto newAsset = std::make_shared<GPUStaticMeshAsset>(
			context,
			uuid,
			nullptr,
			path.string(),
			newTask->cacheTangents.size(),
			sizeof(processor.getTangents()[0]),
			newTask->cacheNormals.size(),
			sizeof(processor.getNormals()[0]),
			newTask->cacheUv0s.size(),
			sizeof(processor.getUv0s()[0]),
			newTask->cachePositions.size(),
			sizeof(processor.getPositions()[0]),
			newTask->cacheIndices.size(),
			sizeof(processor.getIndices()[0])
		);

		context->insertEngineAsset(uuid, newAsset);
		newTask->meshAssetGPU = newAsset;
		return newTask;
	}
}