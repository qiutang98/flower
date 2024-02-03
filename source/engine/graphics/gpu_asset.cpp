#include "gpu_asset.h"
#include "context.h"
#include <engine/asset/asset_texture.h>
#include <engine/engine.h>
#include <stb/stb_image.h>
#include <engine/asset/asset_staticmesh.h>
#include <engine/asset/assimp_import.h>
#include <asset/asset_manager.h>

#define TINYGLTF_NO_INCLUDE_STB_IMAGE_WRITE
#define TINYGLTF_NO_INCLUDE_STB_IMAGE
#define TINYEXR_IMPLEMENTATION
#include <tinyexr/tinyexr.h>

namespace engine
{
	static std::string getRuntimeUniqueGPUAssetName(const std::string& in)
	{
		static size_t GRuntimeId = 0;
		GRuntimeId++;
		return std::format("GPUAssetId: {}. {}.", GRuntimeId, in);
	}

	GPUImageAsset::GPUImageAsset(
		GPUImageAsset* fallback, 
		VkFormat format, 
		const std::string& name, 
		uint32_t mipmapCount, 
		math::uvec3 dimension)
	: UploadAssetInterface(fallback)
	{
		CHECK(m_image == nullptr && "You must ensure image asset only init once.");

		VkImageCreateInfo info{};
		info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		info.flags = {};
		info.imageType = dimension.z != 1U ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
		info.format = format;
		info.extent.width  = dimension.x;
		info.extent.height = dimension.y;
		info.extent.depth  = dimension.z;

		info.arrayLayers = 1;
		info.mipLevels = mipmapCount;

		info.samples       = VK_SAMPLE_COUNT_1_BIT;
		info.tiling        = VK_IMAGE_TILING_OPTIMAL;
		info.usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		info.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
		info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		m_image = std::make_unique<VulkanImage>
		(
			getContext()->getVMAImage(),
			getRuntimeUniqueGPUAssetName(name).c_str(),
			info
		);
	}

	GPUImageAsset::~GPUImageAsset()
	{
		m_image.reset();
	}

	void GPUImageAsset::prepareToUpload(
		RHICommandBufferBase& cmd, VkImageSubresourceRange range)
	{
		m_image->transitionLayout(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, range);
	}

	void GPUImageAsset::finishUpload(
		RHICommandBufferBase& cmd, VkImageSubresourceRange range)
	{
		m_image->transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, range);
	}

	RawAssetTextureLoadTask::RawAssetTextureLoadTask()
	{
		cacheBin = std::make_unique<AssetTextureBin>();
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

			uint32_t mipWidth = std::max<uint32_t>(imageAssetGPU->getSelfImage().getExtent().width >> level, 1);
			uint32_t mipHeight = std::max<uint32_t>(imageAssetGPU->getSelfImage().getExtent().height >> level, 1);
			uint32_t mipDepth = std::max<uint32_t>(imageAssetGPU->getSelfImage().getExtent().depth >> level, 1);

			memcpy((void*)((char*)bufferPtrStart + bufferOffset), currentMip.data(), currentMipSize);

			region.bufferOffset = stageBufferOffset + bufferOffset;
			region.imageSubresource.mipLevel = level;
			region.imageExtent = { mipWidth, mipHeight, mipDepth };

			copyRegions.push_back(region);

			bufferOffset += currentMipSize;
			bufferSize += currentMipSize;
		}

		vkCmdCopyBufferToImage(
			commandBuffer.cmd, 
			stageBuffer, 
			imageAssetGPU->getSelfImage().getImage(),
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 
			(uint32_t)copyRegions.size(), 
			copyRegions.data());

		imageAssetGPU->finishUpload(commandBuffer, rangeAllMips);
	}

	std::shared_ptr<RawAssetTextureLoadTask> RawAssetTextureLoadTask::buildFlatTexture(
		const std::string& name, 
		const UUID& uuid, 
		const glm::uvec4& color, 
		const glm::uvec3& size, 
		VkFormat format)
	{
		ASSERT(!getContext()->isBuiltinAssetExist(uuid), "Persistent asset has exist, don't register repeatly.");
		auto newAsset = std::make_shared<GPUImageAsset>(nullptr, format, name, 1, size);

		// New engine asset.
		getContext()->insertBuiltinAsset(uuid, newAsset);

		// Create new task.
		std::shared_ptr<RawAssetTextureLoadTask> newTask = std::make_shared<RawAssetTextureLoadTask>();
		newTask->imageAssetGPU = newAsset;

		{

			AssetSaveInfo saveInfo = AssetSaveInfo::buildTemp(uuid);
			auto texturePtr = getAssetManager()->createAsset<AssetTexture>(saveInfo).lock();
			texturePtr->initBasicInfo(false, 1, format, size, 1.0f);
		}

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

	std::shared_ptr<RawAssetTextureLoadTask> RawAssetTextureLoadTask::buildTexture(
		const std::filesystem::path& path, 
		const UUID& uuid, 
		VkFormat format, 
		bool bSRGB, 
		uint channel,
		bool bMipmap,
		float alphaCoverage)
	{
		ASSERT(!getContext()->isBuiltinAssetExist(uuid), "Persistent asset has exist, don't register repeatly.");

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
			mipmapCount = getMipLevelsCount(texWidth, texHeight);
		}

		buildMipmapData8Bit(channel, 0, pixels, *newTask->cacheBin, bSRGB, mipmapCount, texWidth, texHeight, alphaCoverage);

		auto newAsset = std::make_shared<GPUImageAsset>(
			nullptr,
			format, 
			path.stem().string(), 
			mipmapCount,
			math::uvec3{ texWidth, texHeight, 1 });

		// New engine asset.
		{
			AssetSaveInfo saveInfo = AssetSaveInfo::buildTemp(uuid);
			auto texturePtr = getAssetManager()->createAsset<AssetTexture>(saveInfo).lock();
			texturePtr->initBasicInfo(bSRGB, mipmapCount, format, { texWidth, texHeight, 1 }, alphaCoverage);

			getContext()->insertBuiltinAsset(uuid, newAsset);
		}


		newTask->imageAssetGPU = newAsset;

		stbi_image_free(pixels);
		return newTask;
	}



	std::shared_ptr<RawAssetTextureLoadTask> RawAssetTextureLoadTask::buildExrTexture(
		const std::filesystem::path& path, const UUID& uuid, EImageFormatExr inFormat, const char* layerName)
	{
		VkFormat format = VK_FORMAT_R32_SFLOAT;
		int32_t pixelCount = 1;
		if (inFormat == EImageFormatExr::RGBA)
		{
			format = VK_FORMAT_R32G32B32A32_SFLOAT;
			pixelCount = 4;
		}

		ASSERT(!getContext()->isBuiltinAssetExist(uuid), "Persistent asset has exist, don't register repeatly.");

		float* loadPixelsRGBA;
		int width, height;

		const char* err = nullptr;
		if (TINYEXR_SUCCESS != LoadEXRWithLayer(&loadPixelsRGBA, &width, &height, path.string().c_str(), layerName, &err))
		{
			// Log error message if exist.
			if (err)
			{
				LOG_ERROR("Err import exr: {}.", err);
				FreeEXRErrorMessage(err);
			}

			LOG_ERROR("Fail import exr file {}!", path.string());

			// Load exr file fail, just return nullptr.
			return nullptr;
		}

		std::shared_ptr<RawAssetTextureLoadTask> newTask = std::make_shared<RawAssetTextureLoadTask>();

		const int32_t mipmapCount = 1;
		const bool bSrgb = false;
		const float alphaCoverage = 1.0f;

		{
			AssetTextureBin& cacheBin = *newTask->cacheBin;

			cacheBin.mipmapDatas.resize(mipmapCount);
			CHECK(mipmapCount == 1);

			auto& destMipData = cacheBin.mipmapDatas[0];

			// Float -> uint8_t.
			destMipData.resize(width * height * pixelCount * 4);
			float* pDestData = (float*)destMipData.data();
			if (inFormat == EImageFormatExr::RGBA)
			{
				memcpy(pDestData, loadPixelsRGBA, destMipData.size());
			}
			else
			{
				// De-interval from exr load.
				for (size_t i = 0; i < width * height; i++)
				{
					for (size_t j = 0; j < pixelCount; j++)
					{
						pDestData[i * pixelCount + j] = loadPixelsRGBA[i * 4 + j];
					}
				}
			}
		}

		auto newAsset = std::make_shared<GPUImageAsset>(
			nullptr,
			format,
			path.stem().string(),
			mipmapCount,
			math::uvec3{ width, height, 1 });

		// New engine asset.
		{
			AssetSaveInfo saveInfo = AssetSaveInfo::buildTemp(uuid);
			auto texturePtr = getAssetManager()->createAsset<AssetTexture>(saveInfo).lock();
			texturePtr->initBasicInfo(bSrgb, mipmapCount, format, { width, height, 1 }, alphaCoverage);

			getContext()->insertBuiltinAsset(uuid, newAsset);
		}

		newTask->imageAssetGPU = newAsset;

		free(loadPixelsRGBA);
		return newTask;
	}


	GPUStaticMeshAsset::GPUStaticMeshAsset(
		std::weak_ptr<AssetStaticMesh> asset,
		GPUStaticMeshAsset* fallback,
		const std::string& name,
		uint32_t verticesNum,
		uint32_t indicesNum)
		: UploadAssetInterface(fallback)
		, m_asset(asset)
		, m_verticesNum(verticesNum)
		, m_indicesNum(indicesNum)
	{
		// Bindless fetch, transfer copy.
		auto bufferFlagBasic = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		VmaAllocationCreateFlags bufferFlagVMA = {};
		if (getContext()->getGraphicsState().bSupportRaytrace)
		{
			// Raytracing accelerate struct, random shader fetch by address.
			bufferFlagBasic |= 
				VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | 
				VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
			bufferFlagVMA = {};
		}

		makeComponent(
			&m_indices,
			bufferFlagBasic | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
			getRuntimeUniqueGPUAssetName(name + "_indices"),
			bufferFlagVMA,
			sizeof(VertexIndexType), m_indicesNum);

		makeComponent(
			&m_positions,
			bufferFlagBasic | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			getRuntimeUniqueGPUAssetName(name + "_positions"),
			bufferFlagVMA,
			sizeof(VertexPosition), m_verticesNum);

		makeComponent(
			&m_normals,
			bufferFlagBasic | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			getRuntimeUniqueGPUAssetName(name + "_normals"),
			bufferFlagVMA,
			sizeof(VertexNormal), m_verticesNum);

		makeComponent(
			&m_uv0s,
			bufferFlagBasic | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			getRuntimeUniqueGPUAssetName(name + "_uv0s"),
			bufferFlagVMA,
			sizeof(VertexUv0), m_verticesNum);

		makeComponent(
			&m_tangents,
			bufferFlagBasic | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			getRuntimeUniqueGPUAssetName(name + "_tangents"),
			bufferFlagVMA,
			sizeof(VertexTangent), m_verticesNum);
	}


	GPUStaticMeshAsset::~GPUStaticMeshAsset()
	{
		GPUStaticMeshAsset* fallback = nullptr;
		const bool bReleasing = (Engine::get()->getModuleState() == Engine::EModuleState::Releasing);
		if (!bReleasing)
		{
			fallback = getContext()->getBuiltinStaticMeshBox().get();
		}

		freeComponent(&m_indices,   bReleasing ? nullptr : fallback->getIndices().buffer.get());
		freeComponent(&m_positions, bReleasing ? nullptr : fallback->getPositions().buffer.get());
		freeComponent(&m_normals,   bReleasing ? nullptr : fallback->getNormals().buffer.get());
		freeComponent(&m_uv0s,      bReleasing ? nullptr : fallback->getUV0s().buffer.get());
		freeComponent(&m_tangents,  bReleasing ? nullptr : fallback->getTangents().buffer.get());
	}


	uint32_t GPUStaticMeshAsset::getSize() const
	{
		return 
			  m_indices.buffer->getSize() 
			+ m_positions.buffer->getSize() 
			+ m_tangents.buffer->getSize() 
			+ m_normals.buffer->getSize() 
			+ uint32_t(m_uv0s.buffer->getSize());
	}

	void GPUStaticMeshAsset::makeComponent(
		ComponentBuffer* in, 
		VkBufferUsageFlags flags, 
		const std::string name,
		VmaAllocationCreateFlags vmaFlags,
		uint32_t stripeSize,
		uint32_t num)
	{
		in->num = num;
		in->stripeSize = stripeSize;

		VkDeviceSize size = stripeSize * num;

		in->buffer = std::make_unique<VulkanBuffer>(getContext()->getVMABuffer(), name, flags, vmaFlags, size);
		in->bindless = getContext()->getBindlessSSBOs().updateBufferToBindlessDescriptorSet(
			in->buffer->getVkBuffer(), 0, size);
	}

	void GPUStaticMeshAsset::freeComponent(ComponentBuffer* in, VulkanBuffer* fallback)
	{
		if (in->bindless != ~0)
		{
			getContext()->getBindlessSSBOs().freeBindlessImpl(
				in->bindless, fallback);
		}

		*in = { };
	}

	void AssetRawStaticMeshLoadTask::uploadFunction(
		uint32_t stageBufferOffset,
		void* bufferPtrStart,
		RHICommandBufferBase& commandBuffer,
		VulkanBuffer& stageBuffer)
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
			CHECK(meshAssetGPU->getIndices().buffer->getSize() == cacheIndices.size());
			VkBufferCopy regionIndex{};
			regionIndex.size = VkDeviceSize(cacheIndices.size());
			regionIndex.srcOffset = stageBufferOffset + indicesOffsetInSrcBuffer;
			regionIndex.dstOffset = 0;
			vkCmdCopyBuffer(
				commandBuffer.cmd,
				stageBuffer,
				meshAssetGPU->getIndices().buffer->getVkBuffer(),
				1,
				&regionIndex);
		}

		{
			CHECK(meshAssetGPU->getTangents().buffer->getSize() == cacheTangents.size());
			VkBufferCopy regionVertex{};
			regionVertex.size = VkDeviceSize(cacheTangents.size());
			regionVertex.srcOffset = stageBufferOffset + tangentOffsetInSrcBuffer;
			regionVertex.dstOffset = 0;
			vkCmdCopyBuffer(
				commandBuffer.cmd,
				stageBuffer,
				meshAssetGPU->getTangents().buffer->getVkBuffer(),
				1,
				&regionVertex);
		}
		{
			CHECK(meshAssetGPU->getNormals().buffer->getSize() == cacheNormals.size());
			VkBufferCopy regionVertex{};
			regionVertex.size = VkDeviceSize(cacheNormals.size());
			regionVertex.srcOffset = stageBufferOffset + normalOffsetInSrcBuffer;
			regionVertex.dstOffset = 0;
			vkCmdCopyBuffer(
				commandBuffer.cmd,
				stageBuffer,
				meshAssetGPU->getNormals().buffer->getVkBuffer(),
				1,
				&regionVertex);
		}
		{
			CHECK(meshAssetGPU->getUV0s().buffer->getSize() == cacheUv0s.size());
			VkBufferCopy regionVertex{};
			regionVertex.size = VkDeviceSize(cacheUv0s.size());
			regionVertex.srcOffset = stageBufferOffset + uv0OffsetInSrcBuffer;
			regionVertex.dstOffset = 0;
			vkCmdCopyBuffer(
				commandBuffer.cmd,
				stageBuffer,
				meshAssetGPU->getUV0s().buffer->getVkBuffer(),
				1,
				&regionVertex);
		}

		{
			CHECK(meshAssetGPU->getPositions().buffer->getSize() == cachePositions.size());
			VkBufferCopy regionVertex{};
			regionVertex.size = VkDeviceSize(cachePositions.size());
			regionVertex.srcOffset = stageBufferOffset + positionsOffsetInSrcBuffer;
			regionVertex.dstOffset = 0;
			vkCmdCopyBuffer(
				commandBuffer.cmd,
				stageBuffer,
				meshAssetGPU->getPositions().buffer->getVkBuffer(),
				1,
				&regionVertex);
		}
	}

	std::shared_ptr<AssetRawStaticMeshLoadTask> AssetRawStaticMeshLoadTask::buildFromPath(
		GPUStaticMeshAsset* fallback,
		const std::filesystem::path& path,
		const UUID& uuid,
		std::shared_ptr<AssetStaticMesh> assetIn)
	{
		// If no asset input, it is builtin.
		bool bBuiltin = !assetIn;

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

		const auto verticesCount = processor.getVerticesCount();
		const auto indicesCount = processor.getIndicesCount();

		newTask->cacheTangents.resize(verticesCount * sizeof(processor.getTangents()[0]));
		newTask->cacheNormals.resize(verticesCount * sizeof(processor.getNormals()[0]));
		newTask->cacheUv0s.resize(verticesCount * sizeof(processor.getUv0s()[0]));
		newTask->cachePositions.resize(verticesCount * sizeof(processor.getPositions()[0]));
		newTask->cacheIndices.resize(indicesCount * sizeof(processor.getIndices()[0]));

		{
			memcpy((void*)(newTask->cacheTangents.data()), (void*)processor.getTangents().data(), newTask->cacheTangents.size());
			memcpy((void*)(newTask->cacheNormals.data()), (void*)processor.getNormals().data(), newTask->cacheNormals.size());
			memcpy((void*)(newTask->cacheUv0s.data()), (void*)processor.getUv0s().data(), newTask->cacheUv0s.size());
			memcpy((void*)(newTask->cachePositions.data()), (void*)processor.getPositions().data(), newTask->cachePositions.size());
			memcpy((void*)(newTask->cacheIndices.data()), (void*)processor.getIndices().data(), newTask->cacheIndices.size());
		}

		ASSERT(processor.getSubmeshInfo().size() == 1, "Engine mesh only support one material and one submesh!");

		std::weak_ptr<AssetStaticMesh> assetWeak = assetIn;
		if (bBuiltin)
		{
			// Build asset in map.
			AssetSaveInfo saveInfo = AssetSaveInfo::buildTemp(uuid);
			auto newMeshAsset = getAssetManager()->createAsset<AssetStaticMesh>(saveInfo).lock();
			processor.fillMeshAssetMeta(*newMeshAsset);

			assetWeak = newMeshAsset;
		}

		auto newAsset = std::make_shared<GPUStaticMeshAsset>(
			assetWeak, 
			fallback, 
			path.string(), 
			(uint32_t)verticesCount, 
			(uint32_t)indicesCount);

		if (bBuiltin)
		{
			getContext()->insertBuiltinAsset(uuid, newAsset);
		}
		else
		{
			getContext()->insertLRUAsset(uuid, newAsset);
		}

		newTask->meshAssetGPU = newAsset;
		return newTask;
	}

	void AssetStaticMeshLoadTask::finishCallback()
	{
		meshAssetGPU->setAsyncLoadState(false);
	}

	BLASBuilder& GPUStaticMeshAsset::getOrBuilddBLAS()
	{
		if (!m_blasBuilder.isInit())
		{
			const auto& submeshes = m_asset.lock()->getSubMeshes();
			const uint32_t maxVertex = getVerticesCount();

			std::vector<BLASBuilder::BlasInput> allBlas(submeshes.size());
			for (size_t i = 0; i < submeshes.size(); i++)
			{
				const auto& submesh = submeshes[i];

				const uint32_t maxPrimitiveCount = submesh.indicesCount / 3;

				// Describe buffer as array of VertexObj.
				VkAccelerationStructureGeometryTrianglesDataKHR triangles{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR };
				triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;  // vec3 vertex position data.
				triangles.vertexData.deviceAddress = m_positions.buffer->getDeviceAddress();
				triangles.vertexStride = m_positions.stripeSize;
				triangles.indexType = VK_INDEX_TYPE_UINT32;
				triangles.indexData.deviceAddress = m_indices.buffer->getDeviceAddress();
				triangles.maxVertex = maxVertex;

				// Identify the above data as containing opaque triangles.
				VkAccelerationStructureGeometryKHR asGeom{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
				asGeom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
				asGeom.flags = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
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

}