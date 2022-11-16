#include "Pch.h"
#include "TextureManager.h"

#include <stb/stb_image.h>
#include <stb/stb_image_resize.h>

#pragma warning(disable: 4244)

namespace Flower
{
	const UUID EngineTextures::GWhiteTextureUUID = "0d6e103f-138a-482a-8a28-5116631a2e32";
	const UUID EngineTextures::GGreyTextureUUID = "6caa6c06-3c71-4b36-bb88-e0c577a06c60";
	const UUID EngineTextures::GBlackTextureUUID = "c11cc2f7-3c5d-458d-a2b4-68ebf7612948";
	const UUID EngineTextures::GTranslucentTextureUUID = "e515dc68-4947-4ffe-83c3-1696e3aeaf2d";
	const UUID EngineTextures::GNormalTextureUUID = "1b26e66d-855b-4c1b-b13d-d88305a78c9e";
	const UUID EngineTextures::GDefaultSpecularUUID = "424e83d1-ec41-464d-9733-7fcded599fbb";

	const UUID EngineTextures::GCloudWeatherUUID = "a27c14c4-2bd5-4d58-b416-801ff2d8c71e";
	const UUID EngineTextures::GCloudGradientUUID = "987b3e4d-27b7-4d49-bf53-6399a63b61cb";

	uint32_t EngineTextures::GWhiteTextureId = 0;
	uint32_t EngineTextures::GGreyTextureId = 0;
	uint32_t EngineTextures::GBlackTextureId = 0;
	uint32_t EngineTextures::GTranslucentTextureId = 0;
	uint32_t EngineTextures::GNormalTextureId = 0;
	uint32_t EngineTextures::GDefaultSpecularId = 0;
	uint32_t EngineTextures::GCloudWeatherId = 0;
	uint32_t EngineTextures::GCloudGradientId = 0;

	void ImageAssetHeader::buildSnapshotData2D(std::shared_ptr<ImageAssetBin> inBin)
	{
		CHECK(m_depth == 1u && m_binDataUUID == inBin->getBinUUID());

		if (m_width >= GAssetSnapshotMaxDim || m_height >= GAssetSnapshotMaxDim)
		{
			if (m_width > m_height)
			{
				m_widthSnapShot = GAssetSnapshotMaxDim;
				m_heightSnapShot = m_height / (m_width / GAssetSnapshotMaxDim);
			}
			else if (m_height > m_width)
			{
				m_heightSnapShot = GAssetSnapshotMaxDim;
				m_widthSnapShot = m_width / (m_height / GAssetSnapshotMaxDim);
			}
			else
			{
				m_heightSnapShot = GAssetSnapshotMaxDim;
				m_widthSnapShot = GAssetSnapshotMaxDim;
			}
		}
		else
		{
			m_heightSnapShot = m_height;
			m_widthSnapShot = m_width;
		}

		if (m_bHdr)
		{
			m_snapshotData.resize(m_widthSnapShot * m_heightSnapShot * GAssetTextureChannels * 4);

			stbir_resize_float(
				(float*)inBin->m_rawData.data(),
				m_width,
				m_height,
				0,
				(float*)m_snapshotData.data(),
				m_widthSnapShot,
				m_heightSnapShot,
				0,
				GAssetTextureChannels
			);

			// Post process.
			for (size_t i = 0; i < m_snapshotData.size(); i += GAssetTextureChannels * 4)
			{
				//... TODO:
			}
		}
		else
		{
			m_snapshotData.resize(m_widthSnapShot * m_heightSnapShot * GAssetTextureChannels);

			if (m_bSrgb)
			{
				stbir_resize_uint8_srgb_edgemode(
					inBin->m_rawData.data(),
					m_width,
					m_height,
					0,
					m_snapshotData.data(),
					m_widthSnapShot,
					m_heightSnapShot,
					0,
					GAssetTextureChannels,
					GAssetTextureChannels - 1,
					STBIR_FLAG_ALPHA_PREMULTIPLIED,
					STBIR_EDGE_CLAMP
				);

				// Post process.
				for (size_t i = 0; i < m_snapshotData.size(); i += GAssetTextureChannels)
				{
					m_snapshotData[i + 0] = linearToSrgb(m_snapshotData[i + 0]);
					m_snapshotData[i + 1] = linearToSrgb(m_snapshotData[i + 1]);
					m_snapshotData[i + 2] = linearToSrgb(m_snapshotData[i + 2]);
				}
			}
			else
			{
				stbir_resize_uint8(
					inBin->m_rawData.data(),
					m_width,
					m_height,
					0,
					m_snapshotData.data(),
					m_widthSnapShot,
					m_heightSnapShot,
					0,
					GAssetTextureChannels
				);
			}
		}

		

	}

	float getAlphaCoverageRGBA8(const unsigned char* data, uint32_t width, uint32_t height, float scale, int cutoff)
	{
		// float value may no enough for multi add.
		double value = 0.0;

		// 4 char to 1 uint32_t
		uint32_t* pImg = (uint32_t*)data;

		// Loop all texture to get coverage alpha data.
		for (uint32_t y = 0; y < height; y++)
		{
			for (uint32_t x = 0; x < width; x++)
			{
				// pImg++ to next pixel 4 char = 1 uint32_t
				uint8_t* pPixel = (uint8_t*)pImg++;

				// alpha in pixel[3]
				int alpha = (int)(scale * (float)pPixel[3]);

				if (alpha > 255) { alpha = 255; }
				if (alpha <= cutoff) { continue; }

				value += alpha;
			}
		}
		return (float)(value / (height * width * 255));
	}

	void scaleAlpha(unsigned char* data, uint32_t width, uint32_t height, float scale)
	{
		uint32_t* pImg = (uint32_t*)data;

		for (uint32_t y = 0; y < height; y++)
		{
			for (uint32_t x = 0; x < width; x++)
			{
				uint8_t* pPixel = (uint8_t*)pImg++;

				int alpha = (int)(scale * (float)pPixel[3]);
				if (alpha > 255) { alpha = 255; }

				pPixel[3] = alpha;
			}
		}
	}

	void ImageAssetBin::buildMipmapDataRGBA8(ImageAssetHeader* header, float cutOff)
	{
		float alphaCoverageMip0 = 1.0f;
		m_mipmapData.resize(header->getMipmapCount());

		// Look no good when convert to linear space do mipmap.
		const bool bSrgb = header->isSRGB();

		for (size_t mip = 0; mip < m_mipmapData.size(); mip++)
		{
			auto& destMipData = m_mipmapData[mip];

			uint32_t destWidth = std::max<uint32_t>(header->getWidth() >> mip, 1);
			uint32_t destHeight = std::max<uint32_t>(header->getHeight() >> mip, 1);

			if (mip == 0)
			{
				destMipData = m_rawData;

				alphaCoverageMip0 = cutOff < 1.0f
					? getAlphaCoverageRGBA8(destMipData.data(), destWidth, destHeight, 1.0f, (int)(cutOff * 255))
					: 1.0f;
			}
			else // Other mip process.
			{
				const size_t srcMip = mip - 1;
				const auto& srcMipData = m_mipmapData[srcMip];

				destMipData.resize(destWidth * destHeight * 4);

				for (size_t y = 0; y < destHeight; y++)
				{
					for (size_t x = 0; x < destWidth; x++)
					{
						// Get src data.
						uint32_t srcWidth = std::max<uint32_t>(header->getWidth() >> srcMip, 1);
						uint32_t srcHeight = std::max<uint32_t>(header->getHeight() >> srcMip, 1);

						// Clamp src data fetech edge.
						size_t srcX0 = (size_t)std::min<uint32_t>(uint32_t(x * 2 + 0), srcWidth - 1);
						size_t srcX1 = (size_t)std::min<uint32_t>(uint32_t(x * 2 + 1), srcWidth - 1);
						size_t srcY0 = (size_t)std::min<uint32_t>(uint32_t(y * 2 + 0), srcHeight - 1);
						size_t srcY1 = (size_t)std::min<uint32_t>(uint32_t(y * 2 + 1), srcHeight - 1);

						// Prepare src pixel start pos. * 4 is because per pixel has RGBA four components.
						size_t srcPixelStart[] =
						{
							(srcY0 * srcWidth + srcX0) * 4, // X0Y0
							(srcY0 * srcWidth + srcX1) * 4, // X1Y0
							(srcY1 * srcWidth + srcX0) * 4, // X0Y1
							(srcY1 * srcWidth + srcX1) * 4, // X1Y1
						};

						// Perpixel own 4 uint8_t pack to one uint32_t value.
						const uint32_t* pSrcData = (const uint32_t*)srcMipData.data();

						uint32_t R = 0, G = 0, B = 0, A = 0;
						for (size_t i = 0; i < 4; i++)
						{
							R += bSrgb ? srgbToLinear(srcMipData[srcPixelStart[i] + 0]) : srcMipData[srcPixelStart[i] + 0];
							G += bSrgb ? srgbToLinear(srcMipData[srcPixelStart[i] + 1]) : srcMipData[srcPixelStart[i] + 1];
							B += bSrgb ? srgbToLinear(srcMipData[srcPixelStart[i] + 2]) : srcMipData[srcPixelStart[i] + 2];

							A += srcMipData[srcPixelStart[i] + 3];
						}
						R /= 4; R = bSrgb ? linearToSrgb(R) : R;
						G /= 4; G = bSrgb ? linearToSrgb(G) : G;
						B /= 4; B = bSrgb ? linearToSrgb(B) : B;
						A /= 4;

						// Store to dest.
						size_t destPixelPosStart = (y * destWidth + x) * 4;
						destMipData[destPixelPosStart + 0] = R;
						destMipData[destPixelPosStart + 1] = G;
						destMipData[destPixelPosStart + 2] = B;
						destMipData[destPixelPosStart + 3] = A;
					}
				}

				if (alphaCoverageMip0 < 1.0f)
				{
					float ini = 0;
					float fin = 10;

					float mid;
					float alphaPercentage;

					// find best alpha coverage for mip-map.
					int iter = 0;
					for (; iter < 50; iter++)
					{
						mid = (ini + fin) / 2;
						alphaPercentage = getAlphaCoverageRGBA8(destMipData.data(), destWidth, destHeight, mid, (int)(cutOff * 255));

						if (glm::abs(alphaPercentage - alphaCoverageMip0) < .001) { break; }
						if (alphaPercentage > alphaCoverageMip0) { fin = mid; }
						if (alphaPercentage < alphaCoverageMip0) { ini = mid; }
					}

					scaleAlpha(destMipData.data(), destWidth, destHeight, mid);
				}
			}

			CHECK(destWidth * destHeight * 4 == destMipData.size());
		}
	}

	bool ImageAssetHeader::initFromRaw2DLDR(
		const std::filesystem::path& rawPath,
		bool bSRGB,
		float cutOff,
		bool bBuildMipmap)
	{
		setCacheBinData(std::make_shared<ImageAssetBin>(rawPath.filename().string()));
		auto processingImageBin = getBinData<ImageAssetBin>();

		int32_t texWidth, texHeight, texChannels;
		stbi_uc* pixels = stbi_load(rawPath.string().c_str(), &texWidth, &texHeight, &texChannels, GAssetTextureChannels);

		if (!pixels)
		{
			LOG_ERROR("Fail to load image {0}.", rawPath.string());
			return false;
		}

		m_bSrgb = bSRGB;
		m_width = texWidth;
		m_height = texHeight;
		m_depth = 1u;

		// Copy raw data to bin asset.
		processingImageBin->m_rawData.resize(m_width * m_height * m_depth * GAssetTextureChannels);
		memcpy(processingImageBin->m_rawData.data(), pixels, processingImageBin->m_rawData.size());

		if (m_bSrgb)
		{
			m_format = size_t(VK_FORMAT_R8G8B8A8_SRGB);
		}
		else
		{
			m_format = size_t(VK_FORMAT_R8G8B8A8_UNORM);
		}

		uint32_t mipWidth = m_width;
		uint32_t mipHeight = m_height;
		uint32_t mipCount = 0;
		while (true)
		{
			mipCount++;

			if (mipWidth > 1) { mipWidth >>= 1; }
			if (mipHeight > 1) { mipHeight >>= 1; }

			if (mipWidth == 1 && mipHeight == 1)
			{
				break;
			}
		}

		m_mipmapCount = mipCount;

		if (bBuildMipmap)
		{
			processingImageBin->buildMipmapDataRGBA8(this, cutOff);
		}

		// Build snapshot data from bin.
		buildSnapshotData2D(processingImageBin);


		stbi_image_free(pixels);

		return true;
	}

	bool ImageAssetHeader::initFromRaw2DHDR(const std::filesystem::path& rawPath, bool bBuildMipmap)
	{
		setCacheBinData(std::make_shared<ImageAssetBin>(rawPath.filename().string()));
		auto processingImageBin = getBinData<ImageAssetBin>();

		int32_t texWidth, texHeight, texChannels;
		float* pixels = stbi_loadf(
			rawPath.string().c_str(), 
			&texWidth, &texHeight, &texChannels, GAssetTextureChannels);

		if (!pixels)
		{
			LOG_ERROR("Fail to load image {0}.", rawPath.string());
			return false;
		}

		m_bSrgb = false; // HDR should not in srgb space, which is linear space.
		m_width = texWidth;
		m_height = texHeight;
		m_depth = 1u;

		// Copy raw data to bin asset.
		processingImageBin->m_rawData.resize(m_width * m_height * m_depth * GAssetTextureChannels * 4); // One float = 4 char 
		memcpy(processingImageBin->m_rawData.data(), (uint8_t*)pixels, processingImageBin->m_rawData.size());

		// HDR Format.
		m_format = size_t(VK_FORMAT_R32G32B32A32_SFLOAT);

		uint32_t mipWidth = m_width;
		uint32_t mipHeight = m_height;
		uint32_t mipCount = 0;

		if (!bBuildMipmap)
		{
			mipCount = 1;
		}
		else
		{
			while (true)
			{
				mipCount++;

				if (mipWidth > 1) { mipWidth >>= 1; }
				if (mipHeight > 1) { mipHeight >>= 1; }

				if (mipWidth == 1 && mipHeight == 1)
				{
					break;
				}
			}
		}

		m_mipmapCount = mipCount;

		if (bBuildMipmap)
		{
			CHECK_ENTRY();

			// TODO: HDR mipmap generate.
			// processingImageBin->buildMipmapDataRGBA8(this, cutOff);
		}

		// Build snapshot data from bin.
		buildSnapshotData2D(processingImageBin);


		stbi_image_free(pixels);

		return true;
	}

	static std::string getRuntimeUniqueImageAssetName(const std::string& in)
	{
		static size_t GRuntimeId = 0;
		GRuntimeId++;
		return "Flower_ImageAssetId:" + std::to_string(GRuntimeId) + in;
	}

	GPUImageAsset::GPUImageAsset(
		bool bPersistent,
		GPUImageAsset* fallback,
		VkFormat format,
		const std::string& name,
		uint32_t mipmapCount,
		uint32_t width,
		uint32_t height,
		uint32_t depth)
		: LRUAssetInterface(fallback, bPersistent)
	{
		CHECK(m_image == nullptr && "You must ensure image asset only init once.");

		VkImageCreateInfo info{};
		info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		info.flags = {};
		info.imageType = VK_IMAGE_TYPE_2D;
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

		m_image = VulkanImage::create(
			getRuntimeUniqueImageAssetName(name).c_str(),
			info,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	}

	GPUImageAsset::~GPUImageAsset()
	{
		if (m_bindlessIndex != ~0)
		{
			Bindless::Texture->freeBindless(m_bindlessIndex);
		}
	}

	void GPUImageAsset::prepareToUpload(RHICommandBufferBase& cmd, VkImageSubresourceRange range)
	{
		CHECK(m_bindlessIndex == ~0);
		m_image->transitionLayout(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, range);
	}

	void GPUImageAsset::finishUpload(RHICommandBufferBase& cmd, VkImageSubresourceRange range)
	{
		m_image->transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, range);
		m_bindlessIndex = Bindless::Texture->updateTextureToBindlessDescriptorSet(m_image->getView(buildBasicImageSubresource()));

		CHECK(m_bindlessIndex != ~0);
	}

	void TextureContext::init()
	{
		// 2 GB ~ 3 GB Texture
		m_lruCache = std::make_unique<LRUAssetCache<GPUImageAsset>>(2048, 2048 + 1024);
	}

	void TextureContext::release()
	{
		m_lruCache.reset();
	}

	std::shared_ptr<GPUImageAsset> TextureContext::getOrCreateLRUSnapShot(std::shared_ptr<ImageAssetHeader> asset)
	{
		const auto& snapShotUUID = asset->getSnapShotUUID();
		if (!m_lruCache->contain(snapShotUUID))
		{
			auto newTask = SnapshotAssetTextureLoadTask::build(asset);
			GpuUploader::get()->addTask(newTask);
		}

		CHECK(m_lruCache->contain(snapShotUUID));
		return m_lruCache->tryGet(snapShotUUID);
	}

	std::shared_ptr<GPUImageAsset> TextureContext::getOrCreateImage(std::shared_ptr<ImageAssetHeader> asset)
	{
		const auto& imageUUID = asset->getHeaderUUID();
		if (!m_lruCache->contain(imageUUID))
		{
			auto newTask = ImageAssetTextureLoadTask::build(asset);
			GpuUploader::get()->addTask(newTask);
		}

		CHECK(m_lruCache->contain(imageUUID));
		return m_lruCache->tryGet(imageUUID);
	}

	void RawAssetTextureLoadTask::uploadFunction(uint32_t stageBufferOffset, RHICommandBufferBase& commandBuffer, VulkanBuffer& stageBuffer)
	{
		CHECK(cacheRawData.size() <= uploadSize());

		// TODO: Sometimes it trigger race condition here when start editor. need to fix.
		stageBuffer.map();
		memcpy((void*)((char*)stageBuffer.mapped + stageBufferOffset), cacheRawData.data(), uploadSize());
		stageBuffer.unmap();

		imageAssetGPU->prepareToUpload(commandBuffer, buildBasicImageSubresource());

		VkBufferImageCopy region{};
		region.bufferOffset = stageBufferOffset;
		region.bufferRowLength = 0;
		region.bufferImageHeight = 0;
		region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		region.imageSubresource.mipLevel = 0;
		region.imageSubresource.baseArrayLayer = 0;
		region.imageSubresource.layerCount = 1;
		region.imageOffset = { 0, 0, 0 };
		region.imageExtent = imageAssetGPU->getImage().getExtent();

		vkCmdCopyBufferToImage(commandBuffer.cmd, stageBuffer, imageAssetGPU->getImage().getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

		imageAssetGPU->finishUpload(commandBuffer, buildBasicImageSubresource());
	}

	std::shared_ptr<RawAssetTextureLoadTask> RawAssetTextureLoadTask::build(
		const std::filesystem::path& path,
		const UUID& uuid,
		VkFormat format)
	{
		int32_t texWidth, texHeight, texChannels;
		stbi_uc* pixels = stbi_load(path.string().c_str(), &texWidth, &texHeight, &texChannels, GAssetTextureChannels);

		if (!pixels)
		{
			LOG_ERROR("Fail to load image {0}.", path.string());
			return nullptr;
		}

		CHECK(!TextureManager::get()->isAssetExist(uuid) && "Persistent asset has exist, don't register repeatly.");
		auto newAsset = std::shared_ptr<GPUImageAsset>(new GPUImageAsset(
			true, nullptr,
			format,
			path.stem().string(), 1, texWidth, texHeight, 1));
		TextureManager::get()->insertGPUAsset(uuid, newAsset);

		// Create new task.
		std::shared_ptr<RawAssetTextureLoadTask> newTask = std::make_shared<RawAssetTextureLoadTask>();
		newTask->imageAssetGPU = newAsset;

		// Prepare upload data.
		newTask->cacheRawData.resize(texWidth * texHeight * 1 * GAssetTextureChannels);
		memcpy(newTask->cacheRawData.data(), pixels, newTask->cacheRawData.size());

		// NOTE: GPU Memory align, which make small texture size min is 512 byte. And may size no equal.
		//       But at least one thing is guarantee is that cache data size must less than upload Size.
		CHECK(newTask->cacheRawData.size() <= newTask->uploadSize());

		stbi_image_free(pixels);
		return newTask;
	}

	std::shared_ptr<RawAssetTextureLoadTask> RawAssetTextureLoadTask::buildFlatTexture(
		const std::string& name,
		const UUID& uuid,
		const glm::uvec4& color,
		const glm::uvec3& size,
		VkFormat format)
	{
		CHECK(!TextureManager::get()->isAssetExist(uuid) && "Persistent asset has exist, don't register repeatly.");
		auto newAsset = std::shared_ptr<GPUImageAsset>(new GPUImageAsset(
			true, nullptr,
			format,
			name, 1, size.x, size.y, size.z));
		TextureManager::get()->insertGPUAsset(uuid, newAsset);

		// Create new task.
		std::shared_ptr<RawAssetTextureLoadTask> newTask = std::make_shared<RawAssetTextureLoadTask>();
		newTask->imageAssetGPU = newAsset;

		// Prepare upload data.
		newTask->cacheRawData.resize(size.x * size.y * size.z * GAssetTextureChannels);
		for (size_t i = 0; i < newTask->cacheRawData.size(); i += GAssetTextureChannels)
		{
			newTask->cacheRawData[i + 0] = uint8_t(color.x);
			newTask->cacheRawData[i + 1] = uint8_t(color.y);
			newTask->cacheRawData[i + 2] = uint8_t(color.z);
			newTask->cacheRawData[i + 3] = uint8_t(color.w);
		}
		// NOTE: GPU Memory align, which make small texture size min is 512 byte. And may size no equal.
		//       But at least one thing is guarantee is that cache data size must less than upload Size.
		CHECK(newTask->cacheRawData.size() <= newTask->uploadSize());
		return newTask;
	}

	void SnapshotAssetTextureLoadTask::uploadFunction(
		uint32_t stageBufferOffset,
		RHICommandBufferBase& commandBuffer,
		VulkanBuffer& stageBuffer)
	{
		stageBuffer.map();
		memcpy((void*)((char*)stageBuffer.mapped + stageBufferOffset), cacheHeader->getSnapShotData().data(), uploadSize());
		stageBuffer.unmap();

		imageAssetGPU->prepareToUpload(commandBuffer, buildBasicImageSubresource());

		VkBufferImageCopy region{};
		region.bufferOffset = stageBufferOffset;
		region.bufferRowLength = 0;
		region.bufferImageHeight = 0;
		region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		region.imageSubresource.mipLevel = 0;
		region.imageSubresource.baseArrayLayer = 0;
		region.imageSubresource.layerCount = 1;
		region.imageOffset = { 0, 0, 0 };
		region.imageExtent = imageAssetGPU->getImage().getExtent();

		vkCmdCopyBufferToImage(commandBuffer.cmd, stageBuffer, imageAssetGPU->getImage().getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

		imageAssetGPU->finishUpload(commandBuffer, buildBasicImageSubresource());
	}

	std::shared_ptr<SnapshotAssetTextureLoadTask> SnapshotAssetTextureLoadTask::build(std::shared_ptr<ImageAssetHeader> inHeader)
	{
		auto* fallbackWhite = TextureManager::get()->getImage(EngineTextures::GWhiteTextureUUID).get();
		CHECK(fallbackWhite && "Fallback texture must be valid, you forget init engine texture before init.");

		std::shared_ptr<GPUImageAsset> newAsset = std::shared_ptr<GPUImageAsset>(new GPUImageAsset(
			false,
			fallbackWhite,
			inHeader->getFormat(),
			inHeader->getName(),
			1,
			inHeader->getSnapShotWidth(),
			inHeader->getSnapShotHeight(),
			1
		));

		// Register on LRU cache.
		TextureManager::get()->insertGPUAsset(inHeader->getSnapShotUUID(), newAsset);

		auto newTask = std::make_shared<SnapshotAssetTextureLoadTask>(inHeader);
		newTask->imageAssetGPU = newAsset;
		return newTask;
	}


	void ImageAssetTextureLoadTask::uploadFunction(
		uint32_t stageBufferOffset,
		RHICommandBufferBase& commandBuffer,
		VulkanBuffer& stageBuffer)
	{
		auto texBin = std::dynamic_pointer_cast<ImageAssetBin>(cacheHeader->loadBinData());
		CHECK(texBin != nullptr);

		VkImageSubresourceRange rangeAllMips = buildBasicImageSubresource();
		rangeAllMips.levelCount = cacheHeader->getMipmapCount();

		imageAssetGPU->prepareToUpload(commandBuffer, rangeAllMips);

		stageBuffer.map();
		uint32_t bufferOffset = stageBufferOffset;
		uint32_t bufferSize = 0;

		VkBufferImageCopy region{};
		region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		region.imageSubresource.baseArrayLayer = 0;
		region.imageSubresource.layerCount = 1;
		region.imageOffset = { 0, 0, 0 };
		region.bufferOffset = 0;
		region.bufferRowLength = 0;
		region.bufferImageHeight = 0;

		std::vector<VkBufferImageCopy> copyRegions{};

		if (texBin->getMipmapDatas().empty())
		{
			// No mipmap, load from src.
			const auto& srcpDatas = texBin->getRawDatas();

			const uint32_t currentMipSize = (uint32_t)srcpDatas.size();

			uint32_t mipWidth = cacheHeader->getWidth();
			uint32_t mipHeight = cacheHeader->getHeight();

			memcpy((void*)((char*)stageBuffer.mapped + bufferOffset), srcpDatas.data(), currentMipSize);

			region.bufferOffset = bufferOffset;
			region.imageSubresource.mipLevel = 0;
			region.imageExtent = { mipWidth, mipHeight, 1 };

			copyRegions.push_back(region);

			bufferOffset += currentMipSize;
			bufferSize += currentMipSize;
		}
		else
		{
			const auto& mipmapDatas = texBin->getMipmapDatas();
			for (uint32_t level = 0; level < cacheHeader->getMipmapCount(); level++)
			{
				const auto& currentMip = mipmapDatas.at(level);
				const uint32_t currentMipSize = (uint32_t)currentMip.size();

				uint32_t mipWidth = std::max<uint32_t>(cacheHeader->getWidth() >> level, 1);
				uint32_t mipHeight = std::max<uint32_t>(cacheHeader->getHeight() >> level, 1);

				memcpy((void*)((char*)stageBuffer.mapped + bufferOffset), currentMip.data(), currentMipSize);

				region.bufferOffset = bufferOffset;
				region.imageSubresource.mipLevel = level;
				region.imageExtent = { mipWidth, mipHeight, 1 };

				copyRegions.push_back(region);

				bufferOffset += currentMipSize;
				bufferSize += currentMipSize;
			}
		}
		

		CHECK(uploadSize() >= bufferSize);

		vkCmdCopyBufferToImage(commandBuffer.cmd, stageBuffer, imageAssetGPU->getImage().getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, (uint32_t)copyRegions.size(), copyRegions.data());

		stageBuffer.unmap();

		imageAssetGPU->finishUpload(commandBuffer, rangeAllMips);
	}

	std::shared_ptr<ImageAssetTextureLoadTask> ImageAssetTextureLoadTask::build(
		std::shared_ptr<ImageAssetHeader> inHeader)
	{
		auto* fallbackWhite = TextureManager::get()->getImage(EngineTextures::GWhiteTextureUUID).get();
		CHECK(fallbackWhite && "Fallback texture must be valid, you forget init engine texture before init.");

		std::shared_ptr<GPUImageAsset> newAsset = std::shared_ptr<GPUImageAsset>(new GPUImageAsset(
			false,
			fallbackWhite,
			inHeader->getFormat(),
			inHeader->getName(),
			inHeader->getMipmapCount(),
			inHeader->getWidth(),
			inHeader->getHeight(),
			1
		));

		// Register on LRU cache.
		TextureManager::get()->insertGPUAsset(inHeader->getHeaderUUID(), newAsset);

		auto newTask = std::make_shared<ImageAssetTextureLoadTask>(inHeader);
		newTask->imageAssetGPU = newAsset;
		return newTask;
	}


}