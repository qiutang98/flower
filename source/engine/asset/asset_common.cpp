#include "asset_common.h"
#include <asset/asset_system.h>

namespace engine
{
    bool isEngineMetaAsset(const std::string& extension)
    {
		return
			isAssetTextureMeta(extension)    ||
			isAssetStaticMeshMeta(extension) ||
			isAssetMaterialMeta(extension)   ||
			isAssetSceneMeta(extension)      ||
			isAssetPMXMeta(extension);
    }

	bool isAssetTextureMeta(const std::string& extension)
	{
		return extension == ".image";
	}
	bool isAssetStaticMeshMeta(const std::string& extension)
	{
		return extension == ".staticmesh";
	}
	bool isAssetMaterialMeta(const std::string& extension)
	{
		return extension == ".material";
	}
	bool isAssetSceneMeta(const std::string& extension)
	{
		return extension == ".scene";
	}
	bool isAssetPMXMeta(const std::string& extension)
	{
		return extension == ".assetpmx";
	}

	bool AssetInterface::saveAction()
	{
		if (!isDirty())
		{
			return false;
		}

		if (savePathUnvalid())
		{
			LOG_ERROR("Try save un-path asset!");
			return false;
		}

		bool bSaveResult = saveActionImpl();
		if (bSaveResult)
		{
			m_bDirty = false;
		}

		return bSaveResult;
	}

	void AssetInterface::buildSnapshot(uint32_t width, uint32_t height, const uint8_t* buffer)
    {
        m_widthSnapShot = width;
        m_heightSnapShot = height;

        m_snapshotData.resize(m_widthSnapShot * m_heightSnapShot * 4);
        memcpy(m_snapshotData.data(), buffer, m_snapshotData.size());
    }

	std::shared_ptr<GPUImageAsset> AssetInterface::getOrCreateLRUSnapShot(VulkanContext* ct)
	{
		if (!ct->getLRU()->contain(m_snapshotUUID))
		{
			auto newTask = SnapshotAssetTextureLoadTask::build(ct, shared_from_this());
			ct->getAsyncUploader().addTask(newTask);
		}

		return std::dynamic_pointer_cast<GPUImageAsset>(ct->getLRU()->tryGet(m_snapshotUUID));
	}

	bool AssetInterface::setDirty(bool bDirty)
	{
		if (m_bDirty != bDirty)
		{
			m_bDirty = bDirty;

			if (m_bDirty)
			{
				getAssetSystem()->onAssetDirty.broadcast(shared_from_this());
			}

			return true;
		}
		return false;
	}

	std::filesystem::path AssetInterface::getSavePath() const
	{
		auto path = m_assetRelativePathUtf8;

		auto savePath = getAssetSystem()->getProjectRootPath();
		auto filePath = "\\." + m_assetRelativePathUtf8;
		savePath += filePath;

		return savePath;
	}

	void SnapshotAssetTextureLoadTask::uploadFunction(
		uint32_t stageBufferOffset,
		void* bufferPtrStart,
		RHICommandBufferBase& commandBuffer,
		VulkanBuffer& stageBuffer)
	{
		CHECK(cacheAsset->existSnapshot());
		memcpy(bufferPtrStart, cacheAsset->getSnapshotData().data(), uploadSize());

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

	std::shared_ptr<SnapshotAssetTextureLoadTask> SnapshotAssetTextureLoadTask::build(
		VulkanContext* context, std::shared_ptr<AssetInterface> inAsset)
	{
		auto* fallbackWhite = context->getEngineTextureWhite().get();
		ASSERT(fallbackWhite, "Fallback texture must be valid, you forget init engine texture before init.");

		auto newAsset = std::make_shared<GPUImageAsset>(
			context,
			fallbackWhite,
			VK_FORMAT_R8G8B8A8_UNORM, // All snapshot is unorm.
			inAsset->getNameUtf8(),
			1,
			inAsset->getSnapshotWidth(),
			inAsset->getSnapshotHeight(),
			1
		);

		context->insertGPUAsset(inAsset->getSnapshotUUID(), newAsset);

		auto newTask = std::make_shared<SnapshotAssetTextureLoadTask>(inAsset);
		newTask->imageAssetGPU = newAsset;

		return newTask;
	}
}