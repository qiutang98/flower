#include "builtin_resources.h"

#include <utils/utils.h>
#include <stb/stb_image.h>

using namespace engine;

EditorBuiltinResource::EditorBuiltinResource()
{
    auto flushUploadImage = [&](const char* path)
    {
        int32_t texWidth, texHeight, texChannels;
        stbi_uc* pixels = stbi_load(path, &texWidth, &texHeight, &texChannels, 4);
        ASSERT(pixels, "Load builtin folder image fail, check your install path.");

        auto newImage = std::make_unique<VulkanImage>(
            getContext()->getVMAImage(),
            path,
            buildBasicUploadImageCreateInfo(texWidth, texHeight));
        auto stageBuffer = std::make_unique<VulkanBuffer>(
            getContext()->getVMAFrequencyBuffer(),
            "StageUploadBuffer",
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
            texWidth * texHeight * 4,
            pixels);

        getContext()->executeImmediatelyMajorGraphics([&](VkCommandBuffer cmd)
        {
            newImage->transitionLayout(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, buildBasicImageSubresource());

            VkBufferImageCopy region{};
            region.bufferOffset = 0;
            region.bufferRowLength = 0;
            region.bufferImageHeight = 0;
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.mipLevel = 0;
            region.imageSubresource.baseArrayLayer = 0;
            region.imageSubresource.layerCount = 1;
            region.imageOffset = { 0, 0, 0 };
            region.imageExtent = newImage->getExtent();
            vkCmdCopyBufferToImage(cmd, stageBuffer->getVkBuffer(), newImage->getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

            newImage->transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        });
        stbi_image_free(pixels);

        return std::move(newImage);
    };


    folderImage = flushUploadImage("image/folder.png");
    fileImage   = flushUploadImage("image/file.png");
    pawnImage = flushUploadImage("image/pawn.png");

    sunImage = flushUploadImage("image/sun.png");
    postImage = flushUploadImage("image/post.png");
}
