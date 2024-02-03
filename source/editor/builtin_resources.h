#pragma once

#include <engine.h>
#include <graphics/context.h>

struct EditorBuiltinResource : engine::NonCopyable
{
	explicit EditorBuiltinResource();

	std::unique_ptr<engine::VulkanImage> folderImage;
	std::unique_ptr<engine::VulkanImage> fileImage;
	std::unique_ptr<engine::VulkanImage> pawnImage;
	std::unique_ptr<engine::VulkanImage> sunImage;
	std::unique_ptr<engine::VulkanImage> postImage;
};