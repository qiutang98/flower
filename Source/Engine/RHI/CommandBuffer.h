#pragma once
#include "RHICommon.h"

namespace Flower
{
	// Simple collect of command buffer.
	struct RHICommandBufferBase
	{
		VkCommandBuffer cmd;
		VkCommandPool pool;
		uint32_t queueFamily;
	};
}