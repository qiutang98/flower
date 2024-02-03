#pragma once
#include "../utils/utils.h"
#include "../graphics/context.h"
#include "../utils/camera_interface.h"

#include "deferred_renderer.h"

namespace engine
{
	class ReflectionCaptureRenderer : NonCopyable
	{
	public:
		PoolImageSharedRef render(
			uint32_t dimension,
			VkCommandBuffer graphicsCmd, 
			CameraInterface* camera,
			const RuntimeModuleTickData& tickData);

	};
}