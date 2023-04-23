#pragma once
#include "renderer_interface.h"

namespace engine
{
	class DeferredRenderer : public RendererInterface
	{
	public:
		DeferredRenderer(const char* name, VulkanContext* context, CameraInterface* inCam);

		virtual void initImpl() override;

		virtual void tickImpl(const RuntimeModuleTickData& tickData, VkCommandBuffer graphicsCmd, BufferParameterHandle perFrameGPU) override;

		virtual void updateRenderSizeImpl(uint32_t width, uint32_t height, float renderScale, float displayScale) override;
	};
}