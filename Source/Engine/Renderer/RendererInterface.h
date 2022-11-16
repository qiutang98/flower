#pragma once
#include "RendererCommon.h"
#include "RenderTexturePool.h"
#include "RendererTextures.h"
#include "BufferParameter.h"
#include "PassCollector.h"

namespace Flower
{
	class CameraInterface;

	class RendererInterface : NonCopyable
	{
	private:
		std::unique_ptr<BufferParametersRing> m_bufferParameters = nullptr;
		std::unique_ptr<PassCollector> m_passCollector = nullptr;

	protected:
		uint32_t m_tickCount = 0;
		uint32_t m_renderIndex = 0;

		std::string m_name;
		std::unique_ptr<RenderTexturePool> m_rtPool = nullptr;

		CameraInterface* m_camera;

		
		uint32_t m_nativeWidth = GMinRenderDim;
		uint32_t m_nativeHeight = GMinRenderDim;
		float m_renderScale = 1.0f;
		float m_displayScale = 1.0f;

		// Render dim before upscaling.
		uint32_t m_renderWidth = GMinRenderDim;
		uint32_t m_renderHeight = GMinRenderDim;

		// Display dim after upscaling.
		uint32_t m_displayWidth = GMinRenderDim;
		uint32_t m_displayHeight = GMinRenderDim;

		PoolImageSharedRef m_displayOutput;

		GPUTimestamps m_gpuTimer;
		std::vector<TimeStamp> m_timeStamps;

	protected:
		BufferParametersRing* getBuffers();

		virtual void initImpl()
		{

		}
		virtual void tickImpl(const RuntimeModuleTickData& tickData, VkCommandBuffer graphicsCmd)
		{

		}
		virtual void releaseImpl()
		{

		}

		virtual void updateRenderSizeImpl(uint32_t width, uint32_t height, float renderScale, float displayScale) {};

	public:
		RendererInterface(const char* name, CameraInterface* inCam)
			: m_name(name), m_camera(inCam)
		{

		}

		virtual ~RendererInterface()
		{

		}

		// Wether need resize for output.
		bool shouldResizeForOuput() const
		{
			return m_renderWidth != m_displayWidth || m_renderHeight != m_displayHeight;
		}

		void init();
		void tick(const RuntimeModuleTickData& tickData, VkCommandBuffer graphicsCmd);
		void release();
		
		void updateRenderSize(uint32_t width, uint32_t height, float renderScale, float displayScale);
		VulkanImage& getDisplayOutput();
		PoolImageSharedRef getDisplayOutputRef() { return m_displayOutput; }

		uint32_t getRenderWidth() const { return m_renderWidth; }
		uint32_t getRenderHeight() const { return m_renderHeight; }

		uint32_t getDisplayWidth() const { return m_displayWidth; }
		uint32_t getDisplayHeight() const { return m_displayHeight; }

		RenderTexturePool* getRTPool() const { return m_rtPool.get(); }
		PassCollector* getPasses() const { return m_passCollector.get(); }

		const std::vector<TimeStamp>& getTimingValues() { return m_timeStamps; }
	};
}