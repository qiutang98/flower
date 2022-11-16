#include "Pch.h"
#include "RendererInterface.h"
#include "Parameters.h"
#include "../Scene/CameraInterface.h"
#include "Renderer.h"

namespace Flower
{
	static AutoCVarCmd cVarUpdatePasses("cmd.updatePasses", "Update passes shader and pipeline info.");

	BufferParametersRing* RendererInterface::getBuffers()
	{
		if (!m_bufferParameters)
		{
			m_bufferParameters = std::make_unique<BufferParametersRing>();
		}
		return m_bufferParameters.get();
	}


	void RendererInterface::init()
	{
		m_rtPool = std::make_unique<RenderTexturePool>();
		m_passCollector = std::make_unique<PassCollector>();
		m_gpuTimer.init(uint32_t(RHI::GMaxSwapchainCount));
		initImpl();
	}

	void RendererInterface::tick(const RuntimeModuleTickData& tickData, VkCommandBuffer graphicsCmd)
	{
		CVarCmdHandle(cVarUpdatePasses, [&]()
		{
			m_passCollector->updateAllPasses();
		});

		// Tick RT pools.
		m_rtPool->tick();

		// Tick buffer parameters.
		if (m_bufferParameters)
		{
			m_bufferParameters->tick();
		}

		m_gpuTimer.onBeginFrame(graphicsCmd, &m_timeStamps);
		tickImpl(tickData, graphicsCmd);
		m_gpuTimer.onEndFrame();

		// Update tick index state.
		{
			m_tickCount++;
			if (m_tickCount == ~0)
			{
				m_tickCount = 0;
			}
			m_renderIndex = m_tickCount % GBackBufferCount;
		}
	}

	void RendererInterface::release()
	{
		releaseImpl();
		m_gpuTimer.release();
	}

	void RendererInterface::updateRenderSize(uint32_t width, uint32_t height, float renderScale, float displayScale)
	{
		const float validRenderScale = glm::clamp(renderScale, 1e-6f, 1.0f);
		const float validDisplayScale = glm::clamp(displayScale, 1.0f, 10.0f);

		const uint32_t validWidth = glm::clamp(width, (uint32_t)GMinRenderDim, (uint32_t)GMaxRenderDim);
		const uint32_t validHeight = glm::clamp(height, (uint32_t)GMinRenderDim, (uint32_t)GMaxRenderDim);

		bool bShouldReset =
			   (m_nativeWidth  != validWidth)
			|| (m_nativeHeight != validHeight)
			|| (m_renderScale  != validRenderScale)
			|| (m_displayScale != validDisplayScale);

		if (!bShouldReset)
		{
			return;
		}

		m_nativeWidth = validWidth;
		m_nativeHeight = validHeight;
		m_renderScale = validRenderScale;
		m_displayScale = validDisplayScale;

		m_renderWidth = glm::clamp(uint32_t(width * validRenderScale), (uint32_t)GMinRenderDim, (uint32_t)GMaxRenderDim);
		m_renderHeight = glm::clamp(uint32_t(height * validRenderScale), (uint32_t)GMinRenderDim, (uint32_t)GMaxRenderDim);		
		m_displayWidth = glm::clamp(uint32_t(width * validDisplayScale), (uint32_t)GMinRenderDim, (uint32_t)GMaxRenderDim);
		m_displayHeight = glm::clamp(uint32_t(height * validDisplayScale), (uint32_t)GMinRenderDim, (uint32_t)GMaxRenderDim);


		// When size change, display output need recreate.
		m_displayOutput = nullptr;

		// Reset tick count and render index, to clear all temporal accumulate datas.
		m_tickCount = 0;
		m_renderIndex = 0;

		updateRenderSizeImpl(width, height, renderScale, displayScale);
	}

	VulkanImage& RendererInterface::getDisplayOutput()
	{
		if (!m_displayOutput)
		{
			const std::string name = m_name + "::DisplayOutput";
			m_displayOutput = m_rtPool->createPoolImage(name.c_str(),
				m_displayWidth, m_displayHeight, 
				RTFormats::displayOutput(), RTUsages::displayOutput());

			m_displayOutput->getImage().transitionLayoutImmediately(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
		}

		return m_displayOutput->getImage();
	}
}