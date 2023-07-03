#include "renderer_interface.h"
#include "renderer.h"
#include "render_scene.h"

#include <fsr2/src/ffx-fsr2-api/ffx_fsr2.h>

namespace engine
{

	AutoCVarInt32 cVarTAAEnable("r.taa.enable", "enable taa or not.", "engne", 1);
	RendererInterface::RendererInterface(const char* name, VulkanContext* context, CameraInterface* inCam)
		: m_name(name), m_context(context), m_camera(inCam)
	{

		m_renderer = Framework::get()->getEngine().getRuntimeModule<Renderer>();
	}

	void RendererInterface::init()
	{
		m_gpuTimer.init(m_context, m_context->getSwapchain().getBackbufferCount());

		initImpl();
	}

	void RendererInterface::setCameraCut()
	{
		m_tickCount = 0;
	}

	void RendererInterface::updatePerframeData(const RuntimeModuleTickData& tickData)
	{
		GPUPerFrameData perframe{ };

		perframe.appTime = {
			tickData.runTime,
			glm::sin(tickData.runTime),
			glm::cos(tickData.runTime),
			0.0f
		};

		perframe.frameIndex = {
			m_tickCount,
			m_tickCount % 8,
			m_tickCount % 16,
			m_tickCount % 32
		};

		// Update prev-frame jitter data.
		perframe.jitterData.z = m_cacheGPUPerFrameData.jitterData.x;
		perframe.jitterData.w = m_cacheGPUPerFrameData.jitterData.y;

		if(cVarTAAEnable.get() != 0)
		{
			const int32_t jitterPhaseCount = ffxFsr2GetJitterPhaseCount(m_renderWidth, m_displayWidth);
			ffxFsr2GetJitterOffset(&perframe.jitterData.x, &perframe.jitterData.y, m_tickCount, jitterPhaseCount);
			perframe.jitterPeriod = jitterPhaseCount;
		}
		else
		{
			perframe.jitterData.x = 0.0f;
			perframe.jitterData.y = 0.0f;
			perframe.jitterData.z = 0.0f;
			perframe.jitterData.w = 0.0f;

			perframe.jitterPeriod = 1;
		}

		perframe.sky = m_renderer->getScene()->getSkyGPU();
		perframe.basicTextureLODBias = math::log2((float)m_renderWidth / (float)m_displayWidth) - 1.0f;
		m_camera->fillPerframe(perframe);

		if (m_renderer->getScene()->isMMDCameraExist() && Framework::get()->getEngine().getGameRuningState())
		{
			m_renderer->getScene()->fillMMDCameraInfo(perframe, (float)m_renderWidth, (float)m_renderHeight);
		}

		// Update prev cam infos.
		{
			perframe.camInfoPrev = m_cacheGPUPerFrameData.camInfo;
			perframe.camViewProjPrev = m_cacheGPUPerFrameData.camViewProj;
			perframe.camViewProjPrevNoJitter = m_cacheGPUPerFrameData.camViewProjNoJitter;
		}

		perframe.renderWidth = m_renderWidth;
		perframe.renderHeight = m_renderHeight;
		perframe.displayWidth = m_displayWidth;
		perframe.displayHeight = m_displayHeight;

		perframe.camInvertView = math::inverse(perframe.camView);
		perframe.camViewProjNoJitter = perframe.camProjNoJitter * perframe.camView;
		perframe.camInvertProjNoJitter = math::inverse(perframe.camProjNoJitter);
		perframe.camInvertViewProjNoJitter = math::inverse(perframe.camViewProjNoJitter);
		{
			glm::mat4 curJitterMatrix = glm::mat4(1.0f);

			curJitterMatrix[3][0] +=  2.0f * perframe.jitterData.x / (float)m_renderWidth;
			curJitterMatrix[3][1] += -2.0f * perframe.jitterData.y / (float)m_renderHeight;

			perframe.camViewProj = curJitterMatrix * perframe.camViewProjNoJitter;
			perframe.camProj = curJitterMatrix * perframe.camProjNoJitter;

			perframe.camInvertProj = glm::inverse(perframe.camProj);
			perframe.camInvertViewProj = glm::inverse(perframe.camViewProj);
		}

		perframe.skyValid = getRenderer()->getScene()->isSkyExist();
		perframe.skySDSMValid = getRenderer()->getScene()->shouldRenderSDSM();

		const bool bCameraCut =
			m_bCameraCut || (m_tickCount == 0);

		perframe.bCameraCut = bCameraCut ? 1U : 0U;
		perframe.bAutoExposure = getRenderer()->getScene()->getPostprocessVolumeSetting().bAutoExposure ? 1U : 0U;
		perframe.fixExposure = getRenderer()->getScene()->getPostprocessVolumeSetting().fixExposure;

		m_cacheGPUPerFrameData = perframe;
	}

	void RendererInterface::tick(const RuntimeModuleTickData& tickData, VkCommandBuffer graphicsCmd)
	{
		m_gpuTimer.onBeginFrame(graphicsCmd, &m_timeStamps);
		{
			// Collect per frame data.
			updatePerframeData(tickData);

			// Get and upload gpu perframe data.
			auto frameDataGPU = m_context->getBufferParameters().getStaticUniform("FrameData", sizeof(m_cacheGPUPerFrameData));
			frameDataGPU->updateData(m_cacheGPUPerFrameData);

			// Tick actual render logic.
			m_displayDebug = nullptr;
			tickImpl(tickData, graphicsCmd, frameDataGPU);
		}
		m_gpuTimer.onEndFrame();

		// Update tick index state.
		{
			m_tickCount++;
			if (m_tickCount == ~0)
			{
				m_tickCount = 0;
			}
			m_renderIndex = m_tickCount % m_context->getSwapchain().getBackbufferCount();
		}

		m_bCameraCut = false;
	}

	void RendererInterface::release()
	{
		releaseImpl();
		m_gpuTimer.release();
		m_fsr2.reset();
	}

	void RendererInterface::updateRenderSize(uint32_t width, uint32_t height, float renderScale, float displayScale)
	{
		const float validRenderScale = math::clamp(renderScale, 1e-6f, 1.0f);
		const float validDisplayScale = math::clamp(displayScale, 1.0f, 10.0f);

		const uint32_t validWidth = math::clamp(width, (uint32_t)kMinRenderDim, (uint32_t)kMaxRenderDim);
		const uint32_t validHeight = math::clamp(height, (uint32_t)kMinRenderDim, (uint32_t)kMaxRenderDim);


		m_nativeWidth = validWidth;
		m_nativeHeight = validHeight;
		m_renderScale = validRenderScale;
		m_displayScale = validDisplayScale;

		m_gtaoHistory = nullptr;
		m_cloudReconstruction = nullptr;
		m_cloudReconstructionDepth = nullptr;
		m_cloudFogReconstruction = nullptr;

		m_renderWidth = math::clamp(uint32_t(width * validRenderScale), (uint32_t)kMinRenderDim, (uint32_t)kMaxRenderDim);
		m_renderHeight = math::clamp(uint32_t(height * validRenderScale), (uint32_t)kMinRenderDim, (uint32_t)kMaxRenderDim);
		m_displayWidth = math::clamp(uint32_t(width * validDisplayScale), (uint32_t)kMinRenderDim, (uint32_t)kMaxRenderDim);
		m_displayHeight = math::clamp(uint32_t(height * validDisplayScale), (uint32_t)kMinRenderDim, (uint32_t)kMaxRenderDim);


		// When size change, display output need recreate.
		m_displayOutput = nullptr;
		m_displayDebug = nullptr;

		// Reset tick count and render index, to clear all temporal accumulate datas.
		m_tickCount = 0;
		m_renderIndex = 0;

		updateRenderSizeImpl(width, height, renderScale, displayScale);
	}

	VulkanImage& RendererInterface::getDisplayOutput()
	{
		if (!m_displayOutput)
		{
			const std::string name = m_name + "DisplayOutput";
			m_displayOutput = m_context->getRenderTargetPools().createPoolImage(
				name.c_str(), 
				m_displayWidth, 
				m_displayHeight,
				m_context->getSwapchain().getImageFormat(), 
				VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

			m_displayOutput->getImage().transitionLayoutImmediately(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
		}

		return m_displayOutput->getImage();
	}

	VulkanImage& RendererInterface::getDisplayOrDebugOutput()
	{
		getDisplayOutput();

		if (m_displayDebug)
		{
			return m_displayDebug->getImage();
		}

		return getDisplayOutput();
	}
}