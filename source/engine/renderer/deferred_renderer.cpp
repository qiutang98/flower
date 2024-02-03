#include "deferred_renderer.h"
#include "renderer.h"
#include "scene_textures.h"
#include "render_scene.h"
#include "../scene/component/sky_component.h"
#include "../scene/component/postprocess_component.h"
#include "../scene/scene_node.h"
#include "fsr2_context.h"

namespace engine
{
	static inline int32_t getJitterPhaseCount(int32_t renderWidth, int32_t displayWidth)
	{
		const float basePhaseCount = 8.0f;
		const int32_t jitterPhaseCount = int32_t(basePhaseCount * pow((float(displayWidth) / renderWidth), 2.0f));
		return jitterPhaseCount;
	}

	DeferredRenderer::DeferredRenderer()
	{
		// Init gpu timer.
		m_gpuTimer.init(getContext()->getSwapchain().getBackbufferCount());
	}

	DeferredRenderer::~DeferredRenderer()
	{
		m_fsr2.reset();

		getContext()->waitDeviceIdle();
		m_gpuTimer.release();
	}

	void DeferredRenderer::tick(
		const RuntimeModuleTickData& tickData, 
		VkCommandBuffer graphicsCmd,
		CameraInterface* camera)
	{

		m_gpuTimer.onBeginFrame(graphicsCmd, &m_timeStamps);
		{
			auto perFrameGPU = preparePerframe(tickData, camera);

			// Allocated gbuffer data.
			auto gbuffer = GBufferTextures::build(
				m_dimensionConfig.getRenderWidth(),
				m_dimensionConfig.getRenderHeight(),
				m_dimensionConfig.getPostWidth(),
				m_dimensionConfig.getPostHeight());

			gbuffer.clearValue(graphicsCmd);

			{
				m_debugLine.reinit(graphicsCmd);
			}

			renderStaticMeshPrepass(
				graphicsCmd, 
				&gbuffer, 
				getRenderer()->getScene(), 
				perFrameGPU,
				&m_gpuTimer);

			AtmosphereTextures atmosphereTextures{ };
			renderAtmosphere(
				graphicsCmd, 
				&gbuffer, 
				getRenderer()->getScene(), 
				perFrameGPU, 
				m_perframe,
				atmosphereTextures, 
				false,
				&m_gpuTimer);


			ReflectionProbeContext probeBlendContext;
			prepareReflectionCaptureForRender(
				graphicsCmd, 
				getRenderer()->getScene(),
				m_perframe, 
				tickData, 
				atmosphereTextures,
				probeBlendContext);

			SkyLightRenderContext skylightContext = {};
			renderSkylight(
				graphicsCmd, 
				atmosphereTextures, 
				m_perframe,
				getRenderer()->getScene(),
				skylightContext,
				probeBlendContext,
				&m_gpuTimer);

			renderVolumetricCloudShadowDepth(graphicsCmd, &gbuffer, getRenderer()->getScene(), perFrameGPU, atmosphereTextures, m_perframe, skylightContext);

			prepareTerrainLODS(
				graphicsCmd,
				&gbuffer,
				getRenderer()->getScene(),
				perFrameGPU,
				&m_gpuTimer);

			renderTerrainGbuffer(
				graphicsCmd,
				&gbuffer,
				getRenderer()->getScene(),
				perFrameGPU,
				&m_gpuTimer);

			// Build hzb by depth.
			PoolImageSharedRef hzbClosest;
			PoolImageSharedRef hzbFurthest;
			renderHzb(
				hzbClosest, 
				hzbFurthest, 
				graphicsCmd, 
				&gbuffer, 
				getRenderer()->getScene(), 
				perFrameGPU,
				&m_gpuTimer);

			// Render static mesh Gbuffer.
			renderStaticMeshGBuffer(
				graphicsCmd, 
				&gbuffer, 
				getRenderer()->getScene(), 
				perFrameGPU, 
				hzbFurthest,
				&m_gpuTimer,
				&m_debugLine);



			gbuffer.vertexNormal = reconstructNormal(graphicsCmd, &gbuffer, perFrameGPU, getRenderer()->getScene(), &m_gpuTimer);

			auto bentnormalSSAO = renderSSAO(
				graphicsCmd, 
				&gbuffer, 
				getRenderer()->getScene(), 
				perFrameGPU, 
				hzbFurthest); // Use for low mip sample inc texel hit cache.

			auto ssgiImage = renderSSGI(
				graphicsCmd,
				&gbuffer,
				getRenderer()->getScene(),
				perFrameGPU,
				hzbClosest,
				skylightContext,
				probeBlendContext); // Use for hiz raymarching.

			auto sceneDepthRangeBuffer = sceneDepthRangePass(
				graphicsCmd, 
				&gbuffer, 
				perFrameGPU, 
				getRenderer()->getScene(),
				&m_gpuTimer);

			SDSMInfos sunSDSMInfos{ };
			SDSMInfos moonSDSMInfos{ };
			renderSDSM(
				graphicsCmd, 
				&gbuffer, 
				m_perframe.sunLightInfo,
				getRenderer()->getScene(), 
				perFrameGPU, 
				sunSDSMInfos, 
				moonSDSMInfos, 
				sceneDepthRangeBuffer,
				&m_gpuTimer,
				m_history.cloudShadowDepthHistory);

			renderDirectLighting(
				graphicsCmd, 
				&gbuffer, 
				getRenderer()->getScene(), 
				perFrameGPU, 
				atmosphereTextures, 
				sunSDSMInfos, 
				moonSDSMInfos, 
				bentnormalSSAO,
				&m_gpuTimer,
				m_history.averageLum);

			// Prefer per-pixel fog trace.
#if 0
			renderVolumetricFog(
				graphicsCmd,
				&gbuffer,
				getRenderer()->getScene(),
				perFrameGPU,
				atmosphereTextures,
				m_perframe,
				skylightContext,
				sunSDSMInfos);
#endif

			// Composite sky.
			renderAtmosphere(graphicsCmd, &gbuffer, getRenderer()->getScene(), perFrameGPU, m_perframe, atmosphereTextures, true, &m_gpuTimer);

			renderVolumetricCloud(graphicsCmd, &gbuffer, getRenderer()->getScene(), perFrameGPU, atmosphereTextures, m_perframe, skylightContext, sunSDSMInfos);

			renderGIDiffuse(
				graphicsCmd, 
				&gbuffer, getRenderer()->getScene(), 
				perFrameGPU, 
				bentnormalSSAO,
				ssgiImage,
				skylightContext,
				&m_gpuTimer);

			renderSSSR(
				graphicsCmd,
				&gbuffer,
				getRenderer()->getScene(),
				perFrameGPU,
				hzbClosest,
				bentnormalSSAO,
				skylightContext,
				probeBlendContext);

			// Compute exposure.
			adaptiveExposure(graphicsCmd, &gbuffer, getRenderer()->getScene(), perFrameGPU, tickData);

			getPickPixelObject(
				graphicsCmd, 
				&gbuffer, 
				getRenderer()->getScene(), 
				perFrameGPU);

			renderSelectionOutline(graphicsCmd, &gbuffer, perFrameGPU, getRenderer()->getScene()); 




			// Update prev frame data before postprocess.
			m_history.prevDepth = gbuffer.depthTexture;
			m_history.prevGBufferB = gbuffer.gbufferB;
			m_history.prevNormalVertex = gbuffer.vertexNormal;
			m_history.prevHdrBeforeAA = gbuffer.hdrSceneColor;
			m_history.prevGBufferID = gbuffer.gbufferId;
			m_history.prevHZBFurthest = gbuffer.hzbFurthest;
			m_history.prevHZBClosest = gbuffer.hzbClosest;

			{
				gbuffer.gbufferId->getImage().transitionShaderReadOnly(graphicsCmd);
			}

			// if (m_dimensionConfig.getRenderWidth() < m_dimensionConfig.getPostWidth())
			if (true)
			{
				renderFSR2(getFSR2(), graphicsCmd, &gbuffer, getRenderer()->getScene(), perFrameGPU, tickData, m_perframe, &m_gpuTimer);
			}
			else
			{
				// Temporal anti-alias upscale.
				temporalAntiAliasUpscale(graphicsCmd, &gbuffer, perFrameGPU, getRenderer()->getScene());
			}


			// Apply exposure.
			{
				applyAdaptiveExposure(graphicsCmd, &gbuffer, getRenderer()->getScene(), perFrameGPU, tickData, m_history.averageLum);
			}

			// Bloom.
			auto bloomTex = renderBloom(
				graphicsCmd, 
				&gbuffer, 
				getRenderer()->getScene(), 
				perFrameGPU,
				m_perframe.postprocessing,
				&m_gpuTimer,
				m_history.averageLum
			);

			// Post processing.
			postprocessing(
				graphicsCmd, 
				&gbuffer, 
				perFrameGPU, 
				getRenderer()->getScene(), 
				bloomTex,
				nullptr);

			renderDebugLine(graphicsCmd, &gbuffer, getRenderer()->getScene(), perFrameGPU);

			renderGrid(graphicsCmd, &gbuffer, perFrameGPU);

			// Output must convert to shader read only in the end of rendering.
			getOutputVulkanImage().transitionShaderReadOnly(graphicsCmd);
		}
		m_gpuTimer.onEndFrame();

		// Update tick index state.
		{
			m_tickCount++;
			if (m_tickCount == ~0)
			{
				m_tickCount = 0;
			}

			m_renderIndex = m_tickCount % getContext()->getSwapchain().getBackbufferCount();
		}

		m_bCameraCut = false;
	}

	FSR2Context* DeferredRenderer::getFSR2()
	{
		if (m_fsr2 == nullptr)
		{
			m_fsr2 = std::make_unique<FSR2Context>();

			m_fsr2->onCreateWindowSizeDependentResources(
				m_dimensionConfig.getRenderWidth(),
				m_dimensionConfig.getRenderHeight(),
				m_dimensionConfig.getPostWidth(),
				m_dimensionConfig.getPostHeight());
		}

		return m_fsr2.get();
	}

	//////////////////////////////////////////////////////////////////////////////////////////
	//////////////////~Gettter
	////////////////////////////////////

	PoolImageSharedRef DeferredRenderer::getOutput()
	{
		if (!m_outputImage)
		{
			static uint32_t sOutputIndex = 0;
			const std::string name = std::format("DeferredRenderer output image {}", sOutputIndex);
			sOutputIndex ++;

			m_outputImage = getContext()->getRenderTargetPools().createPoolImage(
				name.c_str(),
				m_dimensionConfig.getOutputWidth(),
				m_dimensionConfig.getOutputHeight(),
				getContext()->getSwapchain().getImageFormat(),
				VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

			m_outputImage->getImage().transitionLayoutImmediately(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
		}

		return m_outputImage;
	}

	VulkanImage& DeferredRenderer::getOutputVulkanImage()
	{
		return getOutput()->getImage();
	}

	bool DeferredRenderer::updateDimension(
		uint32_t outputWidth,
		uint32_t outputHeight,
		float renderScaleToPost,
		float postScaleToOutput)
	{
		bool bChange = m_dimensionConfig.updateDimension(outputWidth, outputHeight, renderScaleToPost, postScaleToOutput);

		if (bChange)
		{
			clearHistoryResources(true);

			if (m_fsr2)
			{
				m_fsr2->onCreateWindowSizeDependentResources(
					m_dimensionConfig.getRenderWidth(),
					m_dimensionConfig.getRenderHeight(),
					m_dimensionConfig.getPostWidth(),
					m_dimensionConfig.getPostHeight());
			}
		}

		return bChange;
	}

	void DeferredRenderer::clearHistoryResources(bool bClearOutput)
	{
		m_history = {};
		
		m_renderIndex = 0;
		m_tickCount   = 0;

		if (bClearOutput)
		{
			m_outputImage = nullptr;
		}
	}

	BufferParameterHandle DeferredRenderer::preparePerframe(
		const RuntimeModuleTickData& tickData, 
		CameraInterface* camera)
	{
		auto* renderScene = getRenderer()->getScene();
		const auto& renderDim = getDimensions();

		// Copy prev perframe data.
		auto prevPerframe = m_perframe;

		// Now start update current frame data.
		m_perframe.appTime = {
			tickData.runTime,
			glm::sin(tickData.runTime),
			glm::cos(tickData.runTime),
			0.0f
		};

		m_perframe.frameIndex = {
			m_tickCount,
			m_tickCount % 8,
			m_tickCount % 16,
			m_tickCount % 32
		};

		m_perframe.renderWidth  = (float)renderDim.getRenderWidth();
		m_perframe.renderHeight = (float)renderDim.getRenderHeight();
		m_perframe.postWidth    = (float)renderDim.getPostWidth();
		m_perframe.postHeight   = (float)renderDim.getPostHeight();

		m_perframe.bCameraCut = m_bCameraCut;
		{
			m_perframe.bCameraCut |= (m_tickCount == 0);

			// When history upscale texture unvalid, it is camera cut.
			if (!m_history.averageLum)
			{
				m_perframe.bCameraCut = true;
			}
		}

		const bool bEnableCameraJitter = true;
		if (bEnableCameraJitter)
		{
			m_perframe.jitterPeriod = getJitterPhaseCount(renderDim.getRenderWidth(), renderDim.getPostWidth());
			
			// halton23 sequence
			m_perframe.jitterData.x = halton((m_tickCount % m_perframe.jitterPeriod) + 1, 2) - 0.5f;
			m_perframe.jitterData.y = halton((m_tickCount % m_perframe.jitterPeriod) + 1, 3) - 0.5f;

			m_perframe.bEnableJitter = true;
			m_perframe.basicTextureLODBias =
				math::log2((float)renderDim.getRenderWidth() / (float)renderDim.getPostWidth()) - 1.0f;

			m_perframe.bTAAU = renderDim.getRenderWidth() < renderDim.getPostWidth();
		}
		else
		{
			m_perframe.jitterData.x = 0.0f;
			m_perframe.jitterData.y = 0.0f;

			m_perframe.jitterPeriod  = 1;
			m_perframe.bEnableJitter = false;

			// No texture lod bias when no temporal AA jitter.
			m_perframe.basicTextureLODBias = 0.0f;
			m_perframe.bTAAU = false;
		}



		// Prepare camera info.
		camera->fillPerframe(m_perframe);

		// Update prev frame infos.
		{
			m_perframe.camInfoPrev = prevPerframe.camInfo;
			m_perframe.camViewProjPrev = prevPerframe.camViewProj;
			m_perframe.camViewPrev = prevPerframe.camView;
			m_perframe.camViewProjPrevNoJitter = prevPerframe.camViewProjNoJitter;

			m_perframe.jitterData.z = prevPerframe.jitterData.x;
			m_perframe.jitterData.w = prevPerframe.jitterData.y;
		}

		renderScene->fillPerframe(m_perframe, tickData);

		// Post edit.
		{
			m_perframe.renderType = ERendererType_Viewport;

			// We render ssao in deferred renderer.
			m_perframe.postprocessing.ssao_enable = 1;
		}



		// Upload to gpu and get buffer.
		auto perFrameGPU = getContext()->getBufferParameters().getStaticUniform("FrameData", sizeof(m_perframe));
		perFrameGPU->updateData(m_perframe);

		return perFrameGPU;
	}

	DeferredRenderer::DimensionConfig::DimensionConfig()
		: m_renderDim({ kMinRenderDim, kMinRenderDim })
		, m_postDim  ({ kMinRenderDim, kMinRenderDim })
		, m_outputDim({ kMinRenderDim, kMinRenderDim })
	{

	}

	bool DeferredRenderer::DimensionConfig::updateDimension(
		uint32_t outputWidth,
		uint32_t outputHeight,
		float renderScaleToPost,
		float postScaleToOutput)
	{
		CHECK(renderScaleToPost > 0.0 && postScaleToOutput > 0.0);

		auto makeDimSafe = [](math::uvec2& in)
		{
			in = math::clamp(in, { kMinRenderDim, kMinRenderDim }, { kMaxRenderDim, kMaxRenderDim });
		};

		DimensionConfig config { };
		config.m_outputDim = { outputWidth, outputHeight };
		config.m_postDim   = math::ceil(math::vec2(config.m_outputDim) / postScaleToOutput);
		config.m_renderDim = math::ceil(math::vec2(config.m_postDim)   / renderScaleToPost);

		makeDimSafe(config.m_outputDim);
		makeDimSafe(config.m_postDim);
		makeDimSafe(config.m_renderDim);

		bool bChange = (config != *this);
		if (bChange)
		{
			*this = config;
		}
		
		return bChange;
	}



}