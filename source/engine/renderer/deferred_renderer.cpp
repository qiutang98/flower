#include "deferred_renderer.h"
#include "scene_textures.h"
#include "renderer.h"
#include "render_scene.h"
namespace engine
{
    DeferredRenderer::DeferredRenderer(const char* name, VulkanContext* context, CameraInterface* inCam)
        : RendererInterface(name, context, inCam)
    {

    }

	void DeferredRenderer::initImpl()
	{
		getFSR2()->onCreateWindowSizeDependentResources(
			nullptr,
			getDisplayOutput().getOrCreateView(buildBasicImageSubresource()),
			m_renderWidth,
			m_renderHeight,
			m_displayWidth,
			m_displayHeight,
			true);
	}

	void DeferredRenderer::tickImpl(const RuntimeModuleTickData& tickData, VkCommandBuffer graphicsCmd, BufferParameterHandle perFrameGPU)
	{

		if (m_skylightRadiance == nullptr)
		{
			m_skylightRadiance = getContext()->getRenderTargetPools().createPoolCubeImage(
				"SkyIBLIrradiance",
				32,  // Must can divide by 8.
				32,  // Must can divide by 8.
				VK_FORMAT_R16G16B16A16_SFLOAT,
				VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
			);
			m_skylightRadiance->getImage().transitionLayout(graphicsCmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresourceCube());
		}
		if (m_skylightReflection == nullptr)
		{
			m_skylightReflection = getContext()->getRenderTargetPools().createPoolCubeImage(
				"SkyIBLPrefilter",
				128,  // Must can divide by 8.
				128,  // Must can divide by 8.
				VK_FORMAT_R16G16B16A16_SFLOAT,
				VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
				-1    // Need mipmaps.
			);
			m_skylightReflection->getImage().transitionLayout(graphicsCmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresourceCube());
		}


		GBufferTextures gbuffers = GBufferTextures::build(this, m_context);

		// GBuffer clear.
		gbuffers.clearValue(graphicsCmd);



		renderStaticMeshPrepass(graphicsCmd, &gbuffers, m_renderer->getScene(), perFrameGPU);

		renderTerrainGBuffer(graphicsCmd, &gbuffers, perFrameGPU, m_renderer->getScene());

		renderPMXGbuffer(graphicsCmd, &gbuffers, m_renderer->getScene(), perFrameGPU);

		//
		PoolImageSharedRef hzbClosest;
		PoolImageSharedRef hzbFurthest;
		renderHzb(hzbClosest, hzbFurthest, graphicsCmd, &gbuffers, m_renderer->getScene(), perFrameGPU);

		// Render static mesh Gbuffer.
		renderStaticMeshGBuffer(graphicsCmd, &gbuffers, m_renderer->getScene(), perFrameGPU, hzbFurthest);



		auto ssaoBentNormal = renderSSGI(graphicsCmd, &gbuffers, m_renderer->getScene(), perFrameGPU, hzbClosest);

		SDSMInfos sdsmInfos{};
		renderSDSM(graphicsCmd, &gbuffers, m_renderer->getScene(), perFrameGPU, sdsmInfos);

		AtmosphereTextures atmosphereTextures{};

		if (m_renderer->getScene()->getSky() != nullptr)
		{
			renderAtmosphere(graphicsCmd, &gbuffers, m_renderer->getScene(), perFrameGPU, atmosphereTextures, &sdsmInfos, false);
			renderSkylight(graphicsCmd, atmosphereTextures);
		}

		

		deferredLighting(
			graphicsCmd, 
			&gbuffers, 
			m_renderer->getScene(), 
			perFrameGPU, 
			sdsmInfos.mainViewMask, 
			atmosphereTextures, 
			ssaoBentNormal);

		renderAtmosphere(graphicsCmd, &gbuffers, m_renderer->getScene(), perFrameGPU, atmosphereTextures, &sdsmInfos, true);
		renderVolumetricCloud(graphicsCmd, &gbuffers, m_renderer->getScene(), perFrameGPU, atmosphereTextures);



		renderSSSR(graphicsCmd, &gbuffers, m_renderer->getScene(), perFrameGPU, hzbClosest, ssaoBentNormal);

		renderPMXTranslucent(graphicsCmd, &gbuffers, m_renderer->getScene(), perFrameGPU);
		renderFSR2(graphicsCmd, &gbuffers, m_renderer->getScene(), perFrameGPU, tickData);

		adaptiveExposure(graphicsCmd, &gbuffers, m_renderer->getScene(), perFrameGPU, tickData);

		auto bloomTex = renderBloom(graphicsCmd, &gbuffers, m_renderer->getScene(), perFrameGPU);

		renderTonemapper(graphicsCmd, &gbuffers, perFrameGPU, m_renderer->getScene(), bloomTex);
		renderSelectionOutline(graphicsCmd, &gbuffers);
		renderGrid(graphicsCmd, &gbuffers, perFrameGPU);

		getPickPixelObject(graphicsCmd, &gbuffers);

		// Final output layout transition.
		getDisplayOutput().transitionLayout(graphicsCmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
		if (m_displayDebug)
		{
			m_displayDebug->getImage().transitionLayout(graphicsCmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
		}

		// Update prev frame data.
		m_prevDepth = gbuffers.depthTexture;
		m_prevGBufferB = gbuffers.gbufferB;
		m_prevHDR = gbuffers.hdrSceneColor;
	}

	void DeferredRenderer::updateRenderSizeImpl(
		uint32_t width,
		uint32_t height,
		float renderScale,
		float displayScale)
	{
		// Flush render state.
		m_context->waitDeviceIdle();

		getFSR2()->onCreateWindowSizeDependentResources(
			nullptr,
			getDisplayOutput().getOrCreateView(buildBasicImageSubresource()),
			m_renderWidth,
			m_renderHeight,
			m_displayWidth,
			m_displayHeight,
			true);

		// Reset tick state.
		m_tickCount = 0;
		m_renderIndex = 0;
	}


}
