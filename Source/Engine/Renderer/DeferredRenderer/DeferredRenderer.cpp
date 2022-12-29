#include "Pch.h"
#include "DeferredRenderer.h"
#include "../Parameters.h"
#include "../../Scene/CameraInterface.h"
#include "../Renderer.h"
#include "../SceneTextures.h"
#include "Pass/FSR2Pass.h"
#include "../RenderSettingContext.h"
#include "../../Scene/Component/PMXComponent.h"

namespace Flower
{
	DeferredRenderer::DeferredRenderer(const char* name, CameraInterface* inCam)
		: RendererInterface(name, inCam)
	{
		m_fsr2 = std::make_unique<FSR2Context>();
	}

	void DeferredRenderer::initImpl()
	{
		m_fsr2->onCreateWindowSizeDependentResources(
			nullptr,
			getDisplayOutput().getView(buildBasicImageSubresource()),
			m_renderWidth,
			m_renderHeight,
			m_displayWidth,
			m_displayHeight,
			true);
	}

	void DeferredRenderer::updateViewData(BufferParamRefPointer viewData, const RuntimeModuleTickData& tickData)
	{
		auto* renderer = GEngine->getRuntimeModule<Renderer>();
		auto* renderScene = renderer->getRenderScene();


		GPUViewData view{};

		// Update prev cam infos.
		{
			view.camInfoPrev = m_cacheViewData.camInfo;
			view.camViewProjPrev = m_cacheViewData.camViewProj;
			view.camViewProjPrevNoJitter = m_cacheViewData.camViewProjNoJitter;
		}

		// These setting use basic camera.
		view.exposure = m_camera->getExposure();
		view.ev100 = m_camera->getEv100();
		view.evCompensation = m_camera->exposureCompensation;
		view.cameraAtmosphereMoveScale = m_camera->atmosphereMoveScale;
		view.cameraAtmosphereOffsetHeight = m_camera->atmosphereHeightOffset;

		if (renderScene->isPMXPlayWithCamera())
		{
			auto playingPMX = renderScene->getPlayingPMXWithCamera();
			auto frame = playingPMX->getCurrentFrameCameraData(
				float(m_renderWidth),
				float(m_renderHeight),
				m_camera->getZNear(),
				m_camera->getZFar());

			view.camWorldPos = { frame.worldPos, 1.0f };
			view.camInfo =
			{
				frame.fovy,
				(float)m_renderWidth / (float)m_renderHeight,
				m_camera->getZNear(),
				m_camera->getZFar()
			};

			view.camView = frame.viewMat;
			view.camProjNoJitter = frame.projMat;


			CameraInterface::Frustum frustum = CameraInterface::Frustum::get(view.camProjNoJitter * view.camView);
			view.frustumPlanes[0] = frustum.planes[0];
			view.frustumPlanes[1] = frustum.planes[1];
			view.frustumPlanes[2] = frustum.planes[2];
			view.frustumPlanes[3] = frustum.planes[3];
			view.frustumPlanes[4] = frustum.planes[4];
			view.frustumPlanes[5] = frustum.planes[5];
		}
		else
		{
			view.camWorldPos = { m_camera->getPosition(), 1.0f };
			view.camInfo =
			{
				m_camera->getFovY(),
				m_camera->getAspect(),
				m_camera->getZNear(),
				m_camera->getZFar()
			};
			view.camView = m_camera->getViewMatrix();
			view.camProjNoJitter = m_camera->getProjectMatrix();

			CameraInterface::Frustum frustum = m_camera->getWorldFrustum();

			view.frustumPlanes[0] = frustum.planes[0];
			view.frustumPlanes[1] = frustum.planes[1];
			view.frustumPlanes[2] = frustum.planes[2];
			view.frustumPlanes[3] = frustum.planes[3];
			view.frustumPlanes[4] = frustum.planes[4];
			view.frustumPlanes[5] = frustum.planes[5];
		}

		// Build other info.
		view.camInvertView = glm::inverse(view.camView);
		view.camViewProjNoJitter = view.camProjNoJitter * view.camView;
		view.camInvertProjNoJitter = glm::inverse(view.camProjNoJitter);
		view.camInvertViewProjNoJitter = glm::inverse(view.camViewProjNoJitter);
		{
			glm::mat4 curJitterMatrix = glm::mat4(1.0f);
			curJitterMatrix[3][0] += m_cacheFrameData.jitterData.x;
			curJitterMatrix[3][1] += m_cacheFrameData.jitterData.y;

			view.camViewProj = curJitterMatrix * view.camViewProjNoJitter;
			view.camProj = curJitterMatrix * view.camProjNoJitter;

			view.camInvertProj = glm::inverse(view.camProj);
			view.camInvertViewProj = glm::inverse(view.camViewProj);
		}


		viewData->buffer.updateData(view);

		m_cacheViewData = view;
	}

	void DeferredRenderer::updateFrameData(BufferParamRefPointer frameData, const RuntimeModuleTickData& tickData)
	{
		auto* renderer = GEngine->getRuntimeModule<Renderer>();
		auto* renderScene = renderer->getRenderScene();

		GPUFrameData frame{};
		frame.appTime = {
			tickData.runTime,
			glm::sin(tickData.runTime),
			glm::cos(tickData.runTime),
			0.0f
		};

		frame.frameIndex = {
			m_tickCount,
			m_tickCount % 8,
			m_tickCount % 16,
			m_tickCount % 32
		};

		// Update prev-frame jitter data.
		frame.jitterData.z = m_cacheFrameData.jitterData.x;
		frame.jitterData.w = m_cacheFrameData.jitterData.y;
		{
			const int32_t jitterPhaseCount = ffxFsr2GetJitterPhaseCount(m_renderWidth, m_displayWidth);
			ffxFsr2GetJitterOffset(&frame.jitterData.x, &frame.jitterData.y, m_tickCount, jitterPhaseCount);

			frame.jitterData.x =  2.0f * frame.jitterData.x / (float)m_renderWidth;
			frame.jitterData.y = -2.0f * frame.jitterData.y / (float)m_renderHeight;
			frame.jitterPeriod = jitterPhaseCount;
		}

		auto& ibl = RenderSettingManager::get()->ibl;
		frame.globalIBLEnable = ibl.iblEnable() ? 1u : 0u;
		frame.globalIBLIntensity = ibl.intensity;

		// Fill importance light infos.
		const auto& importanceLights = renderScene->getImportanceLights();
		frame.directionalLightCount = importanceLights.directionalLightCount;
		if (frame.directionalLightCount > 0)
		{
			frame.directionalLight = importanceLights.directionalLights;
		}

		frame.basicTextureLODBias = m_fsr2->config.lodTextureBasicBias;

		frame.bCameraCut = m_cameraCutState;
		frame.staticMeshCount = (uint32_t)renderScene->getCollectStaticMeshes().size();
		frame.bSdsmDraw = ((frame.staticMeshCount > 0) && (frame.directionalLightCount > 0)) ? 1 : 0;


		// Fill atmosphere info.
		frame.earthAtmosphere = renderScene->getEarthAtmosphere();
		{
			// Same with glsl.
			frame.earthAtmosphere.camWorldPos =
				m_camera->getPosition() * 0.001f * m_camera->atmosphereMoveScale +
				glm::vec3{ 0.0f, m_camera->atmosphereHeightOffset + frame.earthAtmosphere.bottomRadius, 0.0f };


			// Then we need to compute cloud's project matrix.
			glm::mat4 shadowView = glm::lookAtRH(
				// Camera look from cloud top position.
				frame.earthAtmosphere.camWorldPos - glm::normalize(frame.directionalLight.direction) * frame.earthAtmosphere.cloudAreaThickness,
				// Camera look at earth center.
				frame.earthAtmosphere.camWorldPos,
				// Up direction. 
				glm::vec3{ 0.0f, 1.0f, 0.0f } // Z up ?
			);
			glm::mat4 shadowProj = glm::orthoRH_ZO(
				-frame.earthAtmosphere.cloudShadowExtent,
				frame.earthAtmosphere.cloudShadowExtent,
				-frame.earthAtmosphere.cloudShadowExtent,
				frame.earthAtmosphere.cloudShadowExtent,
				 10.0f * frame.earthAtmosphere.cloudAreaThickness, // Also reverse z for cloud shadow depth.
				-10.0f * frame.earthAtmosphere.cloudAreaThickness
			);

			frame.earthAtmosphere.cloudSpaceViewProject = shadowProj * shadowView;
			frame.earthAtmosphere.cloudSpaceViewProjectInverse = glm::inverse(frame.earthAtmosphere.cloudSpaceViewProject);
		}

		frameData->buffer.updateData(frame);

		// Update cache frame.
		m_cacheFrameData = frame;
	}

	void DeferredRenderer::tickImpl(const RuntimeModuleTickData& tickData, VkCommandBuffer graphicsCmd)
	{
		auto* renderer = GEngine->getRuntimeModule<Renderer>();
		auto* renderScene = renderer->getRenderScene();

		

		BufferParamRefPointer viewDataGPU = getBuffers()->getStaticUniform("ViewData", sizeof(GPUViewData));
		BufferParamRefPointer frameDataGPU = getBuffers()->getStaticUniform("FrameData", sizeof(GPUFrameData));

		// Update frame data before view data.
		updateFrameData(frameDataGPU, tickData);
		updateViewData(viewDataGPU, tickData);

		SceneTextures sceneTexures(this);

		{
			auto blueNoiseMisc = renderBlueNoiseMisc(graphicsCmd, renderer, &sceneTexures, renderScene, viewDataGPU, frameDataGPU, tickData);

			// GBuffer clear.
			{
				auto& hdrSceneColor = sceneTexures.getHdrSceneColor()->getImage();
				auto& gbufferA = sceneTexures.getGbufferA()->getImage();
				auto& gbufferB = sceneTexures.getGbufferB()->getImage();
				auto& gbufferS = sceneTexures.getGbufferS()->getImage();
				auto& gbufferV = sceneTexures.getGbufferV()->getImage();
				auto& gbufferComposition = sceneTexures.getGbufferUpscaleTranslucencyAndComposition()->getImage();
				auto& gbufferUpscaleMask = sceneTexures.getGbufferUpscaleReactive()->getImage();
				// Depth clear in mesh draw pass.
				auto& depthZ = sceneTexures.getDepth()->getImage();

				

				RHI::ScopePerframeMarker marker(graphicsCmd, "GBuffer Clear", { 1.0f, 1.0f, 0.0f, 1.0f });
				VkClearColorValue zeroClear = 
				{
					.uint32 = {0,0,0,0}
				};

				hdrSceneColor.transitionLayout(graphicsCmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
				gbufferA.transitionLayout(graphicsCmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
				gbufferB.transitionLayout(graphicsCmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
				gbufferS.transitionLayout(graphicsCmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
				gbufferV.transitionLayout(graphicsCmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
				gbufferComposition.transitionLayout(graphicsCmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
				gbufferUpscaleMask.transitionLayout(graphicsCmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

				depthZ.transitionLayout(graphicsCmd, VK_IMAGE_LAYOUT_GENERAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));

				auto rangeClear = buildBasicImageSubresource();
				vkCmdClearColorImage(graphicsCmd, hdrSceneColor.getImage(), VK_IMAGE_LAYOUT_GENERAL, &zeroClear, 1, &rangeClear);
				vkCmdClearColorImage(graphicsCmd, gbufferA.getImage(), VK_IMAGE_LAYOUT_GENERAL, &zeroClear, 1, &rangeClear);
				vkCmdClearColorImage(graphicsCmd, gbufferB.getImage(), VK_IMAGE_LAYOUT_GENERAL, &zeroClear, 1, &rangeClear);
				vkCmdClearColorImage(graphicsCmd, gbufferS.getImage(), VK_IMAGE_LAYOUT_GENERAL, &zeroClear, 1, &rangeClear);
				vkCmdClearColorImage(graphicsCmd, gbufferV.getImage(), VK_IMAGE_LAYOUT_GENERAL, &zeroClear, 1, &rangeClear);
				vkCmdClearColorImage(graphicsCmd, gbufferComposition.getImage(), VK_IMAGE_LAYOUT_GENERAL, &zeroClear, 1, &rangeClear);
				vkCmdClearColorImage(graphicsCmd, gbufferUpscaleMask.getImage(), VK_IMAGE_LAYOUT_GENERAL, &zeroClear, 1, &rangeClear);

				auto depthClear = VkClearDepthStencilValue{ 0.0f, 1 };
				auto rangeClearDepth = RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT);
				vkCmdClearDepthStencilImage(graphicsCmd, depthZ.getImage(), VK_IMAGE_LAYOUT_GENERAL, &depthClear, 1, &rangeClearDepth);

				hdrSceneColor.transitionLayout(graphicsCmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
				gbufferA.transitionLayout(graphicsCmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
				gbufferB.transitionLayout(graphicsCmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
				gbufferS.transitionLayout(graphicsCmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
				gbufferV.transitionLayout(graphicsCmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
				gbufferComposition.transitionLayout(graphicsCmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
				gbufferUpscaleMask.transitionLayout(graphicsCmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
				depthZ.transitionLayout(graphicsCmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, rangeClearDepth);
			}

			// Render terrain.
			renderTerrain(graphicsCmd, renderer, &sceneTexures, renderScene, viewDataGPU, frameDataGPU);

			// Render static mesh Gbuffer.
			renderStaticMeshGBuffer(graphicsCmd, renderer, &sceneTexures, renderScene, viewDataGPU, frameDataGPU);

			renderPMX(graphicsCmd, renderer, &sceneTexures, renderScene, viewDataGPU, frameDataGPU, blueNoiseMisc);

			// When set SDSM after GTAO render, the shadow will flickering, i don't know why, i check all barrier but seems normal.
			// Current make sdsm before GTAO and hiz.
			renderSDSM(graphicsCmd, renderer, &sceneTexures, renderScene, viewDataGPU, frameDataGPU, blueNoiseMisc);

			auto hizTex = renderHiZ(graphicsCmd, renderer, &sceneTexures, renderScene, viewDataGPU, frameDataGPU);
			auto GTAOTex = renderGTAO(graphicsCmd, renderer, &sceneTexures, renderScene, viewDataGPU, frameDataGPU, hizTex, blueNoiseMisc);

			// Prepare sky lut.
			renderAtmosphere(graphicsCmd, renderer, &sceneTexures, renderScene, viewDataGPU, frameDataGPU, false);

			renderBasicLighting(graphicsCmd, renderer, &sceneTexures, renderScene, viewDataGPU, frameDataGPU, GTAOTex);

			renderSSR(graphicsCmd, renderer, &sceneTexures, renderScene, viewDataGPU, frameDataGPU, hizTex, GTAOTex, blueNoiseMisc);



			// Composite sky.
			renderAtmosphere(graphicsCmd, renderer, &sceneTexures, renderScene, viewDataGPU, frameDataGPU, true);

			renderVolumetricCloud(graphicsCmd, renderer, &sceneTexures, renderScene, viewDataGPU, frameDataGPU, blueNoiseMisc);

			renderPMXTranslucent(graphicsCmd, renderer, &sceneTexures, renderScene, viewDataGPU, frameDataGPU, blueNoiseMisc);


			renderFSR2(graphicsCmd, renderer, &sceneTexures, renderScene, viewDataGPU, frameDataGPU, tickData);
			
			adaptiveExposure(graphicsCmd, renderer, &sceneTexures, renderScene, viewDataGPU, frameDataGPU, tickData);


			auto bloomTex = renderBloom(graphicsCmd, renderer, &sceneTexures, renderScene, viewDataGPU, frameDataGPU);

			
			renderTonemapper(graphicsCmd, renderer, &sceneTexures, renderScene, viewDataGPU, frameDataGPU, bloomTex, blueNoiseMisc);
		}


		// Final output layout transition.
		getDisplayOutput().transitionLayout(graphicsCmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

		m_prevDepth = sceneTexures.getDepth();
		m_prevGBufferB = sceneTexures.getGbufferB();
		m_prevHDR = sceneTexures.getHdrSceneColor();


		m_cameraCutState ++;
	}

	void DeferredRenderer::updateRenderSizeImpl(
		uint32_t width, 
		uint32_t height, 
		float renderScale, 
		float displayScale)
	{
		vkDeviceWaitIdle(RHI::Device);

		m_tickCount = 0;
		m_renderIndex = 0;

		m_prevDepth = nullptr;
		m_prevGBufferB = nullptr;
		m_prevHDR = nullptr;

		//m_cloudReconstruction = nullptr;
		//m_cloudReconstructionDepth = nullptr;
		//m_gtaoHistory = nullptr;
		//m_averageLum = nullptr;

		m_fsr2->onCreateWindowSizeDependentResources(
			nullptr, 
			getDisplayOutput().getView(buildBasicImageSubresource()),
			m_renderWidth,
			m_renderHeight,
			m_displayWidth,
			m_displayHeight,
			true);

		setCameraCut();
	}

	void DeferredRenderer::setCameraCut()
	{
		m_cameraCutState = 0;
	}
	
	VkDescriptorSet BlueNoiseMisc::getSet()
	{
		if (m_set == VK_NULL_HANDLE)
		{
			const auto layoutImage = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

			VkDescriptorImageInfo spp_1_info{ .imageView = spp_1_image->getImage().getView(buildBasicImageSubresource()), .imageLayout = layoutImage };

			RHI::get()->descriptorFactoryBegin()
				.bindImages(0, 1, &spp_1_info, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_ALL)
				.build(m_set);
		}

		return m_set;
	}

	VkDescriptorSetLayout BlueNoiseMisc::s_layout = VK_NULL_HANDLE;
	VkDescriptorSetLayout BlueNoiseMisc::getSetLayout()
	{
		if (s_layout == VK_NULL_HANDLE)
		{
			VkDescriptorSet setTemp;
			RHI::get()->descriptorFactoryBegin()
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_ALL, 0) // 1 spp
				.buildNoInfo(s_layout, setTemp);
		}

		return s_layout;
	}
}