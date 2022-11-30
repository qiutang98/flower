#pragma once
#include "../RendererCommon.h"
#include "../RenderTexturePool.h"
#include "../../Engine.h"
#include "../BufferParameter.h"
#include "../RendererInterface.h"
#include "../RendererTextures.h"
#include "../RenderSceneData.h"
#include "../SceneTextures.h"

namespace Flower
{
	class FSR2Context;

	class BlueNoiseMisc
	{
	public:
		PoolImageSharedRef spp_1_image; // 1 spp rotate by frameindex and golden radio. used for ssr.

		VkDescriptorSet getSet();
		static VkDescriptorSetLayout getSetLayout();
	private:
		VkDescriptorSet m_set = VK_NULL_HANDLE;
		static VkDescriptorSetLayout s_layout;
	};

	class DeferredRenderer : public RendererInterface
	{
	public:
		DeferredRenderer(const char* name, CameraInterface* inCam);

		virtual void initImpl() override;

		virtual void tickImpl(const RuntimeModuleTickData& tickData, VkCommandBuffer graphicsCmd) override;

		virtual void updateRenderSizeImpl(uint32_t width, uint32_t height, float renderScale, float displayScale) override;

	private:
		std::unique_ptr<FSR2Context> m_fsr2 = nullptr;
		GPUFrameData m_cacheFrameData;
		GPUViewData m_cacheViewData;

	private:
		void updateViewData(BufferParamRefPointer viewData, const RuntimeModuleTickData& tickData);
		void updateFrameData(BufferParamRefPointer frameData, const RuntimeModuleTickData& tickData);

		BlueNoiseMisc renderBlueNoiseMisc(
			VkCommandBuffer cmd,
			Renderer* renderer,
			SceneTextures* inTextures,
			RenderSceneData* scene,
			BufferParamRefPointer& viewData,
			BufferParamRefPointer& frameData,
			const RuntimeModuleTickData& tickData
		);

		void renderStaticMeshGBuffer(
			VkCommandBuffer cmd,
			Renderer* renderer,
			SceneTextures* inTextures,
			RenderSceneData* scene,
			BufferParamRefPointer& viewData,
			BufferParamRefPointer& frameData);

		// return hiz cloest.
		PoolImageSharedRef renderHiZ(
			VkCommandBuffer cmd,
			Renderer* renderer,
			SceneTextures* inTextures,
			RenderSceneData* scene,
			BufferParamRefPointer& viewData,
			BufferParamRefPointer& frameData);

		PoolImageSharedRef renderGTAO(
			VkCommandBuffer cmd,
			Renderer* renderer,
			SceneTextures* inTextures,
			RenderSceneData* scene,
			BufferParamRefPointer& viewData,
			BufferParamRefPointer& frameData,
			PoolImageSharedRef inHiz,
			BlueNoiseMisc& inBlueNoise);

		void renderVolumetricCloud(
			VkCommandBuffer cmd,
			Renderer* renderer,
			SceneTextures* inTextures,
			RenderSceneData* scene,
			BufferParamRefPointer& viewData,
			BufferParamRefPointer& frameData);

		void renderSDSM(
			VkCommandBuffer cmd,
			Renderer* renderer,
			SceneTextures* inTextures,
			RenderSceneData* scene,
			BufferParamRefPointer& viewData,
			BufferParamRefPointer& frameData
		);

		void renderBasicLighting(
			VkCommandBuffer cmd,
			Renderer* renderer,
			SceneTextures* inTextures,
			RenderSceneData* scene,
			BufferParamRefPointer& viewData,
			BufferParamRefPointer& frameData,
			PoolImageSharedRef inGTAO);

		void renderSSR(
			VkCommandBuffer cmd,
			Renderer* renderer,
			SceneTextures* inTextures,
			RenderSceneData* scene,
			BufferParamRefPointer& viewData,
			BufferParamRefPointer& frameData,
			PoolImageSharedRef inHiz,
			PoolImageSharedRef inGTAO,
			BlueNoiseMisc& inBlueNoise);

		void renderAtmosphere(
			VkCommandBuffer cmd,
			Renderer* renderer,
			SceneTextures* inTextures,
			RenderSceneData* scene,
			BufferParamRefPointer& viewData,
			BufferParamRefPointer& frameData,
			bool bComposite = false);

		void renderFSR2(
			VkCommandBuffer cmd,
			Renderer* renderer,
			SceneTextures* inTextures,
			RenderSceneData* scene,
			BufferParamRefPointer& viewData,
			BufferParamRefPointer& frameData,
			const RuntimeModuleTickData& tickData);

		PoolImageSharedRef renderBloom(
			VkCommandBuffer cmd,
			Renderer* renderer,
			SceneTextures* inTextures,
			RenderSceneData* scene,
			BufferParamRefPointer& viewData,
			BufferParamRefPointer& frameData);

		void adaptiveExposure(
			VkCommandBuffer cmd,
			Renderer* renderer,
			SceneTextures* inTextures,
			RenderSceneData* scene,
			BufferParamRefPointer& viewData,
			BufferParamRefPointer& frameData,
			const RuntimeModuleTickData& tickData);

		void renderTonemapper(
			VkCommandBuffer cmd,
			Renderer* renderer,
			SceneTextures* inTextures,
			RenderSceneData* scene,
			BufferParamRefPointer& viewData,
			BufferParamRefPointer& frameData,
			PoolImageSharedRef bloomTex,
			BlueNoiseMisc& inBlueNoise);

	private:
		PoolImageSharedRef m_averageLum = nullptr;
		PoolImageSharedRef m_gtaoHistory = nullptr;

		PoolImageSharedRef m_prevHDR = nullptr;
		PoolImageSharedRef m_prevDepth = nullptr;
		PoolImageSharedRef m_prevGBufferB = nullptr;
	};
}