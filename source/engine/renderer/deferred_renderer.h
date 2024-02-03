#pragma once
#include "render_functions.h"
#include "fsr2_context.h"

namespace engine
{
	struct SSSRResource
	{
		PoolImageSharedRef rt_ssrPrevRadiance = nullptr;
		PoolImageSharedRef rt_ssrPrevVariance = nullptr;
		PoolImageSharedRef rt_ssrPrevRoughness = nullptr; // history roughness, from ssr.
		PoolImageSharedRef rt_ssrPrevSampleCount = nullptr; // 

		PoolImageSharedRef rt_ssrRadiance = nullptr;
		PoolImageSharedRef rt_ssrVariance = nullptr;
		PoolImageSharedRef rt_ssrRoughness = nullptr; //
		PoolImageSharedRef rt_ssrSampleCount = nullptr; // 

		PoolImageSharedRef rt_ssrReproject = nullptr; // 
		PoolImageSharedRef rt_ssrAverageRadiance = nullptr; // 

		std::vector<VkDescriptorSet> sets = {};
	};

	struct RenderHistoryInfo
	{
		PoolImageSharedRef averageLum = nullptr;
		PoolImageSharedRef historyUpscale = nullptr;

		PoolImageSharedRef prevDepth    = nullptr;
		PoolImageSharedRef prevGBufferB = nullptr;
		PoolImageSharedRef prevNormalVertex = nullptr;
		PoolImageSharedRef prevHdrBeforeAA = nullptr;
		PoolImageSharedRef prevGBufferID = nullptr;

		PoolImageSharedRef ssgiHistory = nullptr;
		PoolImageSharedRef ssgiMomentHistory = nullptr;
		SSSRResource sssrResources = {};

		PoolImageSharedRef cloudReconstruction = nullptr;
		PoolImageSharedRef cloudReconstructionDepth = nullptr;

		PoolImageSharedRef cloudShadowDepthHistory = nullptr;
		PoolImageSharedRef cloudShadowDepthLowBlurHistory = nullptr;

		PoolImageSharedRef volumetricFogScatterIntensity = nullptr;
		PoolImageSharedRef prevHZBFurthest = nullptr;
		PoolImageSharedRef prevHZBClosest = nullptr;

		PoolImageSharedRef gtaoHistory = nullptr;
	};


	// Deferred realtime renderer, used for viewport editor.
	class DeferredRenderer : NonCopyable
	{
	public:
		static constexpr auto kMinRenderDim = 64U;
		static constexpr auto kMaxRenderDim = 4096U;

		class DimensionConfig
		{
		public:
			DimensionConfig();

			uint32_t getRenderWidth() const { return m_renderDim.x; }
			uint32_t getRenderHeight() const { return m_renderDim.y; }

			uint32_t getPostWidth()  const { return m_postDim.x; }
			uint32_t getPostHeight() const { return m_postDim.y; }

			uint32_t getOutputWidth()  const { return m_outputDim.x; }
			uint32_t getOutputHeight() const { return m_outputDim.y; }

			bool updateDimension(
				uint32_t outputWidth,
				uint32_t outputHeight,
				float renderScaleToPost,
				float postScaleToOutput);

			auto operator<=>(const DimensionConfig&) const = default;

		private:
			// Dimension which render depth, gbuffer, lighting and pre-post effect.
			math::uvec2 m_renderDim;

			// Dimension which render Upscale, post-effect.
			math::uvec2 m_postDim;

			// Dimension which do final upscale to the viewport.
			math::uvec2 m_outputDim;


			float m_renderScaleToPost;
		};

	public:
		void adaptiveExposure(
			VkCommandBuffer cmd,
			GBufferTextures* inGBuffers,
			RenderScene* scene,
			BufferParameterHandle perFrameGPU,
			const RuntimeModuleTickData& tickData);

		void renderGrid(
			VkCommandBuffer cmd,
			GBufferTextures* inGBuffers,
			BufferParameterHandle perFrameGPU);

		void getPickPixelObject(
			VkCommandBuffer cmd, 
			GBufferTextures* inGBuffers,
			RenderScene* scene,
			BufferParameterHandle perFrameGPU);

		void renderSelectionOutline(
			VkCommandBuffer cmd,
			GBufferTextures* inGBuffers,
			BufferParameterHandle perFrameGPU,
			RenderScene* scene);

		void renderSSSR(
			VkCommandBuffer cmd,
			GBufferTextures* inGBuffers,
			RenderScene* scene,
			BufferParameterHandle perFrameGPU,
			PoolImageSharedRef inHiz,
			PoolImageSharedRef inSSAOBentNormal,
			const SkyLightRenderContext& inSky,
			ReflectionProbeContext& reflectionProbeContext);

		void postprocessing(
			VkCommandBuffer cmd,
			GBufferTextures* inGBuffers,
			BufferParameterHandle perFrameGPU,
			RenderScene* scene,
			PoolImageSharedRef bloomTex,
			PoolImageSharedRef debugTex);

		void temporalAntiAliasUpscale(
			VkCommandBuffer cmd,
			GBufferTextures* inGBuffers,
			BufferParameterHandle perFrameGPU,
			RenderScene* scene);

		PoolImageSharedRef renderSSAO(
			VkCommandBuffer cmd,
			GBufferTextures* inGBuffers,
			RenderScene* scene,
			BufferParameterHandle perFrameGPU,
			PoolImageSharedRef inHiz);

		PoolImageSharedRef renderSSGI( 
			VkCommandBuffer cmd,
			GBufferTextures* inGBuffers,
			RenderScene* scene,
			BufferParameterHandle perFrameGPU,
			PoolImageSharedRef inHiz,
			const SkyLightRenderContext& inSky,
			ReflectionProbeContext& reflectionProbeContext);

		void renderDebugLine(
			VkCommandBuffer cmd,
			GBufferTextures* inGBuffers,
			RenderScene* scene,
			BufferParameterHandle perFrameGPU);

		void renderVolumetricCloud(
			VkCommandBuffer cmd,
			GBufferTextures* inGBuffers,
			RenderScene* scene,
			BufferParameterHandle perFrameGPU,
			AtmosphereTextures& inAtmosphere,
			const PerFrameData& perframe,
			const SkyLightRenderContext& skyContext,
			const SDSMInfos& sunSDSMInfos);

		void renderVolumetricFog(
			VkCommandBuffer cmd,
			GBufferTextures* inGBuffers,
			RenderScene* scene,
			BufferParameterHandle perFrameGPU,
			AtmosphereTextures& inAtmosphere,
			const PerFrameData& perframe,
			const SkyLightRenderContext& skyContext,
			const SDSMInfos& sunSDSMInfos);

		void renderVolumetricCloudShadowDepth(
			VkCommandBuffer cmd,
			GBufferTextures* inGBuffers,
			RenderScene* scene,
			BufferParameterHandle perFrameGPU,
			AtmosphereTextures& inAtmosphere,
			const PerFrameData& perframe,
			const SkyLightRenderContext& skyContext);

	public:
		explicit DeferredRenderer();
		virtual ~DeferredRenderer();

		void tick(
			const RuntimeModuleTickData& tickData, 
			VkCommandBuffer graphicsCmd,
			CameraInterface* camera);

		PoolImageSharedRef getOutput();
		VulkanImage& getOutputVulkanImage();

		const DimensionConfig& getDimensions() const { return m_dimensionConfig; }
		const auto& getTimingValues() { return m_timeStamps; }

		// Update dimension, return if change or not for each render/post/output dimension.
		bool updateDimension(
			uint32_t outputWidth,
			uint32_t outputHeight,
			float renderScaleToPost,
			float postScaleToOutput);

		void markCurrentFramePick(math::ivec2 pos, std::function<void(uint32_t pickCallback)>&& callback);

		FSR2Context* getFSR2();

	protected:
		void clearHistoryResources(bool bClearOutput);
		BufferParameterHandle preparePerframe(const RuntimeModuleTickData& tickData, CameraInterface* camera);

	protected:
		// GPU timer.
		GPUTimestamps m_gpuTimer;
		std::vector<GPUTimestamps::TimeStamp> m_timeStamps;

		// Renderer tick counter.
		uint32_t m_tickCount = 0;

		// Renderer render indexing which reset to zero when equal to backbuffer count.
		uint32_t m_renderIndex = 0;

		DimensionConfig m_dimensionConfig = {};

		// Renderer final display output, same with swapchain back buffer format.
		PoolImageSharedRef m_outputImage;

		PerFrameData m_perframe = { };

		RenderHistoryInfo m_history = {};
		PickPixelContext m_pickContext = {};
		DebugLineDrawContext m_debugLine = {};

		std::unique_ptr<FSR2Context> m_fsr2 = nullptr;
		bool m_bCameraCut = true;
	};
}