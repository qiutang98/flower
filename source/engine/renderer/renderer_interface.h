#pragma once

#include <util/util.h>
#include <rhi/rhi.h>
#include <util/camera_interface.h>

#include "fsr2.h"

namespace engine
{
	// bloom prefilter cureve.
	inline glm::vec4 getBloomPrefilter(float threshold, float thresholdSoft)
	{
		float knee = threshold * thresholdSoft;
		glm::vec4 prefilter{ };

		prefilter.x = threshold;
		prefilter.y = prefilter.x - knee;
		prefilter.z = 2.0f * knee;
		prefilter.w = 0.25f / (knee + 0.00001f);

		return prefilter;
	}

	constexpr size_t kMinRenderDim = 64;
	constexpr size_t kMaxRenderDim = 4096;

	struct SDSMInfos
	{
		BufferParameterHandle cascadeInfoBuffer;
		BufferParameterHandle rangeBuffer;
		PoolImageSharedRef shadowDepths;
		PoolImageSharedRef mainViewMask;


		void build(const CascadeShadowConfig* config, class RendererInterface* renderer);
	};

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
	};

	class RendererInterface : NonCopyable
	{
	public:
		const GPUPerFrameData& getFrameData() const { return m_cacheGPUPerFrameData; }
		void setCameraCut();

		float getRenderPercentage() const { return m_renderScale; }
		void setRenderPercentage(float v) { m_renderScale = v; }
	protected:
		// Renderer name.
		std::string m_name;

		VulkanContext* m_context;

		// Render camera.
		CameraInterface* m_camera;

		class Renderer* m_renderer;

		// Inner counter.
		bool m_bCameraCut = false;
		uint32_t m_tickCount = 0;
		uint32_t m_renderIndex = 0;

		// Native window require size.
		uint32_t m_nativeWidth  = kMinRenderDim;
		uint32_t m_nativeHeight = kMinRenderDim;

		// Render dim before upscaling.
		float m_renderScale = 1.0f / 1.5f;
		uint32_t m_renderWidth = kMinRenderDim;
		uint32_t m_renderHeight = kMinRenderDim;

		// Display dim after upscaling.
		float m_displayScale = 1.0f;
		uint32_t m_displayWidth = kMinRenderDim;
		uint32_t m_displayHeight = kMinRenderDim;

		// Renderer display output.
		PoolImageSharedRef m_displayOutput;
		PoolImageSharedRef m_displayDebug = nullptr; 

		// GPU timer.
		GPUTimestamps m_gpuTimer;
		std::vector<GPUTimestamps::TimeStamp> m_timeStamps;

		GPUPerFrameData m_cacheGPUPerFrameData;

		// Mouse position in render area.
		bool m_bPickInThisFrame = false;
		math::ivec2 m_pickPosCurrentFrame;
		BufferParameterHandle m_pickIdBuffer;
		std::function<void(uint32_t)> m_pickCallBack = nullptr;

		// Skylight radiance info.
		PoolImageSharedRef m_skylightRadiance = nullptr;
		PoolImageSharedRef m_skylightReflection = nullptr;
		uint32_t m_skylightUpdateFaceIndex = 0;


		PoolImageSharedRef m_gtaoHistory = nullptr;
		PoolImageSharedRef m_prevHDR = nullptr;
		PoolImageSharedRef m_prevDepth = nullptr;
		PoolImageSharedRef m_prevGBufferB = nullptr;

		PoolImageSharedRef m_cloudReconstruction = nullptr;
		PoolImageSharedRef m_cloudFogReconstruction = nullptr;
		PoolImageSharedRef m_cloudReconstructionDepth = nullptr;

		SSSRResource m_sssrRts;

		PoolImageSharedRef m_averageLum = nullptr;

	private:
		std::unique_ptr<FSR2Context> m_fsr2 = nullptr;

		void updatePerframeData(const RuntimeModuleTickData& tickData);
	protected:
		// Interface of 
		virtual void initImpl() { }
		virtual void tickImpl(const RuntimeModuleTickData& tickData, VkCommandBuffer graphicsCmd, BufferParameterHandle perFrameGPU) { }
		virtual void releaseImpl() { }
		virtual void updateRenderSizeImpl(uint32_t width, uint32_t height, float renderScale, float displayScale) {};



		FSR2Context* getFSR2();

		void renderAtmosphere(
			VkCommandBuffer cmd,
			class GBufferTextures* inGBuffers,
			class RenderScene* scene,
			BufferParameterHandle perFrameGPU,
			struct AtmosphereTextures& inout,
			const SDSMInfos* sdsmInfos,
			bool bComposite);

		// Prepass - static mesh.
		void renderStaticMeshPrepass(
			VkCommandBuffer cmd,
			class GBufferTextures* inGBuffers,
			class RenderScene* scene,
			BufferParameterHandle perFrameGPU
		);

		void renderPMXGbuffer(
			VkCommandBuffer cmd,
			class GBufferTextures* inGBuffers,
			class RenderScene* scene,
			BufferParameterHandle perFrameGPU
		);

		void renderPMXOutline(
			VkCommandBuffer cmd,
			class GBufferTextures* inGBuffers,
			class RenderScene* scene,
			BufferParameterHandle perFrameGPU
		);

		void renderPMXTranslucent(
			VkCommandBuffer cmd,
			class GBufferTextures* inGBuffers,
			class RenderScene* scene,
			BufferParameterHandle perFrameGPU);

		// Hzb - after prepass.
		void renderHzb(
			PoolImageSharedRef& outClosed,
			PoolImageSharedRef& outFurthest,
			VkCommandBuffer cmd,
			class GBufferTextures* inGBuffers,
			class RenderScene* scene,
			BufferParameterHandle perFrameGPU
		);

		PoolImageSharedRef renderGTAO(
			VkCommandBuffer cmd,
			class GBufferTextures* inGBuffers,
			class RenderScene* scene,
			BufferParameterHandle perFrameGPU,
			PoolImageSharedRef inHiz);

		BufferParameterHandle renderVolumetricCloud(
			VkCommandBuffer cmd,
			class GBufferTextures* inGBuffers,
			class RenderScene* scene,
			BufferParameterHandle perFrameGPU,
			AtmosphereTextures& inAtmosphere,
			SDSMInfos& sdsmInfo,
			PoolImageSharedRef hiz);


		void renderSSSR(
			VkCommandBuffer cmd,
			class GBufferTextures* inGBuffers,
			class RenderScene* scene,
			BufferParameterHandle perFrameGPU,
			PoolImageSharedRef inHiz,
			PoolImageSharedRef inSSAO);

		PoolImageSharedRef renderSSGI(
			VkCommandBuffer cmd,
			class GBufferTextures* inGBuffers,
			class RenderScene* scene,
			BufferParameterHandle perFrameGPU,
			PoolImageSharedRef inHiz);

		void renderSkylight(
			VkCommandBuffer cmd, 
			const struct AtmosphereTextures& inAtmosphere
		);

		void renderStaticMeshGBuffer(
			VkCommandBuffer cmd,
			class GBufferTextures* inGBuffers,
			class RenderScene* scene,
			BufferParameterHandle perFrameGPU,
			PoolImageSharedRef hzbFurthest);

		void renderTerrainGBuffer(
			VkCommandBuffer cmd,
			class GBufferTextures* inGBuffers,
			BufferParameterHandle perFrameGPU,
			class RenderScene* scene);

		void renderSDSM(
			VkCommandBuffer cmd,
			class GBufferTextures* inGBuffers,
			class RenderScene* scene,
			BufferParameterHandle perFrameGPU,
			SDSMInfos& inout);

		void deferredLighting(
			VkCommandBuffer cmd,
			class GBufferTextures* inGBuffers,
			class RenderScene* scene,
			BufferParameterHandle perFrameGPU,
			PoolImageSharedRef inSDSMMask,
			AtmosphereTextures& atmosphere,
			PoolImageSharedRef inSSAO,
			SDSMInfos& sdsmInfo);

		void adaptiveExposure(
			VkCommandBuffer cmd,
			class GBufferTextures* inGBuffers,
			class RenderScene* scene,
			BufferParameterHandle perFrameGPU,
			const RuntimeModuleTickData& tickData);

		PoolImageSharedRef renderBloom(
			VkCommandBuffer cmd,
			class GBufferTextures* inGBuffers,
			class RenderScene* scene,
			BufferParameterHandle perFrameGPU);

		void renderFSR2(
			VkCommandBuffer cmd,
			class GBufferTextures* inGBuffers,
			class RenderScene* scene,
			BufferParameterHandle perFrameGPU,
			const RuntimeModuleTickData& tickData);

		void renderTonemapper(
			VkCommandBuffer cmd,
			class GBufferTextures* inGBuffers,
			BufferParameterHandle perFrameGPU,
			class RenderScene* scene,
			PoolImageSharedRef bloomTex,
			BufferParameterHandle lens);

		void renderSelectionOutline(
			VkCommandBuffer cmd,
			class GBufferTextures* inGBuffers,
			BufferParameterHandle perFrameGPU);

		void renderGrid(
			VkCommandBuffer cmd,
			class GBufferTextures* inGBuffers,
			BufferParameterHandle perFrameGPU);

		void getPickPixelObject(
			VkCommandBuffer cmd,
			class GBufferTextures* inGBuffers);



	public:
		RendererInterface(const char* name, VulkanContext* context, CameraInterface* inCam);

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
		VulkanImage& getDisplayOrDebugOutput();
		PoolImageSharedRef getDisplayOutputRef() { return m_displayOutput; }

		// Render dimension.
		uint32_t getRenderWidth() const { return m_renderWidth; }
		uint32_t getRenderHeight() const { return m_renderHeight; }

		// Display dimension.
		uint32_t getDisplayWidth() const { return m_displayWidth; }
		uint32_t getDisplayHeight() const { return m_displayHeight; }

		// Timing stamps.
		const auto& getTimingValues() { return m_timeStamps; }

		void markCurrentFramePick(math::ivec2 pos, std::function<void(uint32_t pickCallback)>&& callback)
		{
			// Only dispatch pick when no exist pick buffer.
			if (m_pickIdBuffer == nullptr)
			{
				m_bPickInThisFrame = true;
				m_pickPosCurrentFrame = pos;
				m_pickCallBack = callback;
			}
		}
	};
}