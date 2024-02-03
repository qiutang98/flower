#pragma once

#include "../utils/utils.h"
#include "../graphics/context.h"
#include "../utils/camera_interface.h"

namespace engine
{
	struct AtmosphereTextures;
	struct GBufferTextures;
	class RenderScene;
	class DeferredRenderer;

	struct DebugLineDrawContext
	{
		uint32_t maxCount;
		BufferParameterHandle verticesGPU;

		BufferParameterHandle verticesDrawCmd;
		BufferParameterHandle verticesCount;

		void beforeRecord(VkCommandBuffer cmd);
		void endRecord(VkCommandBuffer cmd);

		void reinit(VkCommandBuffer cmd);
	};

	extern void updateCloudPass();


	struct SDSMInfos
	{
		BufferParameterHandle cascadeInfoBuffer;

		std::vector<PoolImageSharedRef> shadowDepths;
		PoolImageSharedRef shadowMask;

		void build(const SkyLightInfo* sky, uint width, uint height);
	};

	struct PickPixelContext
	{
		bool bPickInThisFrame = false;
		math::ivec2 pickPosCurrentFrame;
		BufferParameterHandle pickIdBuffer = nullptr;
		std::function<void(uint32_t)> pickCallBack = nullptr;
	};

	extern bool isDebugLineEnable();

	extern void renderAtmosphere(
		VkCommandBuffer cmd,
		GBufferTextures* inGBuffers,
		RenderScene* scene,
		BufferParameterHandle perFrameGPU,
		const PerFrameData& perframe,
		AtmosphereTextures& inout,
		bool bComposite,
		GPUTimestamps* timer = nullptr);

	struct SkyLightRenderContext
	{
		uint irradianceDim         = 16;
		uint convDim               = 128;
		uint irradianceSampleCount = 128;
		uint reflectionSampleCount = 256;

		PoolImageSharedRef skylightRadiance = nullptr;
		PoolImageSharedRef skylightReflection = nullptr;
	};

	struct ReflectionProbeContext
	{
		PoolImageSharedRef probe0 = nullptr;
		vec3 probe0Position       = vec3(0.0f);
		vec3 probe0MinExtent      = vec3(-5.0f);
		vec3 probe0MaxExtent      = vec3(5.0f);
		float probe0ValidState = 0.0f;

		PoolImageSharedRef probe1 = nullptr;
		vec3 probe1Position       = vec3(0.0f);
		vec3 probe1MinExtent      = vec3(-5.0f);
		vec3 probe1MaxExtent      = vec3(5.0f);
		float probe1ValidState = 0.0f;
	};

	extern void renderSkylight(
		VkCommandBuffer cmd,
		const struct AtmosphereTextures& inAtmosphere,
		const PerFrameData& perframe,
		RenderScene* scene,
		SkyLightRenderContext& context,
		ReflectionProbeContext& reflectionProbe,
		GPUTimestamps* timer = nullptr
	);

	extern void buildCubemapReflection(
		VkCommandBuffer cmd,
		PoolImageSharedRef cube,
		PoolImageSharedRef& resultImage,
		uint demension);

	extern void renderGIDiffuse(
		VkCommandBuffer cmd,
		GBufferTextures* inGBuffers,
		RenderScene* scene,
		BufferParameterHandle perFrameGPU,
		PoolImageSharedRef bentNormalSSAO,
		PoolImageSharedRef ssgi,
		SkyLightRenderContext& skylightContext,
		GPUTimestamps* timer = nullptr);

	extern void prepareReflectionCaptureForRender(
		VkCommandBuffer cmd,
		RenderScene* scene,
		const PerFrameData& perframe,
		const RuntimeModuleTickData& tickData,
		const struct AtmosphereTextures& inAtmosphere,
		ReflectionProbeContext& result
	);

	extern void renderGIReflection(
		VkCommandBuffer cmd,
		GBufferTextures* inGBuffers,
		RenderScene* scene,
		const PerFrameData& perframe,
		BufferParameterHandle perFrameGPU,
		SkyLightRenderContext& skylightContext,
		GPUTimestamps* timer = nullptr);

	extern void renderStaticMeshGBuffer(
		VkCommandBuffer cmd,
		GBufferTextures* inGBuffers,
		RenderScene* scene,
		BufferParameterHandle perFrameGPU,
		PoolImageSharedRef hzbFurthest,
		GPUTimestamps* timer,
		DebugLineDrawContext* debugLiner);

	extern void renderStaticMeshPrepass(
		VkCommandBuffer cmd,
		GBufferTextures* inGBuffers,
		RenderScene* scene,
		BufferParameterHandle perFrameGPU,
		GPUTimestamps* timer);

	extern void renderHzb(
		PoolImageSharedRef& outClosed,
		PoolImageSharedRef& outFurthest,
		VkCommandBuffer cmd,
		GBufferTextures* inGBuffers,
		RenderScene* scene,
		BufferParameterHandle perFrameGPU,
		GPUTimestamps* timer);

	extern PoolImageSharedRef renderBloom(
		VkCommandBuffer cmd,
		GBufferTextures* inGBuffers,
		RenderScene* scene,
		BufferParameterHandle perFrameGPU,
		const PostprocessVolumeSetting& setting,
		GPUTimestamps* timer,
		PoolImageSharedRef exposureImage);

	extern void renderDirectLighting(
		VkCommandBuffer cmd,
		GBufferTextures* inGBuffers,
		RenderScene* scene,
		BufferParameterHandle perFrameGPU,
		AtmosphereTextures& atmosphere,
		SDSMInfos& sunSdsmInfos,
		SDSMInfos& moonSdsmInfos,
		PoolImageSharedRef bentNormalSSAO,
		GPUTimestamps* timer,
		PoolImageSharedRef exposure = nullptr);

	extern BufferParameterHandle sceneDepthRangePass(
		VkCommandBuffer cmd,
		GBufferTextures* inGBuffers,
		BufferParameterHandle perFrameGPU,
		RenderScene* scene,
		GPUTimestamps* timer);

	extern PoolImageSharedRef reconstructNormal(
		VkCommandBuffer cmd,
		GBufferTextures* inGBuffers,
		BufferParameterHandle perFrameGPU,
		RenderScene* scene,
		GPUTimestamps* timer);

	extern void prepareTerrainLODS(
		VkCommandBuffer cmd,
		GBufferTextures* inGBuffers,
		RenderScene* scene,
		BufferParameterHandle perFrameGPU,
		GPUTimestamps* timer);


	extern void renderTerrainGbuffer(
		VkCommandBuffer cmd,
		GBufferTextures* inGBuffers,
		RenderScene* scene,
		BufferParameterHandle perFrameGPU,
		GPUTimestamps* timer);

	extern void renderSDSM(
		VkCommandBuffer cmd,
		GBufferTextures* inGBuffers,
		const SkyLightInfo& sunSkyInfo,
		RenderScene* scene,
		BufferParameterHandle perFrameGPU,
		SDSMInfos& sunSdsmInfos,
		SDSMInfos& moonSdsmInfos,
		BufferParameterHandle sceneDepthRange,
		GPUTimestamps* timer,
		PoolImageSharedRef sunCloudShadowDepth);

	extern void renderTerrainSDSMDepth(
		VkCommandBuffer cmd,
		BufferParameterHandle perFrameGPU,
		GBufferTextures* inGBuffers,
		RenderScene* scene,
		SDSMInfos& sdsmInfo,
		uint32_t cascadeId);

	extern void renderFSR2(
		class FSR2Context* fsr2,
		VkCommandBuffer cmd,
		GBufferTextures* inGBuffers,
		RenderScene* scene,
		BufferParameterHandle perFrameGPU,
		const RuntimeModuleTickData& tickData,
		const PerFrameData& perframe,
		GPUTimestamps* timer);

	extern void applyAdaptiveExposure(
		VkCommandBuffer cmd,
		GBufferTextures* inGBuffers,
		RenderScene* scene,
		BufferParameterHandle perFrameGPU,
		const RuntimeModuleTickData& tickData,
		PoolImageSharedRef eyeAdapt);
}