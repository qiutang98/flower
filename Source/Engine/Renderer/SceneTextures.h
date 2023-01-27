#pragma once
#include "RendererCommon.h"
#include "RenderTexturePool.h"
#include "RendererInterface.h"

namespace Flower
{
	// 128 * 128 blue noise.
	struct GlobalBlueNoise
	{
		// 16 Spp .////////////////////////////////////

		struct BufferMisc
		{
			// The Sobol sequence buffer.
			std::shared_ptr<VulkanBuffer> sobolBuffer = nullptr;

			// The ranking tile buffer for sampling.
			std::shared_ptr<VulkanBuffer> rankingTileBuffer = nullptr;

			// The scrambling tile buffer for sampling.
			std::shared_ptr<VulkanBuffer> scramblingTileBuffer = nullptr;

			VkDescriptorSet set;
			VkDescriptorSetLayout setLayouts;

			void buildSet();

			void release()
			{
				sobolBuffer = nullptr;
				rankingTileBuffer = nullptr;
				scramblingTileBuffer = nullptr;
			}
		};

		

		// 1 Spp .////////////////////////////////////

		BufferMisc spp_1_buffer;
		BufferMisc spp_2_buffer;
		BufferMisc spp_4_buffer;
		BufferMisc spp_8_buffer;
		BufferMisc spp_16_buffer;

		void init();

		void release()
		{
			spp_1_buffer.release();
			spp_2_buffer.release();
			spp_4_buffer.release();
			spp_8_buffer.release();
			spp_16_buffer.release();
		}
	};

	class StaticTextures : NonCopyable
	{
	private:
		std::unique_ptr<RenderTexturePool> m_rtPool = nullptr;
		std::unique_ptr<PassCollector> m_passCollector = nullptr;

		// BRDF lut.
		PoolImageSharedRef m_brdfLut = nullptr;
		PoolImageSharedRef m_iblEnvCube = nullptr;
		PoolImageSharedRef m_iblIrradiance = nullptr;
		PoolImageSharedRef m_iblPrefilter = nullptr;

		PoolImageSharedRef m_cloudBasicNoise = nullptr;
		PoolImageSharedRef m_cloudWorleyNoise = nullptr;

		void initIBL(VkCommandBuffer cmd, bool bRebuildLut);
		void initCloudTexture(VkCommandBuffer cmd);

	public:
		PoolImageSharedRef getBRDFLut();

		bool isIBLReady() const;
		PoolImageSharedRef getIBLEnvCube();
		PoolImageSharedRef getIBLIrradiance();
		PoolImageSharedRef getIBLPrefilter();

		PoolImageSharedRef getCloudBasicNoise();
		PoolImageSharedRef getCloudWorleyNoise();

		PassCollector* getPasses() { return m_passCollector.get(); }

		void rebuildIBL(VkCommandBuffer cmd, bool bRebuildLut)
		{
			if (bRebuildLut)
			{
				m_brdfLut = nullptr;
			}
			
			m_iblIrradiance = nullptr;
			m_iblPrefilter = nullptr;

			initIBL(cmd, bRebuildLut);
		}

		void rebuildCloudTexture(VkCommandBuffer cmd)
		{
			m_cloudBasicNoise = nullptr;
			m_cloudWorleyNoise = nullptr;

			initCloudTexture(cmd);
		}

		GlobalBlueNoise globalBlueNoise;

		void init();
		void tick();
		void release();
	};
	using StaticTexturesManager = Singleton<StaticTextures>;

	class SceneTextures : NonCopyable
	{
	private:
		RendererInterface* m_renderer;
		RenderTexturePool* m_rtPool;

		PoolImageSharedRef m_hdrSceneColor = nullptr;

		PoolImageSharedRef m_hdrSceneColorUpscale = nullptr;
		 
		// GBuffer A: r8g8b8a8 unorm, .rgb store base color.
		PoolImageSharedRef m_gbufferA = nullptr; 

		// GBuffer B : r16g16b16a16 sfloat, .rgb store worldspace normal, .a is mesh id.
		PoolImageSharedRef m_gbufferB = nullptr; 

		// GBuffer S: r8g8b8a8 unorm, .r is metal, .g is roughness, .b is mesh ao.
		PoolImageSharedRef m_gbufferS = nullptr; 

		// GBuffer V: r16g16 sfloat.
		PoolImageSharedRef m_gbufferV = nullptr;

		/*
			In the context of FSR2, the term "reactivity" means how much influence the samples rendered for the current frame have over the production of the final upscaled image. Typically, 
			samples rendered for the current frame contribute a relatively modest amount to the result computed by FSR2; however, there are exceptions. To produce the best results for fast moving, 
			alpha-blended objects, FSR2 requires the Reproject & accumulate stage to become more reactive for such pixels. As there is no good way to determine from either color, 
			depth or motion vectors which pixels have been rendered using alpha blending, FSR2 performs best when applications explicitly mark such areas.

			Therefore, it is strongly encouraged that applications provide a reactive mask to FSR2. 
			The reactive mask guides FSR2 on where it should reduce its reliance on historical information when compositing the current pixel, 
			and instead allow the current frame's samples to contribute more to the final result. 
			The reactive mask allows the application to provide a value from [0..1] where 0 indicates that the pixel is not at all reactive (and should use the default FSR2 composition strategy), 
			and a value of 1 indicates the pixel should be fully reactive.

			While there are other applications for the reactive mask, 
			the primary application for the reactive mask is producing better results of upscaling images which include alpha-blended objects. 
			A good proxy for reactiveness is actually the alpha value used when compositing an alpha-blended object into the scene, 
			therefore, applications should write alpha to the reactive mask. 
			It should be noted that it is unlikely that a reactive value of close to 1 will ever produce good results. Therefore, we recommend clamping the maximum reactive value to around 0.9.

			If a Reactive mask is not provided to FSR2 (by setting the reactive field of FfxFsr2DispatchDescription to NULL),
			then an internally generated 1x1 texture with a cleared reactive value will be used.
		*/
		PoolImageSharedRef m_gbufferUpscaleReactive = nullptr;


		/*
			In addition to the Reactive mask, FSR2 provides for the application to denote areas of other specialist rendering which should be accounted for during the upscaling process. 
			Examples of such special rendering include areas of raytraced reflections or animated textures.

			While the Reactive mask adjusts the accumulation balance, the Transparency & composition mask adjusts the pixel locks created by FSR2. 
			A pixel with a value of 0 in the Transparency & composition mask does not perform any additional modification to the lock for that pixel. 
			Conversely, a value of 1 denotes that the lock for that pixel should be completely removed.

			If a Transparency & composition mask is not provided to FSR2 (by setting the transparencyAndComposition field of FfxFsr2DispatchDescription to NULL),
			then an internally generated 1x1 texture with a cleared transparency and composition value will be used.
		*/
		PoolImageSharedRef m_gbufferUpscaleTranslucencyAndComposition = nullptr;

		// Scene depth texutre, r32_unorm
		PoolImageSharedRef m_depthTexture = nullptr;

		// SDSM depth textures.
		PoolImageSharedRef m_sdsmDepthTextures = nullptr;

		// SDSM shadow masks
		PoolImageSharedRef m_sdsmShadowMask = nullptr;

		// Atmosphere lut.
		PoolImageSharedRef m_atmosphereTransmittance = nullptr;
		PoolImageSharedRef m_atmosphereSkyView = nullptr;
		PoolImageSharedRef m_atmosphereSkyViewCloudBottom = nullptr;
		PoolImageSharedRef m_atmosphereSkyViewCloudTop = nullptr;
		PoolImageSharedRef m_atmosphereMultiScatter = nullptr;
		PoolImageSharedRef m_atmosphereFroxelScatter = nullptr;
		PoolImageSharedRef m_atmosphereEnvCapture = nullptr;

	public:
		// Release all textures.
		void release();


		PoolImageSharedRef getHdrSceneColor();
		PoolImageSharedRef getHdrSceneColorUpscale();
		PoolImageSharedRef setHdrSceneColorUpscale(PoolImageSharedRef newI);

		PoolImageSharedRef getGbufferA();
		PoolImageSharedRef getGbufferB();
		PoolImageSharedRef getGbufferS();
		PoolImageSharedRef getGbufferV();
		PoolImageSharedRef getGbufferUpscaleReactive();
		PoolImageSharedRef getGbufferUpscaleTranslucencyAndComposition();
		PoolImageSharedRef getDepth();

		void allocateSDSMTexture(uint32_t dimXY, uint32_t cascadeCount);
		PoolImageSharedRef getSDSMDepth();
		bool isSDSMDepthExist() const;
		PoolImageSharedRef getSDSMShadowMask();

		PoolImageSharedRef getAtmosphereTransmittance();
		PoolImageSharedRef getAtmosphereSkyView();
		PoolImageSharedRef getAtmosphereSkyViewCloudBottom();
		PoolImageSharedRef getAtmosphereSkyViewCloudTop();
		PoolImageSharedRef getAtmosphereMultiScatter();
		PoolImageSharedRef getAtmosphereFroxelScatter();
		PoolImageSharedRef getAtmosphereEnvCapture();

	public:
		explicit SceneTextures(RendererInterface* in);
	};
}