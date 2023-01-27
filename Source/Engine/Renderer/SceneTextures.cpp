#include "Pch.h"
#include "SceneTextures.h"	
#include "RenderSettingContext.h"
#include "../AssetSystem/TextureManager.h"

namespace Flower
{
	SceneTextures::SceneTextures(RendererInterface* in)
		: m_renderer(in)
		, m_rtPool(in->getRTPool())
	{

	}

	void SceneTextures::release()
	{
		m_hdrSceneColor = nullptr;
		m_hdrSceneColorUpscale = nullptr;

		m_depthTexture = nullptr;

		m_gbufferA = nullptr;
		m_gbufferB = nullptr;
		m_gbufferS = nullptr;

		m_sdsmDepthTextures = nullptr;
		m_sdsmShadowMask = nullptr;

		m_gbufferUpscaleReactive = nullptr;
		m_gbufferUpscaleTranslucencyAndComposition = nullptr;

		m_atmosphereMultiScatter = nullptr;
		m_atmosphereSkyView = nullptr;
		m_atmosphereTransmittance = nullptr;
		m_atmosphereFroxelScatter = nullptr;
		m_atmosphereEnvCapture = nullptr;
	}

	PoolImageSharedRef SceneTextures::getHdrSceneColor()
	{
		if (!m_hdrSceneColor)
		{
			m_hdrSceneColor = m_rtPool->createPoolImage(
				"HdrSceneColor",
				m_renderer->getRenderWidth(),
				m_renderer->getRenderHeight(),
				RTFormats::hdrSceneColor(),
				RTUsages::hdrSceneColor()
			);
		}

		return m_hdrSceneColor;
	}

	PoolImageSharedRef SceneTextures::getHdrSceneColorUpscale()
	{
		if (!m_hdrSceneColorUpscale)
		{
			m_hdrSceneColorUpscale = m_rtPool->createPoolImage(
				"HdrSceneColorUpscale",
				m_renderer->getDisplayWidth(),
				m_renderer->getDisplayHeight(),
				RTFormats::hdrSceneColor(),
				RTUsages::hdrSceneColor()
			);
		}

		return m_hdrSceneColorUpscale;
	}

	PoolImageSharedRef SceneTextures::setHdrSceneColorUpscale(PoolImageSharedRef newI)
	{
		PoolImageSharedRef src = m_hdrSceneColorUpscale;
		m_hdrSceneColorUpscale = newI;

		return src;
	}

	PoolImageSharedRef SceneTextures::getGbufferA()
	{
		if (!m_gbufferA)
		{
			m_gbufferA = m_rtPool->createPoolImage(
				"GBufferA",
				m_renderer->getRenderWidth(),
				m_renderer->getRenderHeight(),
				RTFormats::gbufferA(),
				RTUsages::gbuffer()
			);
		}

		return m_gbufferA;
	}

	PoolImageSharedRef SceneTextures::getGbufferB()
	{
		if (!m_gbufferB)
		{
			m_gbufferB = m_rtPool->createPoolImage(
				"GBufferB",
				m_renderer->getRenderWidth(),
				m_renderer->getRenderHeight(),
				RTFormats::gbufferB(),
				RTUsages::gbuffer()
			);
		}

		return m_gbufferB;
	}

	PoolImageSharedRef SceneTextures::getGbufferS()
	{
		if (!m_gbufferS)
		{
			m_gbufferS = m_rtPool->createPoolImage(
				"GBufferS",
				m_renderer->getRenderWidth(),
				m_renderer->getRenderHeight(),
				RTFormats::gbufferS(),
				RTUsages::gbuffer()
			);
		}

		return m_gbufferS;
	}

	PoolImageSharedRef SceneTextures::getGbufferV()
	{
		if (!m_gbufferV)
		{
			m_gbufferV = m_rtPool->createPoolImage(
				"GBufferV",
				m_renderer->getRenderWidth(),
				m_renderer->getRenderHeight(),
				RTFormats::gbufferV(),
				RTUsages::gbuffer()
			);
		}

		return m_gbufferV;
	}

	PoolImageSharedRef SceneTextures::getGbufferUpscaleReactive()
	{
		if (!m_gbufferUpscaleReactive)
		{
			m_gbufferUpscaleReactive = m_rtPool->createPoolImage(
				"GBufferUpscaleReactive",
				m_renderer->getRenderWidth(),
				m_renderer->getRenderHeight(),
				RTFormats::gbufferUpscaleReactive(),
				RTUsages::gbuffer()
			);
		}

		return m_gbufferUpscaleReactive;
	}

	PoolImageSharedRef SceneTextures::getGbufferUpscaleTranslucencyAndComposition()
	{
		if (!m_gbufferUpscaleTranslucencyAndComposition)
		{
			m_gbufferUpscaleTranslucencyAndComposition = m_rtPool->createPoolImage(
				"GBufferUpscaleTranslucencyAndComposition",
				m_renderer->getRenderWidth(),
				m_renderer->getRenderHeight(),
				RTFormats::gbufferUpscaleTranslucencyAndComposition(),
				RTUsages::gbuffer()
			);
		}

		return m_gbufferUpscaleTranslucencyAndComposition;
	}

	PoolImageSharedRef SceneTextures::getDepth()
	{
		if (!m_depthTexture)
		{
			m_depthTexture = m_rtPool->createPoolImage(
				"DepthZ",
				m_renderer->getRenderWidth(),
				m_renderer->getRenderHeight(),
				RTFormats::depth(),
				RTUsages::depth()
			);
		}

		return m_depthTexture;
	}

	void SceneTextures::allocateSDSMTexture(uint32_t dimXY, uint32_t cascadeCount)
	{
		CHECK(m_sdsmDepthTextures == nullptr);

		m_sdsmDepthTextures = m_rtPool->createPoolImage(
			"SDSMDepth",
			dimXY * cascadeCount,
			dimXY,
			RTFormats::depth(),
			RTUsages::depth()
		);


	}

	PoolImageSharedRef SceneTextures::getSDSMDepth()
	{
		CHECK(m_sdsmDepthTextures);

		return m_sdsmDepthTextures;
	}

	bool SceneTextures::isSDSMDepthExist() const
	{
		return m_sdsmDepthTextures != nullptr;
	}

	PoolImageSharedRef SceneTextures::getSDSMShadowMask()
	{
		if (!m_sdsmShadowMask)
		{
			m_sdsmShadowMask = m_rtPool->createPoolImage(
				"SDSMShadowMask",
				m_renderer->getRenderWidth(),
				m_renderer->getRenderHeight(),
				RTFormats::sdsmShadowMask(),
				RTUsages::sdsmMask()
			);
		}

		return m_sdsmShadowMask;
	}

	PoolImageSharedRef SceneTextures::getAtmosphereTransmittance()
	{
		if (!m_atmosphereTransmittance)
		{
			m_atmosphereTransmittance = m_rtPool->createPoolImage(
				"AtmosphereTransmittance",
				256, // Must can divide by 8.
				64,  // Must can divide by 8.
				VK_FORMAT_R16G16B16A16_SFLOAT,
				VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
			);
		}

		return m_atmosphereTransmittance;
	}

	constexpr auto SkyViewLookupDim = 256;

	PoolImageSharedRef SceneTextures::getAtmosphereSkyView()
	{
		if (!m_atmosphereSkyView)
		{
			m_atmosphereSkyView = m_rtPool->createPoolImage(
				"AtmosphereSkyView",
				SkyViewLookupDim,
				SkyViewLookupDim,
				VK_FORMAT_R16G16B16A16_SFLOAT,
				VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
			);
		}

		return m_atmosphereSkyView;
	}

	PoolImageSharedRef SceneTextures::getAtmosphereSkyViewCloudBottom()
	{
		if (!m_atmosphereSkyViewCloudBottom)
		{
			m_atmosphereSkyViewCloudBottom = m_rtPool->createPoolImage(
				"AtmosphereSkyViewCloudBottom",
				SkyViewLookupDim,
				SkyViewLookupDim,
				VK_FORMAT_R16G16B16A16_SFLOAT,
				VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
			);
		}

		return m_atmosphereSkyViewCloudBottom;
	}
	PoolImageSharedRef SceneTextures::getAtmosphereSkyViewCloudTop()
	{
		if (!m_atmosphereSkyViewCloudTop)
		{
			m_atmosphereSkyViewCloudTop = m_rtPool->createPoolImage(
				"AtmosphereSkyViewCloudTop",
				SkyViewLookupDim,
				SkyViewLookupDim,
				VK_FORMAT_R16G16B16A16_SFLOAT,
				VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
			);
		}

		return m_atmosphereSkyViewCloudTop;
	}
	PoolImageSharedRef SceneTextures::getAtmosphereMultiScatter()
	{
		if (!m_atmosphereMultiScatter)
		{
			m_atmosphereMultiScatter = m_rtPool->createPoolImage(
				"AtmosphereMultiScatter",
				32,  // Must can divide by 8.
				32,  // Must can divide by 8.
				VK_FORMAT_R16G16B16A16_SFLOAT,
				VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
			);
		}

		return m_atmosphereMultiScatter;
	}

	PoolImageSharedRef SceneTextures::getAtmosphereFroxelScatter()
	{
		if (!m_atmosphereFroxelScatter)
		{
			m_atmosphereFroxelScatter = m_rtPool->createPoolImage(
				"AtmosphereFroxelScatter",
				32,  // Must can divide by 8.
				32,  // Must can divide by 8.
				VK_FORMAT_R16G16B16A16_SFLOAT,
				VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
				1, // Mipmap count.
				32 // Depth
			);
		}

		return m_atmosphereFroxelScatter;
	}

	PoolImageSharedRef SceneTextures::getAtmosphereEnvCapture()
	{
		if (!m_atmosphereEnvCapture)
		{
			m_atmosphereEnvCapture = m_rtPool->createPoolCubeImage(
				"AtmosphereEnvCapture",
				128,  // Must can divide by 8.
				128,  // Must can divide by 8.
				VK_FORMAT_R16G16B16A16_SFLOAT,
				VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
			);
		}

		return m_atmosphereEnvCapture;
	}

	PoolImageSharedRef StaticTextures::getBRDFLut()
	{
		CHECK(m_brdfLut && "You should call initBRDFLut() once when engine init.");
		return m_brdfLut;
	}

	bool StaticTextures::isIBLReady() const
	{
		return m_brdfLut && m_iblIrradiance && m_iblPrefilter;
	}

	PoolImageSharedRef StaticTextures::getIBLEnvCube()
	{
		CHECK(m_iblEnvCube && "You should call initBRDFLut() once when engine init.");
		return m_iblEnvCube;
	}

	PoolImageSharedRef StaticTextures::getIBLIrradiance()
	{
		CHECK(m_iblIrradiance && "You should call initBRDFLut() once when engine init.");
		return m_iblIrradiance;
	}

	PoolImageSharedRef StaticTextures::getIBLPrefilter()
	{
		CHECK(m_iblPrefilter && "You should call initBRDFLut() once when engine init.");
		return m_iblPrefilter;
	}

	PoolImageSharedRef StaticTextures::getCloudBasicNoise()
	{
		CHECK(m_cloudBasicNoise && "You should call initCloudNoise() once when engine init.");
		return m_cloudBasicNoise;
	}

	PoolImageSharedRef StaticTextures::getCloudWorleyNoise()
	{
		CHECK(m_cloudWorleyNoise && "You should call initCloudNoise() once when engine init.");
		return m_cloudWorleyNoise;
	}

	void StaticTextures::init()
	{
		m_passCollector = std::make_unique<PassCollector>();
		m_rtPool = std::make_unique<RenderTexturePool>();

		auto cmd = RHI::get()->createMajorGraphicsCommandBuffer();
		RHICheck(vkResetCommandBuffer(cmd, 0));
		VkCommandBufferBeginInfo cmdBeginInfo = RHICommandbufferBeginInfo(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
		RHICheck(vkBeginCommandBuffer(cmd, &cmdBeginInfo));
		{
			// Init basic resource for engine.
			initIBL(cmd, true);

			initCloudTexture(cmd);
		}
		RHICheck(vkEndCommandBuffer(cmd));

		VkPipelineStageFlags waitFlags = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
		RHISubmitInfo cmdSubmitInfo{};
		cmdSubmitInfo.setWaitStage(&waitFlags).setCommandBuffer(&cmd, 1);
		std::vector<VkSubmitInfo> infosRawSubmit{ cmdSubmitInfo };
		RHI::get()->submitNoFence((uint32_t)infosRawSubmit.size(), infosRawSubmit.data());

		vkDeviceWaitIdle(RHI::Device);

		globalBlueNoise.init();
	}

	void StaticTextures::tick()
	{
		m_rtPool->tick();
	}

	void StaticTextures::release()
	{
		vkDeviceWaitIdle(RHI::Device);
		m_rtPool.reset();
		m_passCollector.reset();
		globalBlueNoise.release();
	}

	struct IBLPrefilterPushConst
	{
		float perceptualRoughness;
	};

	class IBLComputePass : public PassInterface
	{
	public:
		VkPipeline lutPipeline = VK_NULL_HANDLE;
		VkPipelineLayout lutPipelineLayout = VK_NULL_HANDLE;
		VkDescriptorSetLayout lutSetLayout = VK_NULL_HANDLE;

		VkPipeline irradiancePipeline = VK_NULL_HANDLE;
		VkPipelineLayout irradiancePipelineLayout = VK_NULL_HANDLE;
		VkDescriptorSetLayout irradianceSetLayout = VK_NULL_HANDLE;

		VkPipeline sphericalMapToCubePipeline = VK_NULL_HANDLE;
		VkPipelineLayout sphericalMapToCubePipelineLayout = VK_NULL_HANDLE;
		VkDescriptorSetLayout sphericalMapToCubeSetLayout = VK_NULL_HANDLE;

		VkPipeline prefilterPipeline = VK_NULL_HANDLE;
		VkPipelineLayout prefilterPipelineLayout = VK_NULL_HANDLE;
		VkDescriptorSetLayout prefilterSetLayout = VK_NULL_HANDLE;

	public:
		virtual void init() override
		{
			CHECK(lutPipeline == VK_NULL_HANDLE);
			CHECK(lutPipelineLayout == VK_NULL_HANDLE);
			CHECK(lutSetLayout == VK_NULL_HANDLE);

			// Config code.
			RHI::get()->descriptorFactoryBegin()
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0) // outLut
				.buildNoInfoPush(lutSetLayout);

			std::vector<VkDescriptorSetLayout> setLayouts =
			{
				lutSetLayout, // Owner setlayout.
			};
			auto shaderModule = RHI::ShaderManager->getShader("BRDFLut.comp.spv", true);

			// Vulkan buid functions.
			VkPipelineLayoutCreateInfo plci = RHIPipelineLayoutCreateInfo();
			plci.setLayoutCount = (uint32_t)setLayouts.size();
			plci.pSetLayouts = setLayouts.data();
			lutPipelineLayout = RHI::get()->createPipelineLayout(plci);
			VkPipelineShaderStageCreateInfo shaderStageCI{};
			shaderStageCI.module = shaderModule;
			shaderStageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			shaderStageCI.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			shaderStageCI.pName = "main";
			VkComputePipelineCreateInfo computePipelineCreateInfo{};
			computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
			computePipelineCreateInfo.layout = lutPipelineLayout;
			computePipelineCreateInfo.flags = 0;
			computePipelineCreateInfo.stage = shaderStageCI;
			RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &lutPipeline));

			{
				CHECK(sphericalMapToCubePipeline == VK_NULL_HANDLE);
				CHECK(sphericalMapToCubePipelineLayout == VK_NULL_HANDLE);
				CHECK(sphericalMapToCubeSetLayout == VK_NULL_HANDLE);

				// Config code.
				RHI::get()->descriptorFactoryBegin()
					.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0) // out cube
					.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1) // hdr src
					.buildNoInfoPush(sphericalMapToCubeSetLayout);

				std::vector<VkDescriptorSetLayout> setLayouts =
				{
					sphericalMapToCubeSetLayout, // Owner setlayout.
					RHI::SamplerManager->getCommonDescriptorSetLayout()
				};
				auto shaderModule = RHI::ShaderManager->getShader("SphericalToCube.comp.spv", true);

				// Vulkan buid functions.
				VkPipelineLayoutCreateInfo plci = RHIPipelineLayoutCreateInfo();
				plci.setLayoutCount = (uint32_t)setLayouts.size();
				plci.pSetLayouts = setLayouts.data();
				sphericalMapToCubePipelineLayout = RHI::get()->createPipelineLayout(plci);
				VkPipelineShaderStageCreateInfo shaderStageCI{};
				shaderStageCI.module = shaderModule;
				shaderStageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				shaderStageCI.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				shaderStageCI.pName = "main";
				VkComputePipelineCreateInfo computePipelineCreateInfo{};
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.layout = sphericalMapToCubePipelineLayout;
				computePipelineCreateInfo.flags = 0;
				computePipelineCreateInfo.stage = shaderStageCI;
				RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &sphericalMapToCubePipeline));
			}

			{
				CHECK(irradiancePipeline == VK_NULL_HANDLE);
				CHECK(irradiancePipelineLayout == VK_NULL_HANDLE);
				CHECK(irradianceSetLayout == VK_NULL_HANDLE);

				// Config code.
				RHI::get()->descriptorFactoryBegin()
					.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0) // out irradiance
					.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1) // sample cube
					.buildNoInfoPush(irradianceSetLayout);

				std::vector<VkDescriptorSetLayout> setLayouts =
				{
					irradianceSetLayout, // Owner setlayout.
					RHI::SamplerManager->getCommonDescriptorSetLayout()
				};
				auto shaderModule = RHI::ShaderManager->getShader("IBLIrradiance.comp.spv", true);

				// Vulkan buid functions.
				VkPipelineLayoutCreateInfo plci = RHIPipelineLayoutCreateInfo();
				plci.setLayoutCount = (uint32_t)setLayouts.size();
				plci.pSetLayouts = setLayouts.data();
				irradiancePipelineLayout = RHI::get()->createPipelineLayout(plci);
				VkPipelineShaderStageCreateInfo shaderStageCI{};
				shaderStageCI.module = shaderModule;
				shaderStageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				shaderStageCI.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				shaderStageCI.pName = "main";
				VkComputePipelineCreateInfo computePipelineCreateInfo{};
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.layout = irradiancePipelineLayout;
				computePipelineCreateInfo.flags = 0;
				computePipelineCreateInfo.stage = shaderStageCI;
				RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &irradiancePipeline));
			}

			{
				CHECK(prefilterPipeline == VK_NULL_HANDLE);
				CHECK(prefilterPipelineLayout == VK_NULL_HANDLE);
				CHECK(prefilterSetLayout == VK_NULL_HANDLE);

				// Config code.
				RHI::get()->descriptorFactoryBegin()
					.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0) // out prefilter
					.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1) // sample cube
					.buildNoInfoPush(prefilterSetLayout);

				std::vector<VkDescriptorSetLayout> setLayouts =
				{
					prefilterSetLayout, // Owner setlayout.
					RHI::SamplerManager->getCommonDescriptorSetLayout()
				};
				auto shaderModule = RHI::ShaderManager->getShader("IBLPrefilter.comp.spv", true);

				// Vulkan buid functions.
				VkPipelineLayoutCreateInfo plci = RHIPipelineLayoutCreateInfo();

				VkPushConstantRange pushConstRange{ .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(IBLPrefilterPushConst) };

				plci.pushConstantRangeCount = 1;
				plci.pPushConstantRanges = &pushConstRange;

				plci.setLayoutCount = (uint32_t)setLayouts.size();
				plci.pSetLayouts = setLayouts.data();
				prefilterPipelineLayout = RHI::get()->createPipelineLayout(plci);
				VkPipelineShaderStageCreateInfo shaderStageCI{};
				shaderStageCI.module = shaderModule;
				shaderStageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				shaderStageCI.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				shaderStageCI.pName = "main";
				VkComputePipelineCreateInfo computePipelineCreateInfo{};
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.layout = prefilterPipelineLayout;
				computePipelineCreateInfo.flags = 0;
				computePipelineCreateInfo.stage = shaderStageCI;
				RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &prefilterPipeline));
			}
		}

		virtual void release() override
		{
			RHISafeRelease(lutPipeline);
			RHISafeRelease(lutPipelineLayout);
			lutSetLayout = VK_NULL_HANDLE;

			RHISafeRelease(irradiancePipeline);
			RHISafeRelease(irradiancePipelineLayout);
			irradianceSetLayout = VK_NULL_HANDLE;

			RHISafeRelease(sphericalMapToCubePipeline);
			RHISafeRelease(sphericalMapToCubePipelineLayout);
			sphericalMapToCubeSetLayout = VK_NULL_HANDLE;

			RHISafeRelease(prefilterPipeline);
			RHISafeRelease(prefilterPipelineLayout);
			prefilterSetLayout = VK_NULL_HANDLE;
		}
	};

	void StaticTextures::initIBL(VkCommandBuffer cmd, bool bRebuildLut)
	{
		auto* pass = m_passCollector->getPass<IBLComputePass>();

		if (bRebuildLut)
		{
			CHECK(m_brdfLut == nullptr && "BRDF lut only init once.");
			m_brdfLut = m_rtPool->createPoolImage(
				"BRDFLut",
				256u,
				256u,
				RTFormats::brdfLut(),
				RTUsages::brdfLut()
			);

			m_brdfLut->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
			{
				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->lutPipeline);

				VkDescriptorImageInfo lutImageInfo = RHIDescriptorImageInfoStorage(m_brdfLut->getImage().getView(buildBasicImageSubresource()));
				std::vector<VkWriteDescriptorSet> writes
				{
					RHIPushWriteDescriptorSetImage(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &lutImageInfo),
				};

				// Push owner set #0.
				RHI::PushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->lutPipelineLayout, 0, uint32_t(writes.size()), writes.data());

				vkCmdDispatch(cmd, getGroupCount(m_brdfLut->getImage().getExtent().width, 8), getGroupCount(m_brdfLut->getImage().getExtent().height, 8), 1);
			}
			m_brdfLut->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
		}
		
		if (RenderSettingManager::get()->ibl.iblEnable())
		{
			// TODO: When ibl bake ready, should release this hdr src owner lazy 3 frame.
			// TODO: Also should release env cubemap.
			RenderSettingManager::get()->ibl.setDirty(false);

			std::vector<VkDescriptorSet> compPassSets =
			{
				 RHI::SamplerManager->getCommonDescriptorSet()
			};

			m_iblEnvCube = m_rtPool->createPoolCubeImage(
				"GlobalIBLCubemap",
				512,  // Must can divide by 8.
				512,  // Must can divide by 8.
				VK_FORMAT_R16G16B16A16_SFLOAT,
				VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
				-1
			);

			// Spherical to cubemap level 0.
			{
				auto cubemapViewRange = VkImageSubresourceRange
				{
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 6
				};

				m_iblEnvCube->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, cubemapViewRange);
				{
					vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->sphericalMapToCubePipeline);
					vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
						pass->sphericalMapToCubePipelineLayout, 1,
						(uint32_t)compPassSets.size(), compPassSets.data(),
						0, nullptr
					);

					VkDescriptorImageInfo imageInfo = RHIDescriptorImageInfoStorage(m_iblEnvCube->getImage().getView(cubemapViewRange, VK_IMAGE_VIEW_TYPE_CUBE));
					VkDescriptorImageInfo hdrInfo = RHIDescriptorImageInfoSample(RenderSettingManager::get()->ibl.hdrSrc->getImage().getView(buildBasicImageSubresource()));
					std::vector<VkWriteDescriptorSet> writes
					{
						RHIPushWriteDescriptorSetImage(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &imageInfo),
						RHIPushWriteDescriptorSetImage(1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &hdrInfo),
					};

					// Push owner set #0.
					RHI::PushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->sphericalMapToCubePipelineLayout, 0, uint32_t(writes.size()), writes.data());

					// Mip 0
					vkCmdDispatch(cmd,
						getGroupCount(m_iblEnvCube->getImage().getExtent().width >> 0, 8),
						getGroupCount(m_iblEnvCube->getImage().getExtent().height >> 0, 8),
						6
					);
				}
				
				// Mip 0 as src input.
				m_iblEnvCube->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, cubemapViewRange);

				VkImageMemoryBarrier barrier{};
				barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
				barrier.image = m_iblEnvCube->getImage().getImage();
				barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
				barrier.subresourceRange.baseArrayLayer = 0;
				barrier.subresourceRange.layerCount = 6;
				barrier.subresourceRange.levelCount = 1;

				// Generate cubemap mips.
				int32_t mipWidth = m_iblEnvCube->getImage().getExtent().width;
				int32_t mipHeight = m_iblEnvCube->getImage().getExtent().height;
				for (uint32_t i = 1; i < m_iblEnvCube->getImage().getInfo().mipLevels; i++)
				{
					// Layout for write.
					barrier.subresourceRange.baseMipLevel = i;
					barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
					barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
					barrier.srcAccessMask = VK_ACCESS_NONE;
					barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
					vkCmdPipelineBarrier(cmd,
						VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
						0, nullptr,
						0, nullptr,
						1, &barrier);

					VkImageBlit blit{};

					blit.srcOffsets[0] = { 0, 0, 0 };
					blit.dstOffsets[0] = { 0, 0, 0 };

					blit.srcOffsets[1] = { mipWidth, mipHeight, 1 };
					blit.dstOffsets[1] = { mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1 };

					blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
					blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

					blit.srcSubresource.mipLevel = i - 1;
					blit.dstSubresource.mipLevel = i;

					blit.srcSubresource.baseArrayLayer = 0;
					blit.dstSubresource.baseArrayLayer = 0;

					blit.srcSubresource.layerCount = 6; // Cube map.
					blit.dstSubresource.layerCount = 6;

					vkCmdBlitImage(cmd,
						m_iblEnvCube->getImage().getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
						m_iblEnvCube->getImage().getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
						1, &blit,
						VK_FILTER_LINEAR
					);

					// Layout for read.
					barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
					barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
					barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
					barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
					vkCmdPipelineBarrier(cmd,
						VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
						0, nullptr,
						0, nullptr,
						1, &barrier);

					if (mipWidth  > 1) mipWidth  /= 2;
					if (mipHeight > 1) mipHeight /= 2;
				}

				cubemapViewRange.levelCount = m_iblEnvCube->getImage().getInfo().mipLevels;
				m_iblEnvCube->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, cubemapViewRange);
			}

			auto inCubeViewRange = VkImageSubresourceRange
			{
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel = 0,
				.levelCount = m_iblEnvCube->getImage().getInfo().mipLevels,
				.baseArrayLayer = 0,
				.layerCount = 6
			};
			VkDescriptorImageInfo hdrCubeInfo = RHIDescriptorImageInfoSample(m_iblEnvCube->getImage().getView(inCubeViewRange, VK_IMAGE_VIEW_TYPE_CUBE));

			// Irradiance
			{
				m_iblIrradiance = m_rtPool->createPoolCubeImage(
					"GlobalIBLIrradiance",
					128,  // Must can divide by 8.
					128,  // Must can divide by 8.
					VK_FORMAT_R16G16B16A16_SFLOAT,
					VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
				);

				auto irradianceViewRange = VkImageSubresourceRange
				{
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 6
				};

				m_iblIrradiance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, irradianceViewRange);
				{
					vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->irradiancePipeline);

					VkDescriptorImageInfo imageInfo = RHIDescriptorImageInfoStorage(m_iblIrradiance->getImage().getView(irradianceViewRange, VK_IMAGE_VIEW_TYPE_CUBE));
					std::vector<VkWriteDescriptorSet> writes
					{
						RHIPushWriteDescriptorSetImage(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &imageInfo),
						RHIPushWriteDescriptorSetImage(1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &hdrCubeInfo),
					};

					// Push owner set #0.
					RHI::PushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->irradiancePipelineLayout, 0, uint32_t(writes.size()), writes.data());

					vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
						pass->irradiancePipelineLayout, 1,
						(uint32_t)compPassSets.size(), compPassSets.data(),
						0, nullptr
					);

					vkCmdDispatch(cmd, getGroupCount(m_iblIrradiance->getImage().getExtent().width, 8), getGroupCount(m_iblIrradiance->getImage().getExtent().height, 8), 6);
				}
				m_iblIrradiance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, irradianceViewRange);
			}


			// Prefilter
			{
				m_iblPrefilter = m_rtPool->createPoolCubeImage(
					"GlobalIBLPrefilter",
					512,  // Must can divide by 8.
					512,  // Must can divide by 8.
					VK_FORMAT_R16G16B16A16_SFLOAT,
					VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
					-1
				);

				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->prefilterPipeline);

				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
					pass->prefilterPipelineLayout, 1,
					(uint32_t)compPassSets.size(), compPassSets.data(),
					0, nullptr
				);

				const float deltaRoughness = 1.0f / std::max(float(m_iblPrefilter->getImage().getInfo().mipLevels), 1.0f);

				for (uint32_t i = 0; i < m_iblPrefilter->getImage().getInfo().mipLevels; i ++)
				{

					auto viewRange = VkImageSubresourceRange
					{
						.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
						.baseMipLevel = i,
						.levelCount = 1,
						.baseArrayLayer = 0,
						.layerCount = 6
					};

					m_iblPrefilter->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, viewRange);
					{
						VkDescriptorImageInfo imageInfo = RHIDescriptorImageInfoStorage(m_iblPrefilter->getImage().getView(viewRange, VK_IMAGE_VIEW_TYPE_CUBE));
						
						std::vector<VkWriteDescriptorSet> writes
						{
							RHIPushWriteDescriptorSetImage(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &imageInfo),
							RHIPushWriteDescriptorSetImage(1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &hdrCubeInfo),
						};

						// Push owner set #0.
						RHI::PushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->prefilterPipelineLayout, 0, uint32_t(writes.size()), writes.data());

						IBLPrefilterPushConst push{ .perceptualRoughness = float(i) * deltaRoughness };

						vkCmdPushConstants(cmd, pass->prefilterPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

						vkCmdDispatch(cmd, 
							getGroupCount(glm::max(1u, m_iblPrefilter->getImage().getExtent().width >> i), 8),
							getGroupCount(glm::max(1u, m_iblPrefilter->getImage().getExtent().height >> i), 8),
							6);
					}
					m_iblPrefilter->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, viewRange);
				}
			}
		}

	}


	namespace blueNoise_16_Spp
	{
		// blue noise sampler 16spp.
		#include <samplerCPP/samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_16spp.cpp>
	}

	namespace blueNoise_8_Spp
	{
		// blue noise sampler 8spp.
		#include <samplerCPP/samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_8spp.cpp>
	}

	namespace blueNoise_4_Spp
	{
		// blue noise sampler 4spp.
		#include <samplerCPP/samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_4spp.cpp>
	}

	namespace blueNoise_2_Spp
	{
		// blue noise sampler 2spp.
		#include <samplerCPP/samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_2spp.cpp>
	}

	namespace blueNoise_1_Spp
	{
		// blue noise sampler 1spp.
		#include <samplerCPP/samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_1spp.cpp>
	}
	
	void GlobalBlueNoise::init()
	{
		auto buildBuffer = [](const char* name, void* ptr, VkDeviceSize size)
		{
			return VulkanBuffer::create(
				name,
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				EVMAUsageFlags::StageCopyForUpload,
				size,
				ptr
			);
		};

		#define WORK_CODE \
		{\
			BlueNoiseWorkingBuffer.sobolBuffer = buildBuffer(BlueNoiseWorkingName, (void*)BlueNoiseWorkingSpace::sobol_256spp_256d, sizeof(BlueNoiseWorkingSpace::sobol_256spp_256d));\
			BlueNoiseWorkingBuffer.rankingTileBuffer = buildBuffer(BlueNoiseWorkingName, (void*)BlueNoiseWorkingSpace::rankingTile, sizeof(BlueNoiseWorkingSpace::rankingTile));\
			BlueNoiseWorkingBuffer.scramblingTileBuffer = buildBuffer(BlueNoiseWorkingName, (void*)BlueNoiseWorkingSpace::scramblingTile, sizeof(BlueNoiseWorkingSpace::scramblingTile));\
			BlueNoiseWorkingBuffer.buildSet();\
		}


		#define BlueNoiseWorkingBuffer spp_1_buffer
		#define BlueNoiseWorkingSpace blueNoise_1_Spp
		#define BlueNoiseWorkingName "Sobel_1_spp_buffer"
					WORK_CODE
		#undef BlueNoiseWorkingBuffer
		#undef BlueNoiseWorkingSpace 
		#undef BlueNoiseWorkingName


		#define BlueNoiseWorkingBuffer spp_2_buffer
		#define BlueNoiseWorkingSpace blueNoise_2_Spp
		#define BlueNoiseWorkingName "Sobel_2_spp_buffer"
					WORK_CODE
		#undef BlueNoiseWorkingBuffer
		#undef BlueNoiseWorkingSpace 
		#undef BlueNoiseWorkingName

		#define BlueNoiseWorkingBuffer spp_4_buffer
		#define BlueNoiseWorkingSpace blueNoise_4_Spp
		#define BlueNoiseWorkingName "Sobel_4_spp_buffer"
					WORK_CODE
		#undef BlueNoiseWorkingBuffer
		#undef BlueNoiseWorkingSpace 
		#undef BlueNoiseWorkingName

		#define BlueNoiseWorkingBuffer spp_8_buffer
		#define BlueNoiseWorkingSpace blueNoise_8_Spp
		#define BlueNoiseWorkingName "Sobel_8_spp_buffer"
					WORK_CODE
		#undef BlueNoiseWorkingBuffer
		#undef BlueNoiseWorkingSpace 
		#undef BlueNoiseWorkingName

		#define BlueNoiseWorkingBuffer spp_16_buffer
		#define BlueNoiseWorkingSpace blueNoise_16_Spp
		#define BlueNoiseWorkingName "Sobel_16_spp_buffer"
					WORK_CODE
		#undef BlueNoiseWorkingBuffer
		#undef BlueNoiseWorkingSpace 
		#undef BlueNoiseWorkingName

		#undef WORK_CODE
	}

	void GlobalBlueNoise::BufferMisc::buildSet()
	{
		VkDescriptorBufferInfo sobolInfo{};
		sobolInfo.buffer = sobolBuffer->getVkBuffer();
		sobolInfo.offset = 0;
		sobolInfo.range = sobolBuffer->getSize();

		VkDescriptorBufferInfo rankingTileInfo{};
		rankingTileInfo.buffer = rankingTileBuffer->getVkBuffer();
		rankingTileInfo.offset = 0;
		rankingTileInfo.range = rankingTileBuffer->getSize();

		VkDescriptorBufferInfo scramblingTileInfo{};
		scramblingTileInfo.buffer = scramblingTileBuffer->getVkBuffer();
		scramblingTileInfo.offset = 0;
		scramblingTileInfo.range = scramblingTileBuffer->getSize();

		RHI::get()->descriptorFactoryBegin()
			.bindBuffers(0, 1, &sobolInfo, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
			.bindBuffers(1, 1, &rankingTileInfo, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
			.bindBuffers(2, 1, &scramblingTileInfo, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
			.build(set, setLayouts);
	}

}