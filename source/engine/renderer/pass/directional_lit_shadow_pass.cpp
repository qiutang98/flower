#include "../deferred_renderer.h"
#include "../render_scene.h"
#include "../renderer.h"
#include "../scene_textures.h"
#include "../../scene/component/sky_component.h"

namespace engine
{
	constexpr float kDepthBiasClampMin = 0.0f;

	static_assert(kMaxCascadeNum % 4 == 0);
	struct GPUSDSMPushConst
	{
		uint sdsmShadowDepthIndices[kMaxCascadeNum];

		// For culling.
		uint cullCountPercascade;
		uint cascadeCount;
		uint cascadeId;
		uint bSDSM;

		vec3 lightDirection;
		float maxDrawDepthDistance;

		uint percascadeDimXY;
		float cascadeSplitLambda;
		float filterSize;
		float cascadeMixBorder;

		float contactShadowLength;
		uint contactShadowSampleNum;
		uint bContactShadow;
		uint bCloudShadow;
	};
	static_assert(sizeof(GPUSDSMPushConst) <= kMaxPushConstSize);


	struct RtPipePush
	{
		vec3 lightDirection;
		float lightRadius;

		float rayMinRange;
		float rayMaxRange;
	};

	class SDSMPass : public PassInterface
	{
	public:


		std::unique_ptr<ComputePipeResources> cascadePipe;
		std::unique_ptr<ComputePipeResources> cullPipe;
		std::unique_ptr<GraphicPipeResources> depthPipe;
		std::unique_ptr<ComputePipeResources> resolvePipe;

		std::unique_ptr<ComputePipeResources> rtPipe;

	protected:
		virtual void onInit() override
		{
			{
				VkDescriptorSetLayout rtSetLayout = VK_NULL_HANDLE;
				getContext()->descriptorFactoryBegin()
					.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 0) // shadow
					.bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1) // inFrameData
					.bindNoInfo(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 2) // AS
					.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 3) // inDepth
					.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4)
					.buildNoInfoPush(rtSetLayout);

				std::vector<VkDescriptorSetLayout> layouts{
					rtSetLayout,
					m_context->getSamplerCache().getCommonDescriptorSetLayout(),
					getRenderer()->getBlueNoise().spp_1_buffer.setLayouts,
					m_context->getBindlessSSBOSetLayout()
					, m_context->getBindlessSSBOSetLayout()
					, m_context->getBindlessSamplerSetLayout()
					, m_context->getBindlessTextureSetLayout()
				};

				rtPipe = std::make_unique<ComputePipeResources>("shader/rt_shadow_directionalLit.glsl", sizeof(RtPipePush), layouts);
			}

			VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;

			getContext()->descriptorFactoryBegin()
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  0) // inDepth
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1) // SSBOCascadeInfoBuffer
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  2) // inGbufferB
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  3) // inCloudShadowDepth
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  4) // inTerrainShadowDepth
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  5) // imageShadowMask
				.bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 6) // frameData
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7) // objectDatas
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8) // inSceneDepthRange
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 9) // indirectCommands
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,10) // drawCount


				.buildNoInfoPush(setLayout);

			std::vector<VkDescriptorSetLayout> basicSetLayouts = 
			{
				setLayout,
				getContext()->getSamplerCache().getCommonDescriptorSetLayout(),
				getRenderer()->getBlueNoise().spp_8_buffer.setLayouts,
				getContext()->getBindlessSSBOSetLayout(),
				getContext()->getBindlessSSBOSetLayout(),
				getContext()->getBindlessSamplerSetLayout(),
				getContext()->getBindlessTexture().getSetLayout(),
			};

			{
				auto variant = ShaderVariant("shader/shadow_directionalLit.glsl").setStage(eComputeShader).setMacro(L"CASCADE_PREPARE_PASS");
				cascadePipe = std::make_unique<ComputePipeResources>(variant, sizeof(GPUSDSMPushConst), basicSetLayouts);
			}

			{
				auto variant = ShaderVariant("shader/shadow_directionalLit.glsl").setStage(eComputeShader).setMacro(L"CASCADE_CULL_PASS");
				cullPipe = std::make_unique<ComputePipeResources>(variant, sizeof(GPUSDSMPushConst), basicSetLayouts);
			}

			{
				ShaderVariant vertexShaderVariant("shader/shadow_directionalLit.glsl");
				vertexShaderVariant.setStage(EShaderStage::eVertexShader).setMacro(L"CASCADE_DEPTH_PASS");

				ShaderVariant fragmentShaderVariant("shader/shadow_directionalLit.glsl");
				fragmentShaderVariant.setStage(EShaderStage::ePixelShader).setMacro(L"CASCADE_DEPTH_PASS");

				depthPipe = std::make_unique<GraphicPipeResources>(
                    vertexShaderVariant,
                    fragmentShaderVariant,
					basicSetLayouts,
					(uint32_t)sizeof(GPUSDSMPushConst),
					std::vector<VkFormat>{ },
					std::vector<VkPipelineColorBlendAttachmentState>{ },
					GBufferTextures::depthTextureFormat(),
					VK_CULL_MODE_NONE,
					VK_COMPARE_OP_GREATER,
					true,
					true);
			}

			{
				auto variant = ShaderVariant("shader/shadow_directionalLit.glsl").setStage(eComputeShader).setMacro(L"SHADOW_MASK_EVALUATE_PASS");
				resolvePipe = std::make_unique<ComputePipeResources>(variant, sizeof(GPUSDSMPushConst), basicSetLayouts);
			}
		}

		virtual void release() override
		{
			cascadePipe.reset();
			cullPipe.reset();
			depthPipe.reset();
			resolvePipe.reset();
			rtPipe.reset();
		}
	};

	void SDSMInfos::build(const SkyLightInfo* sky, uint width, uint height)
	{
		const auto* config = sky ? &sky->cascadeConfig : nullptr;
		const bool bFallback = (config == nullptr);

		cascadeInfoBuffer = getContext()->getBufferParameters().getStaticStorageGPUOnly(
			"CascadeInfos",
			bFallback ? sizeof(uint32_t) : sizeof(CascadeInfo) * config->cascadeCount
		);

		// Pack all shadow depths in one atlas.
		shadowDepths.resize(bFallback ? 1 : sky->cascadeConfig.cascadeCount);
		for (size_t i = 0; i < shadowDepths.size(); i++)
		{
			shadowDepths[i] = getContext()->getRenderTargetPools().createPoolImage(
				"SDSMDepth",
				bFallback ? 1u : config->percascadeDimXY,
				bFallback ? 1u : config->percascadeDimXY,
				GBufferTextures::depthTextureFormat(),
				VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
			);
		}


		shadowMask = getContext()->getRenderTargetPools().createPoolImage(
			"SDSMShadowMask",
			bFallback ? 1u : width,
			bFallback ? 1u : height,
			VK_FORMAT_R8G8_UNORM,
			VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
		);
	}

	void engine::renderSDSM(
		VkCommandBuffer cmd,
		GBufferTextures* inGBuffers,
		const SkyLightInfo& sunSkyInfo,
		RenderScene* scene,
		BufferParameterHandle perFrameGPU,
		SDSMInfos& sunSDSMInfos,
		SDSMInfos& moonSDSMInfos,
		BufferParameterHandle sceneDepthRange,
		GPUTimestamps* timer,
		PoolImageSharedRef sunCloudShadowDepth)
	{


		if (timer)
		{
			timer->getTimeStamp(cmd, "terrain shadow depth");
		}

		sunSDSMInfos.build(nullptr, 1U, 1U);
		moonSDSMInfos.build(nullptr, 1U, 1U);

		const uint32_t objectCount = (uint32_t)scene->getObjectCollector().size();
		if (objectCount <= 0)
		{
			return;
		}

		auto* skyComp = scene->getSkyComponent();
		if (!skyComp) { return; }

		auto& sceneDepthZ = inGBuffers->depthTexture->getImage();
		auto& gBufferB = inGBuffers->gbufferB->getImage();

		sceneDepthZ.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));

		gBufferB.transitionShaderReadOnly(cmd);

		auto* pass = getContext()->getPasses().get<SDSMPass>();


		auto buildSDSM = [&](const SkyLightInfo& skyInfo, SDSMInfos& worker)
		{
			GPUSDSMPushConst pushConst
			{
				.cullCountPercascade = (uint32_t)objectCount,
				.cascadeCount = (uint32_t)skyInfo.cascadeConfig.cascadeCount,
				.bSDSM = (uint)skyInfo.cascadeConfig.bSDSM,
				.lightDirection = math::normalize(skyInfo.direction),
				.maxDrawDepthDistance = skyInfo.cascadeConfig.maxDrawDepthDistance,
				.percascadeDimXY = (uint)skyInfo.cascadeConfig.percascadeDimXY,
				.cascadeSplitLambda = skyInfo.cascadeConfig.splitLambda,
				.filterSize = skyInfo.cascadeConfig.filterSize,
				.cascadeMixBorder = skyInfo.cascadeConfig.cascadeMixBorder,
				.contactShadowLength = skyInfo.cascadeConfig.contactShadowLen,
				.contactShadowSampleNum = (uint)skyInfo.cascadeConfig.contactShadowSampleNum,
				.bContactShadow = (uint)skyInfo.cascadeConfig.bContactShadow,
				.bCloudShadow = (uint)(sunCloudShadowDepth != nullptr)
			};

			
			worker.build(&skyInfo, sceneDepthZ.getExtent().width, sceneDepthZ.getExtent().height);

			for (int i = skyInfo.cascadeConfig.cascadeCount - 1; i >= 0; i--)
			{
				pushConst.sdsmShadowDepthIndices[i] =
					worker.shadowDepths[i]->getImage().getOrCreateView(RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT)).srvBindless;
			}

			auto& cascadeBuffer = worker.cascadeInfoBuffer;
			auto& sdsmDepth     = worker.shadowDepths;
			auto& sdsmMask      = worker.shadowMask;

			PushSetBuilder commonSetBuilder(cmd);
			commonSetBuilder
				.addSRV(sceneDepthZ, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT))
				.addBuffer(cascadeBuffer)
				.addSRV(gBufferB)
				.addSRV(sunCloudShadowDepth == nullptr ? 
					getContext()->getBuiltinTextureTranslucent()->getSelfImage() : sunCloudShadowDepth->getImage())
				.addSRV(getContext()->getBuiltinTextureTranslucent()->getSelfImage())
				.addUAV(sdsmMask)
				.addBuffer(perFrameGPU)
				.addBuffer(scene->getObjectBufferGPU() ? scene->getObjectBufferGPU() : cascadeBuffer)
				.addBuffer(sceneDepthRange);

			{
				ScopePerframeMarker marker(cmd, "PrepareCascadeInfo", { 1.0f, 0.0f, 0.0f, 1.0f }, nullptr);

				pass->cascadePipe->bindAndPushConst(cmd, &pushConst);
				auto setBuilder = commonSetBuilder;
				setBuilder.push(pass->cascadePipe.get());

				pass->cascadePipe->bindSet(cmd, std::vector<VkDescriptorSet>{
					getContext()->getSamplerCache().getCommonDescriptorSet(),
					getRenderer()->getBlueNoise().spp_8_buffer.set,
					getContext()->getBindlessSSBOSet(), 
					getContext()->getBindlessSSBOSet(), 
					getContext()->getBindlessSamplerSet(),
					getContext()->getBindlessTexture().getSet(),
				}, 1);

				vkCmdDispatch(cmd, getGroupCount(kMaxCascadeNum, 32), 1, 1);

				VkBufferMemoryBarrier2 endBufferBarrier = RHIBufferBarrier(cascadeBuffer->getBuffer()->getVkBuffer(),
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
				RHIPipelineBarrier(cmd, 0, 1, &endBufferBarrier, 0, nullptr);
			}

			auto indirectDrawCommandBuffer = getContext()->getBufferParameters().getIndirectStorage(
				"SDSMMeshIndirectCommand", objectCount * sizeof(StaticMeshDrawCommand));

			auto indirectDrawCountBuffer = getContext()->getBufferParameters().getIndirectStorage(
				"SDSMMeshIndirectCount", sizeof(uint32_t));

			auto staticMeshSetBuilder = commonSetBuilder;
			staticMeshSetBuilder
				.addBuffer(indirectDrawCommandBuffer)
				.addBuffer(indirectDrawCountBuffer);

			for (int i = skyInfo.cascadeConfig.cascadeCount - 1; i >= 0; i--)
			{
				// Update push const.
				pushConst.cascadeId = i;

				// Culling.
				{
					ScopePerframeMarker marker(cmd, std::format("SDSMCulling {}", i), { 1.0f, 0.0f, 0.0f, 1.0f }, nullptr);

					vkCmdFillBuffer(cmd, *indirectDrawCountBuffer->getBuffer(), 0, indirectDrawCountBuffer->getBuffer()->getSize(), 0u);
					vkCmdFillBuffer(cmd, *indirectDrawCommandBuffer->getBuffer(), 0, indirectDrawCommandBuffer->getBuffer()->getSize(), 0u);
					std::array<VkBufferMemoryBarrier2, 2> fillBarriers
					{
						RHIBufferBarrier(indirectDrawCommandBuffer->getBuffer()->getVkBuffer(),
							VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
							VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT),
						RHIBufferBarrier(indirectDrawCountBuffer->getBuffer()->getVkBuffer(),
							VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
							VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT),
					};
					RHIPipelineBarrier(cmd, 0, (uint32_t)fillBarriers.size(), fillBarriers.data(), 0, nullptr);

					pass->cullPipe->bindAndPushConst(cmd, &pushConst);
					staticMeshSetBuilder.push(pass->cullPipe.get());

					pass->cullPipe->bindSet(cmd, std::vector<VkDescriptorSet>{
						getContext()->getSamplerCache().getCommonDescriptorSet(),
						getRenderer()->getBlueNoise().spp_8_buffer.set,
						getContext()->getBindlessSSBOSet(), 
						getContext()->getBindlessSSBOSet(), 
						getContext()->getBindlessSamplerSet(),
						getContext()->getBindlessTexture().getSet(),
					}, 1);

					vkCmdDispatch(cmd, getGroupCount(objectCount, 64), 1, 1);

					// End buffer barrier.
					std::array<VkBufferMemoryBarrier2, 2> endBufferBarriers
					{
						RHIBufferBarrier(indirectDrawCommandBuffer->getBuffer()->getVkBuffer(),
							VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
							VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT),

						RHIBufferBarrier(indirectDrawCountBuffer->getBuffer()->getVkBuffer(),
							VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
							VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT),
					};
					RHIPipelineBarrier(cmd, 0, (uint32_t)endBufferBarriers.size(), endBufferBarriers.data(), 0, nullptr);
				}

				// Render Depth.
				worker.shadowDepths[i]->getImage().transitionLayout(
					cmd, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 
					RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));

				VkRenderingAttachmentInfo depthAttachment = getDepthAttachment(worker.shadowDepths[i]);
				{
					ScopeRenderCmdObject renderCmdScope(cmd, nullptr, std::format("SDSMDepth {}", i), worker.shadowDepths[i]->getImage(), {}, depthAttachment);

					vkCmdSetDepthBias(cmd, 
						skyInfo.cascadeConfig.shadowBiasConst, kDepthBiasClampMin,
						skyInfo.cascadeConfig.shadowBiasSlope);

					VkRect2D scissor{ };
					scissor.extent = { (uint32_t)skyInfo.cascadeConfig.percascadeDimXY, (uint32_t)skyInfo.cascadeConfig.percascadeDimXY };
					scissor.offset = { 0, 0 };

					VkViewport viewport{ };
					viewport.minDepth = 0.0f;
					viewport.maxDepth = 1.0f;
					viewport.x = 0;
					viewport.y = (float)skyInfo.cascadeConfig.percascadeDimXY;
					viewport.height = -(float)skyInfo.cascadeConfig.percascadeDimXY;
					viewport.width  =  (float)skyInfo.cascadeConfig.percascadeDimXY;

					vkCmdSetScissor(cmd, 0, 1, &scissor);
					vkCmdSetViewport(cmd, 0, 1, &viewport);

					renderTerrainSDSMDepth(cmd, perFrameGPU, inGBuffers, scene, worker, i);

					pass->depthPipe->bindAndPushConst(cmd, &pushConst);
					staticMeshSetBuilder.push(pass->depthPipe.get());

					pass->depthPipe->bindSet(cmd, std::vector<VkDescriptorSet>{
						getContext()->getSamplerCache().getCommonDescriptorSet(),
						getRenderer()->getBlueNoise().spp_1_buffer.set,
						getContext()->getBindlessSSBOSet(), 
						getContext()->getBindlessSSBOSet(), 
						getContext()->getBindlessSamplerSet(),
						getContext()->getBindlessTexture().getSet(),
					}, 1);

					vkCmdDrawIndirectCount(cmd,
						indirectDrawCommandBuffer->getBuffer()->getVkBuffer(),
						0,
						indirectDrawCountBuffer->getBuffer()->getVkBuffer(),
						0,
						objectCount,
						sizeof(StaticMeshDrawCommand)
					);
				}

				worker.shadowDepths[i]->getImage().transitionLayout(
					cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));
			}
			if (timer)
			{
				timer->getTimeStamp(cmd, "sdsm depth");
			}
		

			sdsmMask->getImage().transitionGeneral(cmd);
			// if (skyInfo.shadowType == EShadowType_CascadeShadowMap || !scene->isTLASValid())
			if (true)
			{
				ScopePerframeMarker marker(cmd, "SDSM resolve", { 1.0f, 0.0f, 0.0f, 1.0f }, timer);

				pass->resolvePipe->bindAndPushConst(cmd, &pushConst);
				commonSetBuilder.push(pass->resolvePipe.get());

				pass->resolvePipe->bindSet(cmd, std::vector<VkDescriptorSet>{
					getContext()->getSamplerCache().getCommonDescriptorSet(),
					getRenderer()->getBlueNoise().spp_8_buffer.set,
					getContext()->getBindlessSSBOSet(),
					getContext()->getBindlessSSBOSet(),
					getContext()->getBindlessSamplerSet(),
					getContext()->getBindlessTexture().getSet(),
				}, 1);

				vkCmdDispatch(cmd, getGroupCount(
					sdsmMask->getImage().getExtent().width, 8), 
					getGroupCount(sdsmMask->getImage().getExtent().height, 8), 1);
			}
			else
			{
				ScopePerframeMarker marker(cmd, "Rt shadow", { 1.0f, 0.0f, 0.0f, 1.0f }, timer);

				RtPipePush push{};
				push.lightDirection = skyInfo.direction;
				push.rayMaxRange = skyInfo.rayTraceConfig.rayMaxRange;
				push.rayMinRange = skyInfo.rayTraceConfig.rayMinRange;
				push.lightRadius = skyInfo.rayTraceConfig.lightRadius;

				pass->rtPipe->bindAndPushConst(cmd, &push);
				PushSetBuilder(cmd)
					.addUAV(sdsmMask)
					.addBuffer(perFrameGPU)
					.addAS(scene->getTLAS())
					.addSRV(sceneDepthZ, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT))
					.addBuffer(scene->getObjectBufferGPU())
					.push(pass->rtPipe.get());

				pass->rtPipe->bindSet(cmd, std::vector<VkDescriptorSet>{
					getContext()->getSamplerCache().getCommonDescriptorSet(),
					getRenderer()->getBlueNoise().spp_1_buffer.set,
					getContext()->getBindlessSSBOSet()
						, getContext()->getBindlessSSBOSet()
						, getContext()->getBindlessSamplerSet()
						, getContext()->getBindlessTextureSet()

				}, 1);

				vkCmdDispatch(cmd, 
					getGroupCount(sdsmMask->getImage().getExtent().width, 8), 
					getGroupCount(sdsmMask->getImage().getExtent().height, 8), 1);
			}

			sdsmMask->getImage().transitionShaderReadOnly(cmd);
		};

		buildSDSM(sunSkyInfo, sunSDSMInfos);


	}
}