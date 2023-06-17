#include "../renderer_interface.h"
#include "../render_scene.h"
#include "../renderer.h"
#include "../scene_textures.h"

namespace engine
{
	// Sample distribution shadow map implement here.
	struct GPUDepthRange
	{
		uint32_t minDepth;
		uint32_t maxDepth;
	};

	struct GPUSDSMPushConst
	{
		uint32_t cullCountPercascade;
		uint32_t cascadeCount;

		uint32_t cascadeId;
		uint32_t perCascadeMaxCount;

		uint32_t bHeightmapValid;
		float heightfiledDump;
	};

	class SDSMPass : public PassInterface
	{
	public:
		// Common set layout.
		VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;

		std::unique_ptr<ComputePipeResources> depthRangePipe;
		std::unique_ptr<ComputePipeResources> cascadePipe;
		std::unique_ptr<ComputePipeResources> cullPipe;
		std::unique_ptr<GraphicPipeResources> depthPipe;
		std::unique_ptr<ComputePipeResources> resolvePipe;

	protected:
		virtual void onInit() override
		{
			getContext()->descriptorFactoryBegin()
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCommonShaderStage, 0) // inDepth
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCommonShaderStage, 1) // SSBODepthRangeBuffer
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCommonShaderStage, 2) // SSBOCascadeInfoBuffer
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCommonShaderStage, 3) // inGbufferA
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCommonShaderStage, 4) // inShadowDepth
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCommonShaderStage, 5) // inGbufferB
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCommonShaderStage, 6) // inGbufferS
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCommonShaderStage, 7) // imageShadowMask
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCommonShaderStage, 8) // inGbufferB
				.bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kCommonShaderStage, 9) // frameData
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCommonShaderStage, 10) // objectDatas
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCommonShaderStage, 11) // indirectCommands
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCommonShaderStage, 12) // drawCount
				.buildNoInfoPush(setLayout);

			std::vector<VkDescriptorSetLayout> basicSetLayouts = {
				setLayout,                
				getContext()->getSamplerCache().getCommonDescriptorSetLayout(), 
			};

			depthRangePipe = std::make_unique<ComputePipeResources>("shader/sdsm_range.comp.spv", (uint32_t)sizeof(GPUSDSMPushConst), basicSetLayouts);
			cascadePipe = std::make_unique<ComputePipeResources>("shader/sdsm_cascade.comp.spv", (uint32_t)sizeof(GPUSDSMPushConst), basicSetLayouts);
			cullPipe = std::make_unique<ComputePipeResources>("shader/sdsm_cull.comp.spv", (uint32_t)sizeof(GPUSDSMPushConst), basicSetLayouts);

			std::vector<VkDescriptorSetLayout> depthSetLayouts = 
			{
				  setLayout
				, m_context->getBindlessSSBOSetLayout()
				, m_context->getBindlessSSBOSetLayout()
				, m_context->getBindlessTextureSetLayout()
				, m_context->getBindlessSamplerSetLayout()
			};
			depthPipe = std::make_unique<GraphicPipeResources>(
				"shader/sdsm_depth.vert.spv", 
				"shader/sdsm_depth.frag.spv",
				depthSetLayouts,
				(uint32_t)sizeof(GPUSDSMPushConst),
				std::vector<VkFormat>{ },
				std::vector<VkPipelineColorBlendAttachmentState>{ },
				GBufferTextures::depthTextureFormat(),
				VK_CULL_MODE_FRONT_BIT,
				VK_COMPARE_OP_GREATER,
				true,
				true);

			std::vector<VkDescriptorSetLayout> resolveSetLayouts =
			{
				  setLayout
				, m_context->getSamplerCache().getCommonDescriptorSetLayout()
				, getRenderer()->getBlueNoise().spp_1_buffer.setLayouts
			};
			resolvePipe = std::make_unique<ComputePipeResources>("shader/sdsm_resolve.comp.spv", (uint32_t)sizeof(GPUSDSMPushConst), resolveSetLayouts);
		}

		virtual void release() override
		{
			depthRangePipe.reset();
			cascadePipe.reset();
			cullPipe.reset();
			depthPipe.reset();
			resolvePipe.reset();
		}
	};

	void SDSMInfos::build(const CascadeShadowConfig* config, RendererInterface* renderer)
	{
		const bool bFallback = (config == nullptr) || (renderer == nullptr);

		cascadeInfoBuffer = getContext()->getBufferParameters().getStaticStorageGPUOnly(
			"CascadeInfos",
			bFallback ? sizeof(uint32_t) : sizeof(GPUCascadeInfo) * config->cascadeCount
		);

		shadowDepths = getContext()->getRenderTargetPools().createPoolImage(
			"SDSMDepth",
			bFallback ? 1u : config->percascadeDimXY * config->cascadeCount,
			bFallback ? 1u : config->percascadeDimXY,
			GBufferTextures::depthTextureFormat(),
			VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
		);

		mainViewMask = getContext()->getRenderTargetPools().createPoolImage(
			"SDSMShadowMask",
			bFallback ? 1u : renderer->getRenderWidth(),
			bFallback ? 1u : renderer->getRenderHeight(),
			VK_FORMAT_R8_UNORM,
			VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
		);
	}

	void RendererInterface::renderSDSM(
		VkCommandBuffer cmd, 
		GBufferTextures* inGBuffers,
		RenderScene* scene, 
		BufferParameterHandle perFrameGPU,
		SDSMInfos& sdsmInfos)
	{
		sdsmInfos.build(nullptr, nullptr);
		if (!scene->shouldRenderSDSM())
		{
			return;
		}



		const auto& gpuInfo = scene->getSkyGPU();
		const bool bStaticMeshRenderSDSM = gpuInfo.rayTraceShadow == 0;

		auto& sceneDepthZ = inGBuffers->depthTexture->getImage();
		sceneDepthZ.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));

		const uint32_t staticMeshCount = bStaticMeshRenderSDSM ?(uint32_t)scene->getStaticMeshObjects().size() : 0;

		auto& gBufferA = inGBuffers->gbufferA->getImage();
		auto& gBufferB = inGBuffers->gbufferB->getImage();
		auto& gBufferS = inGBuffers->gbufferS->getImage();
		gBufferA.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		gBufferB.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		gBufferS.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));

		auto* pass = m_context->getPasses().get<SDSMPass>();
		auto rangeBuffer = m_context->getBufferParameters().getStaticStorageGPUOnly("SDSMRangeBuffer", sizeof(GPUDepthRange));

		sdsmInfos.build(&gpuInfo.cacsadeConfig, this);
		auto& cascadeBuffer = sdsmInfos.cascadeInfoBuffer;
		auto& sdsmDepth = sdsmInfos.shadowDepths;
		auto& sdsmMask = sdsmInfos.mainViewMask;

		// Basic setBuilder.
		auto& terrains = scene->getTerrains();
		auto& pmxes = scene->getPMXes();

		PushSetBuilder setBuilder(cmd);
		setBuilder
			.addSRV(sceneDepthZ, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT))
			.addBuffer(rangeBuffer)
			.addBuffer(cascadeBuffer)
			.addSRV(gBufferA)
			.addSRV(sdsmDepth, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT))
			.addSRV(gBufferB)
			.addSRV(gBufferS)
			.addUAV(sdsmMask)
			.addSRV(terrains.empty() ? gBufferA : terrains[0].lock()->getHeightfiledImage())
			.addBuffer(perFrameGPU)
			.addBuffer(scene->getStaticMeshObjectsGPU());

		GPUSDSMPushConst pushConst
		{
			.cullCountPercascade = (uint32_t)staticMeshCount,
			.cascadeCount = (uint32_t)gpuInfo.cacsadeConfig.cascadeCount,

			.cascadeId = 0,
			.perCascadeMaxCount = (uint32_t)staticMeshCount,

			.bHeightmapValid = terrains.empty() ? 0U : 1U,
			.heightfiledDump = terrains.empty() ? 1.0f : terrains[0].lock()->getSetting().dumpFactor,
		};

		{
			ScopePerframeMarker marker(cmd, "DepthRangeCompute", { 1.0f, 0.0f, 0.0f, 1.0f });

			GPUDepthRange clearRangeValue = { .minDepth = ~0u, .maxDepth = 0u };
			vkCmdUpdateBuffer(cmd, *rangeBuffer->getBuffer(), 0, rangeBuffer->getBuffer()->getSize(), &clearRangeValue);
			std::array<VkBufferMemoryBarrier2, 1> fillBarriers
			{
				RHIBufferBarrier(rangeBuffer->getBuffer()->getVkBuffer(),
					VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT),
			};
			RHIPipelineBarrier(cmd, 0, (uint32_t)fillBarriers.size(), fillBarriers.data(), 0, nullptr);

			pass->depthRangePipe->bindAndPushConst(cmd, &pushConst);
			setBuilder.push(pass->depthRangePipe.get());

			// Block dim is 3x3.
			vkCmdDispatch(cmd,
				getGroupCount(sceneDepthZ.getExtent().width / 3 + 1, 8),
				getGroupCount(sceneDepthZ.getExtent().height / 3 + 1, 8), 1);

			VkBufferMemoryBarrier2 endBufferBarrier = RHIBufferBarrier(rangeBuffer->getBuffer()->getVkBuffer(),
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
			RHIPipelineBarrier(cmd, 0, 1, &endBufferBarrier, 0, nullptr);
		}

		{
			ScopePerframeMarker marker(cmd, "PrepareCascadeInfo", { 1.0f, 0.0f, 0.0f, 1.0f });

			pass->cascadePipe->bindAndPushConst(cmd, &pushConst);
			setBuilder.push(pass->cascadePipe.get());

			vkCmdDispatch(cmd, getGroupCount(kMaxCascadeNum, 32), 1, 1);

			VkBufferMemoryBarrier2 endBufferBarrier = RHIBufferBarrier(cascadeBuffer->getBuffer()->getVkBuffer(),
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
			RHIPipelineBarrier(cmd, 0, 1, &endBufferBarrier, 0, nullptr);
		}

		

		{
			const auto cullingCount = math::max(1U, gpuInfo.cacsadeConfig.cascadeCount * staticMeshCount);

			auto indirectDrawCommandBuffer = m_context->getBufferParameters().getIndirectStorage("SDSMMeshIndirectCommand",
				cullingCount * sizeof(GPUStaticMeshDrawCommand));

			auto indirectDrawCountBuffer = m_context->getBufferParameters().getIndirectStorage("SDSMMeshIndirectCount",
				sizeof(uint32_t)* gpuInfo.cacsadeConfig.cascadeCount);

			auto staticMeshSetBuilder = setBuilder;
			staticMeshSetBuilder
				.addBuffer(indirectDrawCommandBuffer)
				.addBuffer(indirectDrawCountBuffer);

			// Culling.
			{
				ScopePerframeMarker marker(cmd, "SDSMCulling", { 1.0f, 0.0f, 0.0f, 1.0f });

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

				vkCmdDispatch(cmd, getGroupCount(cullingCount, 64), 1, 1);

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
			sdsmDepth->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));
			VkRenderingAttachmentInfo depthAttachment = getDepthAttachment(sdsmDepth);
			{

				ScopeRenderCmdObject renderCmdScope(cmd, "SDSMShadowDepth", sdsmDepth->getImage(), {}, depthAttachment);

				vkCmdSetDepthBias(cmd, gpuInfo.cacsadeConfig.shadowBiasConst, 0, gpuInfo.cacsadeConfig.shadowBiasSlope);
				{
					// Set depth bias for shadow depth rendering to avoid shadow artifact.
					for (uint32_t cascadeIndex = 0; cascadeIndex < gpuInfo.cacsadeConfig.cascadeCount; cascadeIndex++)
					{
						VkRect2D scissor{};
						scissor.extent = { (uint32_t)gpuInfo.cacsadeConfig.percascadeDimXY, (uint32_t)gpuInfo.cacsadeConfig.percascadeDimXY };
						scissor.offset = { int32_t(gpuInfo.cacsadeConfig.percascadeDimXY * cascadeIndex), 0 };

						VkViewport viewport{};
						viewport.minDepth = 0.0f;
						viewport.maxDepth = 1.0f;
						viewport.y = (float)gpuInfo.cacsadeConfig.percascadeDimXY;
						viewport.height = -(float)gpuInfo.cacsadeConfig.percascadeDimXY;
						viewport.x = (float)gpuInfo.cacsadeConfig.percascadeDimXY * (float)cascadeIndex;
						viewport.width = (float)gpuInfo.cacsadeConfig.percascadeDimXY;

						vkCmdSetScissor(cmd, 0, 1, &scissor);
						vkCmdSetViewport(cmd, 0, 1, &viewport);

						if(bStaticMeshRenderSDSM)
						{
							pass->depthPipe->bind(cmd);
							staticMeshSetBuilder.push(pass->depthPipe.get());

							pass->depthPipe->bindSet(cmd, std::vector<VkDescriptorSet>{
								m_context->getBindlessSSBOSet()
									, m_context->getBindlessSSBOSet()
									, m_context->getBindlessTextureSet()
									, m_context->getBindlessSamplerSet()
							}, 1);

							pushConst.cascadeId = cascadeIndex;
							pass->depthPipe->pushConst(cmd, &pushConst);

							vkCmdDrawIndirectCount(cmd,
								indirectDrawCommandBuffer->getBuffer()->getVkBuffer(),
								cascadeIndex * sizeof(GPUStaticMeshDrawCommand) * staticMeshCount,
								indirectDrawCountBuffer->getBuffer()->getVkBuffer(),
								cascadeIndex * sizeof(uint32_t),
								staticMeshCount,
								sizeof(GPUStaticMeshDrawCommand)
							);
						}

						for (auto& pmx : pmxes)
						{
							if (auto comp = pmx.lock())
							{
								comp->onRenderSDSMDepthCollect(
									cmd, perFrameGPU, inGBuffers, scene, 
									this, sdsmInfos, cascadeIndex);
							}
						}

						// Also render all terrain depth here.
						for (auto& terrain : terrains)
						{
							if (auto comp = terrain.lock())
							{
								comp->renderSDSMDepth(cmd, perFrameGPU, inGBuffers, scene, this, sdsmInfos, cascadeIndex);
							}
						}
					}
				}
			}
		}

		{
			ScopePerframeMarker marker(cmd, "SDSM resolve", { 1.0f, 0.0f, 0.0f, 1.0f });
			sdsmDepth->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));
			sdsmMask->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));

			pass->resolvePipe->bindAndPushConst(cmd, &pushConst);
			setBuilder.push(pass->resolvePipe.get());

			pass->resolvePipe->bindSet(cmd, std::vector<VkDescriptorSet>{
				  m_context->getSamplerCache().getCommonDescriptorSet()
				, m_renderer->getBlueNoise().spp_1_buffer.set
			}, 1);

			vkCmdDispatch(cmd, getGroupCount(sdsmMask->getImage().getExtent().width, 8), getGroupCount(sdsmMask->getImage().getExtent().height, 8), 1);
			sdsmMask->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		}



		m_gpuTimer.getTimeStamp(cmd, "SDSM");
	}
}