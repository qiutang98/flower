#include "../renderer_interface.h"
#include "../render_scene.h"
#include "../renderer.h"
#include "../scene_textures.h"

namespace engine
{
	struct PMXSDSMPushConsts
	{
		math::mat4 modelMatrix;
		uint32_t   colorTexId;
		uint32_t   cascadeId;
	};


	class PMXPass : public PassInterface
	{
	public:
		VkDescriptorSetLayout frameDataSetLayout = VK_NULL_HANDLE;
		std::unique_ptr<GraphicPipeResources> pmxPass;
		std::unique_ptr<GraphicPipeResources> pmxOutlinePass;
		std::unique_ptr<GraphicPipeResources> pmxTranslucencyPass;

		VkDescriptorSetLayout sdsmSetLayout = VK_NULL_HANDLE;
		std::unique_ptr<GraphicPipeResources> renderSDSMDepthPipe;

	protected:
		virtual void onInit() override
		{
			{
				getContext()->descriptorFactoryBegin()
					.bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kCommonShaderStage, 0) // frameData
					.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCommonShaderStage, 1) // frameData
					.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCommonShaderStage, 2) // frameData
					.buildNoInfoPush(frameDataSetLayout);

				std::vector<VkDescriptorSetLayout> commonLayouts =
				{
					frameDataSetLayout,
					getContext()->getSamplerCache().getCommonDescriptorSetLayout(),
					getRenderer()->getBlueNoise().spp_1_buffer.setLayouts,
					getContext()->getBindlessTextureSetLayout(),
										getContext()->getBindlessSSBOSetLayout()
					, getContext()->getBindlessSSBOSetLayout(),
					getContext()->getDynamicUniformBuffers().getSetlayout(),

				};

				pmxPass = std::make_unique<GraphicPipeResources>(
					"shader/pmx_gbuffer.vert.spv",
					"shader/pmx_gbuffer.frag.spv",
					commonLayouts,
					0,
					std::vector<VkFormat>
					{
						GBufferTextures::hdrSceneColorFormat(),
						GBufferTextures::gbufferAFormat(),
						GBufferTextures::gbufferBFormat(),
						GBufferTextures::gbufferSFormat(),
						GBufferTextures::gbufferVFormat(),
						GBufferTextures::getIdTextureFormat(),
						GBufferTextures::gbufferUpscaleReactiveFormat(),
						GBufferTextures::gbufferUpscaleTranslucencyAndCompositionFormat(),
					},
					std::vector<VkPipelineColorBlendAttachmentState>
					{
						RHIColorBlendAttachmentOpauqeState(),
						RHIColorBlendAttachmentOpauqeState(),
						RHIColorBlendAttachmentOpauqeState(),
						RHIColorBlendAttachmentOpauqeState(),
						RHIColorBlendAttachmentOpauqeState(),
						RHIColorBlendAttachmentOpauqeState(),
						RHIColorBlendAttachmentOpauqeState(),
						RHIColorBlendAttachmentOpauqeState(),
					},
					GBufferTextures::depthTextureFormat());

				pmxOutlinePass = std::make_unique<GraphicPipeResources>(
					"shader/pmx_outline_depth.vert.spv",
					"shader/pmx_outline_depth.frag.spv",
					commonLayouts,
					0,
					std::vector<VkFormat>
					{
						GBufferTextures::hdrSceneColorFormat(),
						GBufferTextures::gbufferVFormat(),
					},
					std::vector<VkPipelineColorBlendAttachmentState>
					{
						RHIColorBlendAttachmentOpauqeState(),
						RHIColorBlendAttachmentOpauqeState(),
					},
					GBufferTextures::depthTextureFormat(),
					VK_CULL_MODE_BACK_BIT);

				// Translucency blend.
				VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
				colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
				colorBlendAttachment.blendEnable = VK_TRUE;
				colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
				colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
				colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
				colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
				colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
				colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

				pmxTranslucencyPass = std::make_unique<GraphicPipeResources>(
					"shader/pmx_translucency.vert.spv",
					"shader/pmx_translucency.frag.spv",
					commonLayouts,
					0,
					std::vector<VkFormat>
					{
						GBufferTextures::hdrSceneColorFormat(),
						GBufferTextures::gbufferUpscaleReactiveFormat(),
						GBufferTextures::gbufferUpscaleTranslucencyAndCompositionFormat(),
						GBufferTextures::getIdTextureFormat(),
						GBufferTextures::gbufferVFormat(),
					},
					std::vector<VkPipelineColorBlendAttachmentState>
					{
						colorBlendAttachment,
						RHIColorBlendAttachmentOpauqeState(),
						RHIColorBlendAttachmentOpauqeState(),
						RHIColorBlendAttachmentOpauqeState(),
						RHIColorBlendAttachmentOpauqeState(),
					},
					GBufferTextures::depthTextureFormat(),
					VK_CULL_MODE_NONE,
					VK_COMPARE_OP_GREATER,
					false,
					false,
					std::vector<VkVertexInputAttributeDescription>{},
					0,
					VK_POLYGON_MODE_FILL,
					false);
			}

			{
				getContext()->descriptorFactoryBegin()
					.bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kCommonShaderStage, 0) // frameData
					.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCommonShaderStage, 1) // frameData
					.buildNoInfoPush(sdsmSetLayout);

				std::vector<VkDescriptorSetLayout> commonLayouts =
				{
					sdsmSetLayout,
					getContext()->getSamplerCache().getCommonDescriptorSetLayout(),
					getContext()->getBindlessTextureSetLayout(),
				};

				renderSDSMDepthPipe = std::make_unique<GraphicPipeResources>(
					"shader/pmx_sdsm_depth.vert.spv",
					"shader/pmx_sdsm_depth.frag.spv",
					commonLayouts,
					sizeof(PMXSDSMPushConsts),
					std::vector<VkFormat>{ },
					std::vector<VkPipelineColorBlendAttachmentState>{ },
					GBufferTextures::depthTextureFormat(),
					VK_CULL_MODE_NONE,
					VK_COMPARE_OP_GREATER,
					true,
					true,
					PMXMeshProxy::kInputAttris,
					sizeof(PMXMeshProxy::Vertex));
			}
		}

		virtual void release() override
		{
			pmxPass.reset();
			pmxTranslucencyPass.reset();
			renderSDSMDepthPipe.reset();
			pmxOutlinePass.reset();
		}
	};

	void RendererInterface::renderPMXTranslucent(
		VkCommandBuffer cmd, 
		GBufferTextures* inGBuffers, 
		RenderScene* scene, 
		BufferParameterHandle perFrameGPU)
	{
		if (!scene->isPMXExist())
		{
			return;
		}
		auto* pass = m_context->getPasses().get<PMXPass>();
		auto& hdrSceneColor = inGBuffers->hdrSceneColor->getImage();
		auto& sceneDepthZ = inGBuffers->depthTexture->getImage();
		auto& gbufferComposition = inGBuffers->gbufferUpscaleTranslucencyAndComposition->getImage();
		auto& gbufferUpscaleMask = inGBuffers->gbufferUpscaleReactive->getImage();
		auto& idTexture = inGBuffers->idTexture->getImage();
		auto& selectionMask = inGBuffers->selectionOutlineMask->getImage();
		auto& gbufferV = inGBuffers->gbufferV->getImage();

		selectionMask.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		hdrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		sceneDepthZ.transitionLayout(cmd, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));
		gbufferComposition.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		gbufferUpscaleMask.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		idTexture.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		gbufferV.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));

		std::vector<VkRenderingAttachmentInfo> colorAttachments = ColorAttachmentsBuilder()
			.add(hdrSceneColor, VK_ATTACHMENT_LOAD_OP_LOAD)
			.add(gbufferUpscaleMask, VK_ATTACHMENT_LOAD_OP_LOAD)
			.add(gbufferComposition, VK_ATTACHMENT_LOAD_OP_LOAD)
			.add(idTexture, VK_ATTACHMENT_LOAD_OP_LOAD)
			.add(gbufferV, VK_ATTACHMENT_LOAD_OP_LOAD)
			.result;

		VkRenderingAttachmentInfo depthAttachment = getDepthAttachment(sceneDepthZ, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);
		{
			ScopeRenderCmdObject renderCmdScope(cmd, "PMX translucent", sceneDepthZ, colorAttachments, depthAttachment);
			pass->pmxTranslucencyPass->bind(cmd);

			PushSetBuilder(cmd)
				.addBuffer(perFrameGPU)
				.addUAV(selectionMask)
				.addSRV(m_averageLum ? m_averageLum->getImage() : getContext()->getEngineTextureWhite()->getImage())
				.push(pass->pmxTranslucencyPass.get());

			pass->pmxTranslucencyPass->bindSet(cmd, std::vector<VkDescriptorSet>
			{
				getContext()->getSamplerCache().getCommonDescriptorSet(),
				getRenderer()->getBlueNoise().spp_1_buffer.set,
				getContext()->getBindlessTextureSet(),
					m_context->getBindlessSSBOSet()
					, m_context->getBindlessSSBOSet()
			}, 1);

			const auto& pmxes = scene->getPMXes();
			for (const auto& pmx : pmxes)
			{
				pmx.lock()->onRenderCollect(this, cmd, pass->pmxTranslucencyPass->pipelineLayout, true);
			}

		}

		selectionMask.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		hdrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		sceneDepthZ.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));
		gbufferComposition.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		gbufferUpscaleMask.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		idTexture.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		gbufferV.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));

		m_gpuTimer.getTimeStamp(cmd, "pmx translucent");
	}

	void RendererInterface::renderPMXGbuffer(
		VkCommandBuffer cmd, 
		GBufferTextures* inGBuffers, 
		RenderScene* scene, 
		BufferParameterHandle perFrameGPU)
	{
		if (!scene->isPMXExist())
		{
			return;
		}

		auto* pass = m_context->getPasses().get<PMXPass>();

		auto& hdrSceneColor = inGBuffers->hdrSceneColor->getImage();
		auto& gbufferA = inGBuffers->gbufferA->getImage();
		auto& gbufferB = inGBuffers->gbufferB->getImage();
		auto& gbufferS = inGBuffers->gbufferS->getImage();
		auto& gbufferV = inGBuffers->gbufferV->getImage();
		auto& sceneDepthZ = inGBuffers->depthTexture->getImage();
		auto& gbufferComposition = inGBuffers->gbufferUpscaleTranslucencyAndComposition->getImage();
		auto& gbufferUpscaleMask = inGBuffers->gbufferUpscaleReactive->getImage();
		auto& idTexture = inGBuffers->idTexture->getImage();
		auto& selectionMask = inGBuffers->selectionOutlineMask->getImage();

		selectionMask.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		hdrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		gbufferA.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		gbufferB.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		gbufferS.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		gbufferV.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		sceneDepthZ.transitionLayout(cmd, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));
		gbufferComposition.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		gbufferUpscaleMask.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		idTexture.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));

		std::vector<VkRenderingAttachmentInfo> colorAttachments = ColorAttachmentsBuilder()
			.add(hdrSceneColor, VK_ATTACHMENT_LOAD_OP_LOAD)
			.add(gbufferA, VK_ATTACHMENT_LOAD_OP_LOAD)
			.add(gbufferB, VK_ATTACHMENT_LOAD_OP_LOAD)
			.add(gbufferS, VK_ATTACHMENT_LOAD_OP_LOAD)
			.add(gbufferV, VK_ATTACHMENT_LOAD_OP_LOAD)
			.add(idTexture, VK_ATTACHMENT_LOAD_OP_LOAD)
			.add(gbufferUpscaleMask, VK_ATTACHMENT_LOAD_OP_LOAD)
			.add(gbufferComposition, VK_ATTACHMENT_LOAD_OP_LOAD)
			.result;
		VkRenderingAttachmentInfo depthAttachment = getDepthAttachment(sceneDepthZ, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);

		{
			ScopeRenderCmdObject renderCmdScope(cmd, "PMXGBuffer", sceneDepthZ, colorAttachments, depthAttachment);
			pass->pmxPass->bind(cmd);

			PushSetBuilder(cmd)
				.addBuffer(perFrameGPU)
				.addUAV(selectionMask)
				.addSRV(m_averageLum ? m_averageLum->getImage() : getContext()->getEngineTextureWhite()->getImage())
				.push(pass->pmxPass.get());

			pass->pmxPass->bindSet(cmd, std::vector<VkDescriptorSet>
			{
				getContext()->getSamplerCache().getCommonDescriptorSet(),
				getRenderer()->getBlueNoise().spp_1_buffer.set,
				getContext()->getBindlessTextureSet(),
					m_context->getBindlessSSBOSet()
					, m_context->getBindlessSSBOSet()
			}, 1);

			const auto& pmxes = scene->getPMXes();
			for (const auto& pmx : pmxes)
			{
				pmx.lock()->onRenderCollect(this, cmd, pass->pmxPass->pipelineLayout, false);
			}

		}

		selectionMask.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		hdrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		gbufferA.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		gbufferB.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		gbufferS.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		gbufferV.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		sceneDepthZ.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));
		gbufferComposition.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		gbufferUpscaleMask.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		idTexture.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));

		m_gpuTimer.getTimeStamp(cmd, "pmx gbuffer");
	}

	void RendererInterface::renderPMXOutline(VkCommandBuffer cmd, GBufferTextures* inGBuffers, RenderScene* scene, BufferParameterHandle perFrameGPU)
	{
		if (!scene->isPMXExist())
		{
			return;
		}

		auto* pass = m_context->getPasses().get<PMXPass>();

		auto& hdrSceneColor = inGBuffers->hdrSceneColor->getImage();
		auto& gbufferV = inGBuffers->gbufferV->getImage();
		auto& sceneDepthZ = inGBuffers->depthTexture->getImage();
		auto& selectionMask = inGBuffers->selectionOutlineMask->getImage();

		selectionMask.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		hdrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		gbufferV.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		sceneDepthZ.transitionLayout(cmd, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));

		std::vector<VkRenderingAttachmentInfo> colorAttachments = ColorAttachmentsBuilder()
			.add(hdrSceneColor, VK_ATTACHMENT_LOAD_OP_LOAD)
			.add(gbufferV, VK_ATTACHMENT_LOAD_OP_LOAD)
			.result;
		VkRenderingAttachmentInfo depthAttachment = getDepthAttachment(sceneDepthZ, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);

		{
			ScopeRenderCmdObject renderCmdScope(cmd, "PMXOutline", sceneDepthZ, colorAttachments, depthAttachment);
			pass->pmxOutlinePass->bind(cmd);

			PushSetBuilder(cmd)
				.addBuffer(perFrameGPU)
				.addUAV(selectionMask)
				.addSRV(m_averageLum ? m_averageLum->getImage() : getContext()->getEngineTextureWhite()->getImage())
				.push(pass->pmxOutlinePass.get());

			pass->pmxOutlinePass->bindSet(cmd, std::vector<VkDescriptorSet>
			{
				getContext()->getSamplerCache().getCommonDescriptorSet(),
					getRenderer()->getBlueNoise().spp_1_buffer.set,
					getContext()->getBindlessTextureSet(),
					m_context->getBindlessSSBOSet()
					, m_context->getBindlessSSBOSet()
			}, 1);

			const auto& pmxes = scene->getPMXes();
			for (const auto& pmx : pmxes)
			{
				pmx.lock()->onRenderCollect(this, cmd, pass->pmxOutlinePass->pipelineLayout, false);
			}

		}

		selectionMask.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		hdrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		gbufferV.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		sceneDepthZ.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));

		m_gpuTimer.getTimeStamp(cmd, "pmx outline");
	}


	void PMXComponent::onRenderCollect(
		RendererInterface* renderer, 
		VkCommandBuffer cmd, 
		VkPipelineLayout pipelinelayout, 
		bool bTranslucentPass)
	{
		if (!m_proxy || !m_proxy->isInit())
		{
			return;
		}

		if (auto node = m_node.lock())
		{
			const auto& modelMatrix = node->getTransform()->getWorldMatrix();
			const auto& modelMatrixPrev = node->getTransform()->getPrevWorldMatrix();

			m_proxy->onRenderCollect(
				renderer, cmd, pipelinelayout, modelMatrix, modelMatrixPrev, bTranslucentPass, node->getId(),
				Editor::get()->getSceneNodeSelections().isSelected(SceneNodeSelctor(getNode())));
		}
	}


	void PMXComponent::onRenderTick(const RuntimeModuleTickData& tickData, VkCommandBuffer cmd, 
		std::vector<GPUStaticMeshPerObjectData>& collector, std::vector<VkAccelerationStructureInstanceKHR>& asInstances)
	{
		if (!m_proxy) { return; }
		if (!m_proxy->isInit()) { return; }

		if (auto node = m_node.lock())
		{
			const auto& modelMatrix = node->getTransform()->getWorldMatrix();
			const auto& modelMatrixPrev = node->getTransform()->getPrevWorldMatrix();

			m_proxy->updateAnimation(tickData.gameTime, tickData.deltaTime);

			m_proxy->updateVertex(cmd);

			if (getContext()->getGraphicsCardState().bSupportRaytrace)
			{
				m_proxy->updateBLAS(cmd);
			}


			m_proxy->collectObjectInfos(collector, asInstances, node->getId(), Editor::get()->getSceneNodeSelections().isSelected(SceneNodeSelctor(getNode())),
				modelMatrix, modelMatrixPrev);


		}
	}



	void PMXMeshProxy::onRenderCollect(
		RendererInterface* renderer, 
		VkCommandBuffer cmd, 
		VkPipelineLayout pipelinelayout, 
		const glm::mat4& modelMatrix, 
		const glm::mat4& modelMatrixPrev, 
		bool bTranslucentPass,
		uint32_t sceneNodeId,
		bool bSelected)
	{
		// then draw every submesh.
		size_t subMeshCount = m_mmdModel->GetSubMeshCount();
		for (uint32_t i = 0; i < subMeshCount; i++)
		{
			const auto& subMesh = m_mmdModel->GetSubMeshes()[i];
			const auto& material = m_pmxAsset->getMaterials().at(subMesh.m_materialID);

			if (material.bHide)
			{
				continue;
			}

			bool bShouldDraw = true;
			if (bTranslucentPass)
			{
				bShouldDraw = material.bTranslucent;
			}
			else
			{
				bShouldDraw = !material.bTranslucent;
			}

			if (!bShouldDraw)
			{
				continue;
			}

			PMXGpuParams params{};
			params.modelMatrix = modelMatrix;
			params.modelMatrixPrev = modelMatrixPrev;
			params.texId = material.mmdTex;
			params.spTexID = material.mmdSphereTex;
			params.toonTexID = material.mmdToonTex;
			params.pixelDepthOffset = material.pixelDepthOffset;
			params.sceneNodeId = sceneNodeId;
			params.shadingModel = shadingModelConvert(material.pmxShadingModel);
			params.bSelected = bSelected ? 1 : 0;
			params.indicesArrayId = m_indicesBindless;
			params.normalsArrayId = m_normalBindless;
			params.positionsArrayId = m_positionBindless;
			params.uv0sArrayId = m_uvBindless;
			params.positionsPrevArrayId = m_positionPrevBindless;

			params.translucentUnlitScale = material.translucentUnlitScale;
			params.eyeHighlightScale = material.eyeHighlightScale;

			{
				uint32_t dynamicOffset = getContext()->getDynamicUniformBuffers().alloc(sizeof(params));
				memcpy((char*)(getContext()->getDynamicUniformBuffers().getBuffer()->getMapped()) + dynamicOffset, &params, sizeof(params));
				auto set = getContext()->getDynamicUniformBuffers().getSet();
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelinelayout, 6, 1, &set, 1, &dynamicOffset);
			}

			vkCmdDraw(cmd, subMesh.m_vertexCount, 1, subMesh.m_beginIndex, 0);
		}
	}

	void PMXMeshProxy::collectObjectInfos(
		std::vector<GPUStaticMeshPerObjectData>& collector, 
		std::vector<VkAccelerationStructureInstanceKHR>& asInstances,
		uint32_t sceneNodeId,
		bool bSelected,
		const glm::mat4& modelMatrix,
		const glm::mat4& modelMatrixPrev)
	{
		const size_t objectOffsetId = collector.size();

		VkAccelerationStructureInstanceKHR instanceTamplate{};
		{
			math::mat4 temp = math::transpose(modelMatrix);
			memcpy(&instanceTamplate.transform, &temp, sizeof(VkTransformMatrixKHR));
		}

		size_t subMeshCount = m_mmdModel->GetSubMeshCount();
		for (uint32_t i = 0; i < subMeshCount; i++)
		{
			const auto& subMesh = m_mmdModel->GetSubMeshes()[i];
			const auto& material = m_pmxAsset->getMaterials().at(subMesh.m_materialID);

			if (material.bHide)
			{
				continue;
			}

			bool bShouldDraw = !material.bTranslucent && material.bCastShadow;
			if (!bShouldDraw)
			{
				continue;
			}

			GPUStaticMeshPerObjectData object{};
			object.uv0sArrayId = m_uvBindless;
			object.normalsArrayId = m_normalBindless;
			object.indicesArrayId = m_indicesBindless;
			object.positionsArrayId = m_positionBindless;
			object.objectId = sceneNodeId;
			object.indexStartPosition = subMesh.m_beginIndex;
			object.indexCount = subMesh.m_vertexCount;
			object.objectType = uint32_t(EStaticMeshType::PMXStaticMesh);
			object.positionsPrevArrayId = m_positionPrevBindless;
			object.bSelected = bSelected ? 1 : 0;
			object.modelMatrix = modelMatrix;
			object.modelMatrixPrev = modelMatrixPrev;

			object.material = GPUMaterialStandardPBR::getDefault();
			object.material.baseColorId = material.mmdTex;

			collector.push_back(object);

			if (getContext()->getGraphicsCardState().bSupportRaytrace)
			{
				VkAccelerationStructureInstanceKHR as{};
				as.accelerationStructureReference = m_blasBuilder.getBlasDeviceAddress(i);
				as.mask = 0xFF;
				as.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
				as.instanceShaderBindingTableRecordOffset = 0;
				as.transform = instanceTamplate.transform;
				as.instanceCustomIndex = objectOffsetId + i;

				asInstances.push_back(as);
			}

		}
	}

	void PMXMeshProxy::updateAnimation(float vmdFrameTime, float physicElapsed)
	{
		if (m_vmd)
		{
			m_mmdModel->BeginAnimation();
			m_mmdModel->UpdateAllAnimation(m_vmd.get(), vmdFrameTime * 30.0f, physicElapsed);
			m_mmdModel->EndAnimation();
		}
	}

	void PMXMeshProxy::updateVertex(VkCommandBuffer cmd)
	{
		// Copy last positions, todo: can optimize.
		std::vector<glm::vec3> positionLast = m_mmdModel->getUpdatePositions();

		m_mmdModel->Update();
		const glm::vec3* position = m_mmdModel->GetUpdatePositions();
		const glm::vec3* normal = m_mmdModel->GetUpdateNormals();
		const glm::vec2* uv = m_mmdModel->GetUpdateUVs();
		glm::vec3* positionLastPtr = &positionLast[0];


		// copy vertex buffer gpu. 
		m_stageBufferPosition->copyAndUpload(cmd, position, m_positionBuffer.get());
		m_stageBufferNormal->copyAndUpload(cmd, normal, m_normalBuffer.get());
		m_stageBufferUv->copyAndUpload(cmd, uv, m_uvBuffer.get());
		m_stageBufferPositionPrevFrame->copyAndUpload(cmd, positionLastPtr, m_positionPrevFrameBuffer.get());
	}

	void PMXMeshProxy::updateBLAS(VkCommandBuffer cmd)
	{
		const uint32_t maxVertex = m_mmdModel->GetVertexCount();

		size_t subMeshCount = m_mmdModel->GetSubMeshCount();

		std::vector<BLASBuilder::BlasInput> allBlas(subMeshCount);
		for (size_t i = 0; i < subMeshCount; i++)
		{
			const auto& subMesh = m_mmdModel->GetSubMeshes()[i];

			const uint32_t maxPrimitiveCount = subMesh.m_vertexCount / 3;

			// Describe buffer as array of VertexObj.
			VkAccelerationStructureGeometryTrianglesDataKHR triangles{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR };
			triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;  // vec3 vertex position data.
			triangles.vertexData.deviceAddress = m_positionBuffer->getDeviceAddress();
			triangles.vertexStride = sizeof(math::vec3);
			triangles.indexType = VK_INDEX_TYPE_UINT32;
			triangles.indexData.deviceAddress = m_indexBuffer->getDeviceAddress();
			triangles.maxVertex = maxVertex;

			// Identify the above data as containing opaque triangles.
			VkAccelerationStructureGeometryKHR asGeom{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
			asGeom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
			asGeom.flags = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
			asGeom.geometry.triangles = triangles;

			VkAccelerationStructureBuildRangeInfoKHR offset{ };
			offset.firstVertex = 0; // No vertex offset, current all vertex buffer start from zero.
			offset.primitiveCount = maxPrimitiveCount;
			offset.primitiveOffset = subMesh.m_beginIndex * sizeof(VertexIndexType);
			offset.transformOffset = 0;

			allBlas[i].asGeometry.emplace_back(asGeom);
			allBlas[i].asBuildOffsetInfo.emplace_back(offset);
		}

		if (m_blasBuilder.isInit())
		{
			m_blasBuilder.update(cmd, allBlas,
				VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR |
				VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR);
		}
		else
		{
			m_blasBuilder.build(allBlas,
				VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR |
				VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR);
		}

	}


}

