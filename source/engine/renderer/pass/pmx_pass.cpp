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
					.buildNoInfoPush(frameDataSetLayout);

				std::vector<VkDescriptorSetLayout> commonLayouts =
				{
					frameDataSetLayout,
					getContext()->getSamplerCache().getCommonDescriptorSetLayout(),
					getRenderer()->getBlueNoise().spp_1_buffer.setLayouts,
					getContext()->getBindlessTextureSetLayout(),
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
					GBufferTextures::depthTextureFormat(),
					VK_CULL_MODE_FRONT_BIT,
					VK_COMPARE_OP_GREATER,
					false,
					false,
					PMXMeshProxy::kInputAttris,
					sizeof(PMXMeshProxy::Vertex));

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
					VK_CULL_MODE_FRONT_BIT,
					VK_COMPARE_OP_GREATER,
					false,
					false,
					PMXMeshProxy::kInputAttris,
					sizeof(PMXMeshProxy::Vertex),
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
					VK_CULL_MODE_FRONT_BIT,
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
				.push(pass->pmxTranslucencyPass.get());

			pass->pmxTranslucencyPass->bindSet(cmd, std::vector<VkDescriptorSet>
			{
				getContext()->getSamplerCache().getCommonDescriptorSet(),
				getRenderer()->getBlueNoise().spp_1_buffer.set,
				getContext()->getBindlessTextureSet(),
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
				.push(pass->pmxPass.get());

			pass->pmxPass->bindSet(cmd, std::vector<VkDescriptorSet>
			{
				getContext()->getSamplerCache().getCommonDescriptorSet(),
				getRenderer()->getBlueNoise().spp_1_buffer.set,
				getContext()->getBindlessTextureSet(),
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

	void PMXComponent::onRenderSDSMDepthCollect(
		VkCommandBuffer cmd, 
		BufferParameterHandle perFrameGPU, 
		GBufferTextures* inGBuffers, 
		RenderScene* scene, 
		RendererInterface* renderer, 
		SDSMInfos& sdsmInfo, 
		uint32_t cascadeId)
	{
		if (!m_proxy) { return; }
		if (!m_proxy->isInit()) { return; }

		if (auto node = m_node.lock())
		{
			const auto& modelMatrix = node->getTransform()->getWorldMatrix();
			m_proxy->onRenderSDSMDepthCollect(cmd, perFrameGPU, inGBuffers, scene, renderer, sdsmInfo, cascadeId, modelMatrix);
		}
	}

	void PMXComponent::onRenderTick(const RuntimeModuleTickData& tickData, VkCommandBuffer cmd)
	{
		if (!m_proxy) { return; }
		if (!m_proxy->isInit()) { return; }

		m_proxy->updateVertex(cmd);
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
		VkBuffer vertexBuffer = m_vertexBuffer->getVkBuffer();
		VkBuffer indexBuffer = m_indexBuffer->getVkBuffer();
		const VkDeviceSize offset = 0;
		vkCmdBindIndexBuffer(cmd, indexBuffer, 0, m_indexType);
		vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuffer, &offset);

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

			{
				uint32_t dynamicOffset = getContext()->getDynamicUniformBuffers().alloc(sizeof(params));
				memcpy((char*)(getContext()->getDynamicUniformBuffers().getBuffer()->getMapped()) + dynamicOffset, &params, sizeof(params));
				auto set = getContext()->getDynamicUniformBuffers().getSet();
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelinelayout, 4, 1, &set, 1, &dynamicOffset);
			}

			vkCmdDrawIndexed(cmd, subMesh.m_vertexCount, 1, subMesh.m_beginIndex, 0, 0);
		}
	}

	void PMXMeshProxy::updateVertex(VkCommandBuffer cmd)
	{
		size_t vtxCount = m_mmdModel->GetVertexCount();

		std::vector<glm::vec3> positionLast = m_mmdModel->getUpdatePositions();
		m_mmdModel->Update();
		const glm::vec3* position = m_mmdModel->GetUpdatePositions();
		const glm::vec3* normal = m_mmdModel->GetUpdateNormals();
		const glm::vec2* uv = m_mmdModel->GetUpdateUVs();

		glm::vec3* positionLastPtr = &positionLast[0];
		// Update vertices

		auto bufferSize = VkDeviceSize(sizeof(Vertex) * vtxCount);

		// copy vertex buffer gpu. 
		m_stageBuffer->map();
		{
			void* vbStMem = m_stageBuffer->getMapped();
			auto v = static_cast<Vertex*>(vbStMem);
			for (size_t i = 0; i < vtxCount; i++)
			{
				v->position = *position;
				v->normal = *normal;
				v->uv = *uv;
				v->positionLast = *positionLastPtr;
				v++;
				position++;
				normal++;
				uv++;
				positionLastPtr++;
			}
		}
		m_stageBuffer->unmap();

		// copy to gpu
		VkBufferCopy copyRegion{};
		copyRegion.srcOffset = 0;
		copyRegion.dstOffset = 0;
		copyRegion.size = bufferSize;
		vkCmdCopyBuffer(cmd, m_stageBuffer->getVkBuffer(), m_vertexBuffer->getVkBuffer(), 1, &copyRegion);
	}

	void PMXMeshProxy::onRenderSDSMDepthCollect(
		VkCommandBuffer cmd, 
		BufferParameterHandle perFrameGPU, 
		GBufferTextures* inGBuffers, 
		RenderScene* scene, 
		RendererInterface* renderer, 
		SDSMInfos& sdsmInfo, 
		uint32_t cascadeId,
		const glm::mat4& modelMatrix)
	{
		auto* pass = getContext()->getPasses().get<PMXPass>();
		auto& cascadeBuffer = sdsmInfo.cascadeInfoBuffer;

		VkBuffer vertexBuffer = m_vertexBuffer->getVkBuffer();
		VkBuffer indexBuffer = m_indexBuffer->getVkBuffer();
		const VkDeviceSize offset = 0;
		vkCmdBindIndexBuffer(cmd, indexBuffer, 0, m_indexType);
		vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuffer, &offset);

		pass->renderSDSMDepthPipe->bind(cmd);
		PushSetBuilder(cmd)
			.addBuffer(perFrameGPU)
			.addBuffer(cascadeBuffer)
			.push(pass->renderSDSMDepthPipe.get());

		pass->renderSDSMDepthPipe->bindSet(cmd, 
			std::vector<VkDescriptorSet>{
				getContext()->getSamplerCache().getCommonDescriptorSet(),
				getContext()->getBindlessTextureSet(),
		}, 1);

		// then draw every submesh.
		size_t subMeshCount = m_mmdModel->GetSubMeshCount();
		for (uint32_t i = 0; i < subMeshCount; i++)
		{
			const auto& subMesh = m_mmdModel->GetSubMeshes()[i];
			const auto& material = m_pmxAsset->getMaterials().at(subMesh.m_materialID);

			if (material.bHide || material.bTranslucent)
			{
				continue;
			}

			PMXSDSMPushConsts push
			{
				.modelMatrix = modelMatrix,
				.colorTexId = material.mmdTex,
				.cascadeId = cascadeId,
			};

			pass->renderSDSMDepthPipe->pushConst(cmd, &push);
			vkCmdDrawIndexed(cmd, subMesh.m_vertexCount, 1, subMesh.m_beginIndex, 0, 0);
		}
	}

}

