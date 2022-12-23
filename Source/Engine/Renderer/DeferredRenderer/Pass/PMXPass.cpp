#include "Pch.h"
#include "../DeferredRenderer.h"
#include "../../Renderer.h"
#include "../../RenderSceneData.h"
#include "../../../AssetSystem/MeshManager.h"
#include "../../RendererTextures.h"
#include "../../SceneTextures.h"
#include "../../../AssetSystem/TextureManager.h"
#include "../../../Scene/Component/PMXComponent.h"

namespace Flower
{


	class PMXPass : public PassInterface
	{
	public:
		VkPipeline basicDrawPipeline = VK_NULL_HANDLE;
		VkPipelineLayout basicDrawPipelineLayout = VK_NULL_HANDLE;

	protected:
		virtual void init() override
		{
			{
				CHECK(basicDrawPipeline == VK_NULL_HANDLE);
				CHECK(basicDrawPipelineLayout == VK_NULL_HANDLE);

				auto vertShader = RHI::ShaderManager->getShader("PMX_LightingDraw.vert.spv", true);
				auto fragShader = RHI::ShaderManager->getShader("PMX_LightingDraw.frag.spv", true);

				std::vector<VkDescriptorSetLayout> setLayouts =
				{
					  GetLayoutStatic(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC)
					, GetLayoutStatic(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) // viewData
					, GetLayoutStatic(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) // frameData
					, RHI::SamplerManager->getCommonDescriptorSetLayout() // sampler
					, BlueNoiseMisc::getSetLayout() // Bluenoise
					, StaticTexturesManager::get()->globalBlueNoise.spp_1_buffer.setLayouts // All blue noise set layout is same.
					, Bindless::Texture->getSetLayout() // texture2D array
				};

				std::vector<VkPipelineShaderStageCreateInfo> shaderStages =
				{
					RHIPipelineShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, vertShader),
					RHIPipelineShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, fragShader),
				};

				std::vector<VkFormat> colorAttachmentFormats =
				{
					RTFormats::hdrSceneColor(),
					RTFormats::gbufferA(),
					RTFormats::gbufferB(),
					RTFormats::gbufferS(),
					RTFormats::gbufferV(),
					RTFormats::gbufferUpscaleReactive(),
					RTFormats::gbufferUpscaleTranslucencyAndComposition(),
				};

				std::vector<VkPipelineColorBlendAttachmentState> attachmentBlends =
				{
					RHIColorBlendAttachmentOpauqeState(),
					RHIColorBlendAttachmentOpauqeState(),
					RHIColorBlendAttachmentOpauqeState(),
					RHIColorBlendAttachmentOpauqeState(),
					RHIColorBlendAttachmentOpauqeState(),
					RHIColorBlendAttachmentOpauqeState(),
					RHIColorBlendAttachmentOpauqeState(),
				};

				const VkPipelineRenderingCreateInfo pipelineRenderingCreateInfo
				{
					.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
					.colorAttachmentCount = (uint32_t)colorAttachmentFormats.size(),
					.pColorAttachmentFormats = colorAttachmentFormats.data(),
					.depthAttachmentFormat = RTFormats::depth(),
				};
				VkPipelineColorBlendStateCreateInfo colorBlending
				{
					.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
					.logicOpEnable = VK_FALSE,
					.logicOp = VK_LOGIC_OP_COPY,
					.attachmentCount = uint32_t(attachmentBlends.size()),
					.pAttachments = attachmentBlends.data(),
				};

				auto defaultViewport = RHIDefaultViewportState();
				const auto& deafultDynamicState = RHIDefaultDynamicStateCreateInfo();


				auto vertexInputState = RHIVertexInputStateCreateInfo();

				std::vector<VkVertexInputAttributeDescription> inputAttributes = { 
					{ 0, 0, VK_FORMAT_R32G32B32_SFLOAT, sizeof(float) * 0 }, // pos
					{ 1, 0, VK_FORMAT_R32G32B32_SFLOAT, sizeof(float) * 3 }, // normal
					{ 2, 0, VK_FORMAT_R32G32_SFLOAT,    sizeof(float) * 6 }, // uv0
					{ 3, 0, VK_FORMAT_R32G32B32_SFLOAT, sizeof(float) * 8 }, // pos prev.
				};
				VkVertexInputBindingDescription inputBindingDes =
				{
					.binding = 0,
					.stride = sizeof(PMXMeshProxy::Vertex),
					.inputRate = VK_VERTEX_INPUT_RATE_VERTEX
				};
				vertexInputState.vertexAttributeDescriptionCount = (uint32_t)inputAttributes.size();
				vertexInputState.vertexBindingDescriptionCount = 1;
				vertexInputState.pVertexBindingDescriptions = &inputBindingDes;
				vertexInputState.pVertexAttributeDescriptions = inputAttributes.data();


				auto assemblyCreateInfo = RHIInputAssemblyCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
				auto rasterState = RHIRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL);
				rasterState.cullMode = VK_CULL_MODE_FRONT_BIT;
				auto multiSampleState = RHIMultisamplingStateCreateInfo();
				auto depthStencilState = RHIDepthStencilCreateInfo(true, true, VK_COMPARE_OP_GREATER); // Reverse z.

				VkPipelineLayoutCreateInfo plci = RHIPipelineLayoutCreateInfo();

				plci.setLayoutCount = (uint32_t)setLayouts.size();
				plci.pSetLayouts = setLayouts.data();

				basicDrawPipelineLayout = RHI::get()->createPipelineLayout(plci);

				VkGraphicsPipelineCreateInfo pipelineCreateInfo
				{
					.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
					.pNext = &pipelineRenderingCreateInfo,
					.stageCount = uint32_t(shaderStages.size()),
					.pStages = shaderStages.data(),
					.pVertexInputState = &vertexInputState,
					.pInputAssemblyState = &assemblyCreateInfo,
					.pViewportState = &defaultViewport,
					.pRasterizationState = &rasterState,
					.pMultisampleState = &multiSampleState,
					.pDepthStencilState = &depthStencilState,
					.pColorBlendState = &colorBlending,
					.pDynamicState = &deafultDynamicState,
					.layout = basicDrawPipelineLayout,
				};
				RHICheck(vkCreateGraphicsPipelines(RHI::Device, nullptr, 1, &pipelineCreateInfo, nullptr, &basicDrawPipeline));
			}
		}

		virtual void release() override
		{
			RHISafeRelease(basicDrawPipeline);
			RHISafeRelease(basicDrawPipelineLayout);
		}

	};


	void DeferredRenderer::renderPMX(
		VkCommandBuffer cmd, 
		Renderer* renderer, 
		SceneTextures* inTextures, 
		RenderSceneData* scene, 
		BufferParamRefPointer& viewData, 
		BufferParamRefPointer& frameData, 
		BlueNoiseMisc& inBlueNoise)
	{
		if (!scene->isPMXExist())
		{
			return;
		}

		auto& hdrSceneColor = inTextures->getHdrSceneColor()->getImage();
		auto& gbufferA = inTextures->getGbufferA()->getImage();
		auto& gbufferB = inTextures->getGbufferB()->getImage();
		auto& gbufferS = inTextures->getGbufferS()->getImage();
		auto& gbufferV = inTextures->getGbufferV()->getImage();
		auto& sceneDepthZ = inTextures->getDepth()->getImage();
		auto& gbufferComposition = inTextures->getGbufferUpscaleTranslucencyAndComposition()->getImage();
		auto& gbufferUpscaleMask = inTextures->getGbufferUpscaleReactive()->getImage();

		auto rtsLayout2Attachment = [&]()
		{
			hdrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
			gbufferA.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
			gbufferB.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
			gbufferS.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
			gbufferV.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
			sceneDepthZ.transitionLayout(cmd, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));

			gbufferComposition.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
			gbufferUpscaleMask.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		};
	
		std::vector<VkRenderingAttachmentInfo> colorAttachments =
		{
			// Hdr scene color.
			RHIRenderingAttachmentInfo(
				inTextures->getHdrSceneColor()->getImage().getView(buildBasicImageSubresource()),
				VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE,
				VkClearValue{.color = {0.0f, 0.0f, 0.0f, 0.0f}}),

			// Gbuffer A
			RHIRenderingAttachmentInfo(
				inTextures->getGbufferA()->getImage().getView(buildBasicImageSubresource()),
				VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE,
				VkClearValue{.color = {0.0f, 0.0f, 0.0f, 0.0f}}),

			// Gbuffer B
			RHIRenderingAttachmentInfo(
				inTextures->getGbufferB()->getImage().getView(buildBasicImageSubresource()),
				VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE,
				VkClearValue{.color = {0.0f, 0.0f, 0.0f, 0.0f}}),

			// Gbuffer S
			RHIRenderingAttachmentInfo(
				inTextures->getGbufferS()->getImage().getView(buildBasicImageSubresource()),
				VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE,
				VkClearValue{.color = {0.0f, 0.0f, 0.0f, 0.0f}}),

			// Gbuffer V
			RHIRenderingAttachmentInfo(
				inTextures->getGbufferV()->getImage().getView(buildBasicImageSubresource()),
				VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE,
				VkClearValue{.color = {0.0f, 0.0f, 0.0f, 0.0f}}),

			RHIRenderingAttachmentInfo(
				inTextures->getGbufferUpscaleReactive()->getImage().getView(buildBasicImageSubresource()),
				VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE,
				VkClearValue{.color = {0.0f, 0.0f, 0.0f, 0.0f}}),

			RHIRenderingAttachmentInfo(
				inTextures->getGbufferUpscaleTranslucencyAndComposition()->getImage().getView(buildBasicImageSubresource()),
				VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE,
				VkClearValue{.color = {0.0f, 0.0f, 0.0f, 0.0f}}),
		};

		VkRenderingAttachmentInfo depthAttachment = RHIRenderingAttachmentInfo(
			inTextures->getDepth()->getImage().getView(RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT)),
			VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
			VK_ATTACHMENT_LOAD_OP_LOAD, 
			VK_ATTACHMENT_STORE_OP_STORE,
			VkClearValue{ .depthStencil = {0.0f, 1} }
		);

		uint32_t renderWidth = inTextures->getHdrSceneColor()->getImage().getExtent().width;
		uint32_t renderHeight = inTextures->getHdrSceneColor()->getImage().getExtent().height;

		const VkRenderingInfo renderInfo
		{
			.sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR,
			.renderArea = VkRect2D{.offset {0,0}, .extent {renderWidth, renderHeight}},
			.layerCount = 1,
			.colorAttachmentCount = uint32_t(colorAttachments.size()),
			.pColorAttachments = colorAttachments.data(),
			.pDepthAttachment = &depthAttachment,
		};

		VkRect2D scissor{ .offset{ 0,0 }, .extent {renderWidth, renderHeight} };
		VkViewport viewport
		{
			.x = 0.0f, .y = (float)m_renderHeight,
			.width = (float)renderWidth, .height = -(float)renderHeight,
			.minDepth = 0.0f, .maxDepth = 1.0f,
		};

		rtsLayout2Attachment();
		RHI::ScopePerframeMarker marker(cmd, "pmx", { 1.0f, 0.0f, 0.0f, 1.0f });

		auto* pass = getPasses()->getPass<PMXPass>();

		std::vector<VkDescriptorSet> meshPassSets =
		{
			  viewData->buffer.getSet()  // viewData
			, frameData->buffer.getSet() // frameData
			, RHI::SamplerManager->getCommonDescriptorSet() // samplers.
			, inBlueNoise.getSet()
			, StaticTexturesManager::get()->globalBlueNoise.spp_1_buffer.set // 1spp is good.
			, Bindless::Texture->getSet()
		};

		// offset from #1
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pass->basicDrawPipelineLayout,
			1, (uint32_t)meshPassSets.size(), meshPassSets.data(), 0, nullptr);

		
		vkCmdBeginRendering(cmd, &renderInfo);
		{
			vkCmdSetScissor(cmd, 0, 1, &scissor);
			vkCmdSetViewport(cmd, 0, 1, &viewport);
			vkCmdSetDepthBias(cmd, 0, 0, 0);

			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pass->basicDrawPipeline);

			// Draw pmx.
			const auto& pmxes = scene->getPMXes();
			for (const auto& pmx : pmxes)
			{
				pmx->onRenderCollect(this, cmd, pass->basicDrawPipelineLayout);
			}

			m_gpuTimer.getTimeStamp(cmd, "pmx Rendering");
		}
		vkCmdEndRendering(cmd);

		hdrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
		gbufferA.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
		gbufferB.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
		gbufferS.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
		gbufferV.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
		inTextures->getDepth()->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));

		gbufferComposition.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
		gbufferUpscaleMask.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
	}
}