#include "Pch.h"
#include "../DeferredRenderer.h"
#include "../../Renderer.h"
#include "../../RenderSceneData.h"
#include "../../SceneTextures.h"
#include "../../../AssetSystem/MeshManager.h"
#include "../../RendererTextures.h"
#include "../../../AssetSystem/TextureManager.h"

namespace Flower
{
	// Sample distribution shadow map implement here.

	struct GPUDepthRange
	{
		uint32_t minDepth;
		uint32_t maxDepth;
	};

	struct CascadeCullingPushConst
	{
		uint32_t cullCountPercascade;
		uint32_t cascadeCount;
	};

	struct DepthDrawPushConst
	{
		uint32_t cascadeId;
		uint32_t perCascadeMaxCount;
	};

	class SDSMPass : public PassInterface
	{
	public:
		// Common set layout.
		VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;

		VkPipeline depthRangePipeline = VK_NULL_HANDLE;
		VkPipelineLayout depthRangePipelineLayout = VK_NULL_HANDLE;
		VkPipeline cascadeBuildPipeline = VK_NULL_HANDLE;
		VkPipelineLayout cascadeBuildPipelineLayout = VK_NULL_HANDLE;
		VkPipeline cullingPipeline = VK_NULL_HANDLE;
		VkPipelineLayout cullingPipelineLayout = VK_NULL_HANDLE;
		VkPipeline depthRenderPipeline = VK_NULL_HANDLE;
		VkPipelineLayout depthRenderPipelineLayout = VK_NULL_HANDLE;
		VkPipeline softShadowEvaluatePipeline = VK_NULL_HANDLE;
		VkPipelineLayout softShadowEvaluatePipelineLayout = VK_NULL_HANDLE;

	protected:
		virtual void init() override
		{
			RHI::get()->descriptorFactoryBegin()
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, GCommonShaderStage, 0) // inDepth
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, GCommonShaderStage, 1) // SSBODepthRangeBuffer
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, GCommonShaderStage, 2) // SSBOCascadeInfoBuffer
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, GCommonShaderStage, 3) // inGbufferA
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, GCommonShaderStage, 4) // inShadowDepth
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, GCommonShaderStage, 5) // inGbufferB
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, GCommonShaderStage, 6) // inGbufferS
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, GCommonShaderStage, 7) // imageShadowMask
				.buildNoInfoPush(setLayout);

			// Depth range.
			{
				CHECK(depthRangePipeline == VK_NULL_HANDLE);
				CHECK(depthRangePipelineLayout == VK_NULL_HANDLE);

				std::vector<VkDescriptorSetLayout> setLayouts =
				{
					setLayout // Owner layout.
				};
				auto shaderModule = RHI::ShaderManager->getShader("SDSMDepthRange.comp.spv", true);

				// Vulkan build functions.
				VkPipelineLayoutCreateInfo plci = RHIPipelineLayoutCreateInfo();
				plci.setLayoutCount = (uint32_t)setLayouts.size();
				plci.pSetLayouts = setLayouts.data();
				depthRangePipelineLayout = RHI::get()->createPipelineLayout(plci);

				VkPipelineShaderStageCreateInfo shaderStageCI{};
				shaderStageCI.module = shaderModule;
				shaderStageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				shaderStageCI.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				shaderStageCI.pName = "main";
				VkComputePipelineCreateInfo computePipelineCreateInfo{};
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.layout = depthRangePipelineLayout;
				computePipelineCreateInfo.flags = 0;
				computePipelineCreateInfo.stage = shaderStageCI;
				RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &depthRangePipeline));
			}

			// Cascade build.
			{
				CHECK(cascadeBuildPipeline == VK_NULL_HANDLE);
				CHECK(cascadeBuildPipelineLayout == VK_NULL_HANDLE);

				auto shaderModule = RHI::ShaderManager->getShader("SDSMPrepareCascade.comp.spv", true);
				std::vector<VkDescriptorSetLayout> setLayouts =
				{
					  setLayout // owner set.
					, GetLayoutStatic(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) // viewData
					, GetLayoutStatic(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) // frameData
				};

				// Vulkan build functions.
				VkPipelineLayoutCreateInfo plci = RHIPipelineLayoutCreateInfo();
				plci.setLayoutCount = (uint32_t)setLayouts.size();
				plci.pSetLayouts = setLayouts.data();
				cascadeBuildPipelineLayout = RHI::get()->createPipelineLayout(plci);
				VkPipelineShaderStageCreateInfo shaderStageCI{};
				shaderStageCI.module = shaderModule;
				shaderStageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				shaderStageCI.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				shaderStageCI.pName = "main";
				VkComputePipelineCreateInfo computePipelineCreateInfo{};
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.layout = cascadeBuildPipelineLayout;
				computePipelineCreateInfo.flags = 0;
				computePipelineCreateInfo.stage = shaderStageCI;
				RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &cascadeBuildPipeline));
			}

			// Culling.
			{
				CHECK(cullingPipelineLayout == VK_NULL_HANDLE);
				CHECK(cullingPipeline == VK_NULL_HANDLE);

				auto shaderModule = RHI::ShaderManager->getShader("SDSMCulling.comp.spv", true);
				std::vector<VkDescriptorSetLayout> setLayouts =
				{
					  setLayout // owner set.
					, GetLayoutStatic(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) // objectDatas
					, GetLayoutStatic(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) // indirectCascadeCommands
					, GetLayoutStatic(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) // drawCount
				};

				// Vulkan build functions.
				VkPipelineLayoutCreateInfo plci = RHIPipelineLayoutCreateInfo();
				VkPushConstantRange pushConstant{};
				pushConstant.offset = 0;
				pushConstant.size = sizeof(CascadeCullingPushConst);
				pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				plci.pPushConstantRanges = &pushConstant;
				plci.pushConstantRangeCount = 1;
				plci.setLayoutCount = (uint32_t)setLayouts.size();
				plci.pSetLayouts = setLayouts.data();
				cullingPipelineLayout = RHI::get()->createPipelineLayout(plci);
				VkPipelineShaderStageCreateInfo shaderStageCI{};
				shaderStageCI.module = shaderModule;
				shaderStageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				shaderStageCI.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				shaderStageCI.pName = "main";
				VkComputePipelineCreateInfo computePipelineCreateInfo{};
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.layout = cullingPipelineLayout;
				computePipelineCreateInfo.flags = 0;
				computePipelineCreateInfo.stage = shaderStageCI;
				RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &cullingPipeline));
			}

			// Depth render.
			{
				CHECK(depthRenderPipeline == VK_NULL_HANDLE);
				CHECK(depthRenderPipelineLayout == VK_NULL_HANDLE);


				std::vector<VkDescriptorSetLayout> setLayouts =
				{
					  setLayout // Owner layout.
					, MeshManager::get()->getBindlessVertexBuffers()->getSetLayout() // verticesArray
					, MeshManager::get()->getBindlessIndexBuffers()->getSetLayout()  // indicesArray
					, Bindless::Texture->getSetLayout() // texture2D array
					, Bindless::Sampler->getSetLayout() // sampler2D array
					, GetLayoutStatic(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) // objectDatas
					, GetLayoutStatic(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) // indirectCascadeCommands
				};

				VkPipelineLayoutCreateInfo plci = RHIPipelineLayoutCreateInfo();
				VkPushConstantRange pushConstant{};
				pushConstant.offset = 0;
				pushConstant.size = sizeof(DepthDrawPushConst);
				pushConstant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
				plci.pPushConstantRanges = &pushConstant;
				plci.pushConstantRangeCount = 1;

				plci.setLayoutCount = (uint32_t)setLayouts.size();
				plci.pSetLayouts = setLayouts.data();
				depthRenderPipelineLayout = RHI::get()->createPipelineLayout(plci);

				auto vertShader = RHI::ShaderManager->getShader("SDSMDepth.vert.spv", true);
				auto fragShader = RHI::ShaderManager->getShader("SDSMDepth.frag.spv", true);

				std::vector<VkPipelineShaderStageCreateInfo> shaderStages =
				{
					RHIPipelineShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, vertShader),
					RHIPipelineShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, fragShader),
				};

				auto defaultViewport = RHIDefaultViewportState();
				const auto& deafultDynamicState = RHIDefaultDynamicStateCreateInfo();
				auto vertexInputState = RHIVertexInputStateCreateInfo();
				auto assemblyCreateInfo = RHIInputAssemblyCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
				auto rasterState = RHIRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL);

				// Enable depth bias to reduce shadow ances.
				rasterState.depthBiasEnable = VK_TRUE;

				// Enable depth clamp to avoid shadow hole.
				rasterState.depthClampEnable = VK_TRUE;

				rasterState.cullMode = VK_CULL_MODE_FRONT_BIT;

				auto multiSampleState = RHIMultisamplingStateCreateInfo();
				auto depthStencilState = RHIDepthStencilCreateInfo(true, true, VK_COMPARE_OP_GREATER);


				const VkPipelineRenderingCreateInfo pipelineRenderingCreateInfo
				{
					.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
					.colorAttachmentCount = 0,
					.pColorAttachmentFormats = nullptr,
					.depthAttachmentFormat = RTFormats::depth(),
				};
				VkPipelineColorBlendStateCreateInfo colorBlending
				{
					.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
					.logicOpEnable = VK_FALSE,
					.logicOp = VK_LOGIC_OP_COPY,
					.attachmentCount = 0,
					.pAttachments = nullptr,
				};
				
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
					.layout = depthRenderPipelineLayout,
				};
				RHICheck(vkCreateGraphicsPipelines(RHI::Device, nullptr, 1, &pipelineCreateInfo, nullptr, &depthRenderPipeline));
			}

			// Soft shadow.
			{
				CHECK(softShadowEvaluatePipeline == VK_NULL_HANDLE);
				CHECK(softShadowEvaluatePipelineLayout == VK_NULL_HANDLE);

				auto shaderModule = RHI::ShaderManager->getShader("SDSMEvaluateSoftShadow.comp.spv", true);
				std::vector<VkDescriptorSetLayout> setLayouts =
				{
					  setLayout // owner set.
					, GetLayoutStatic(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) // viewData
					, GetLayoutStatic(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) // frameData
					, RHI::SamplerManager->getCommonDescriptorSetLayout() // Common samplers
				};

				// Vulkan build functions.
				VkPipelineLayoutCreateInfo plci = RHIPipelineLayoutCreateInfo();
				plci.setLayoutCount = (uint32_t)setLayouts.size();
				plci.pSetLayouts = setLayouts.data();
				softShadowEvaluatePipelineLayout = RHI::get()->createPipelineLayout(plci);
				VkPipelineShaderStageCreateInfo shaderStageCI{};
				shaderStageCI.module = shaderModule;
				shaderStageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				shaderStageCI.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				shaderStageCI.pName = "main";
				VkComputePipelineCreateInfo computePipelineCreateInfo{};
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.layout = softShadowEvaluatePipelineLayout;
				computePipelineCreateInfo.flags = 0;
				computePipelineCreateInfo.stage = shaderStageCI;
				RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &softShadowEvaluatePipeline));
			}

		}

		virtual void release() override
		{
			RHISafeRelease(depthRangePipeline);
			RHISafeRelease(depthRangePipelineLayout);
			RHISafeRelease(cascadeBuildPipeline);
			RHISafeRelease(cascadeBuildPipelineLayout);
			RHISafeRelease(cullingPipeline);
			RHISafeRelease(cullingPipelineLayout);
			RHISafeRelease(depthRenderPipeline);
			RHISafeRelease(depthRenderPipelineLayout);
			RHISafeRelease(softShadowEvaluatePipeline);
			RHISafeRelease(softShadowEvaluatePipelineLayout);
		}
	};

	void DeferredRenderer::renderSDSM(
		VkCommandBuffer cmd,
		Renderer* renderer,
		SceneTextures* inTextures,
		RenderSceneData* scene,
		BufferParamRefPointer& viewData,
		BufferParamRefPointer& frameData)
	{
		if (m_cacheFrameData.bSdsmDraw <= 0)
		{
			return;
		}

		uint32_t staticMeshCount = m_cacheFrameData.staticMeshCount;
		const auto& importantLights = scene->getImportanceLights();

		auto& sceneDepthZ = inTextures->getDepth()->getImage();
		auto& gBufferA = inTextures->getGbufferA()->getImage();
		auto& gBufferB = inTextures->getGbufferB()->getImage();
		auto& gBufferS = inTextures->getGbufferS()->getImage();

		sceneDepthZ.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));
		gBufferA.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		gBufferB.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		gBufferS.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));

		auto* pass = getPasses()->getPass<SDSMPass>();

		const auto& lightInfo = importantLights.directionalLights;
		auto rangeBuffer = getBuffers()->getStaticStorageGPUOnly("SDSMRangeBuffer", sizeof(GPUDepthRange));
		auto cascadeInfoBuffer = scene->getCascadeInfoPtr();
		inTextures->allocateSDSMTexture(lightInfo.perCascadeXYDim, lightInfo.cascadeCount);
		auto SDSMDepth = inTextures->getSDSMDepth();
		auto SDSMMask = inTextures->getSDSMShadowMask();

		// Owner set.
		VkDescriptorImageInfo depthInfo = RHIDescriptorImageInfoSample(sceneDepthZ.getView(RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT)));
		VkDescriptorBufferInfo rangeBufferInfo = VkDescriptorBufferInfo{ .buffer = rangeBuffer->buffer.getBuffer()->getVkBuffer(), .offset = 0, .range = rangeBuffer->buffer.getBuffer()->getSize() };
		VkDescriptorBufferInfo cascadeBufferInfo = VkDescriptorBufferInfo{ .buffer = cascadeInfoBuffer->buffer.getBuffer()->getVkBuffer(), .offset = 0, .range = cascadeInfoBuffer->buffer.getBuffer()->getSize() };
		VkDescriptorImageInfo gbufferA = RHIDescriptorImageInfoSample(gBufferA.getView(RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT)));
		VkDescriptorImageInfo sdsmShadowDepthInfo = RHIDescriptorImageInfoSample(SDSMDepth->getImage().getView(RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT)));
		VkDescriptorImageInfo gbufferB = RHIDescriptorImageInfoSample(gBufferB.getView(RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT)));
		VkDescriptorImageInfo gbufferS = RHIDescriptorImageInfoSample(gBufferS.getView(RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT)));
		VkDescriptorImageInfo imageShadowMask = RHIDescriptorImageInfoStorage(SDSMMask->getImage().getView(RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT)));
		std::vector<VkWriteDescriptorSet> writes
		{
			RHIPushWriteDescriptorSetImage(0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &depthInfo),
			RHIPushWriteDescriptorSetBuffer(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &rangeBufferInfo),
			RHIPushWriteDescriptorSetBuffer(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &cascadeBufferInfo),
			RHIPushWriteDescriptorSetImage(3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &gbufferA),
			RHIPushWriteDescriptorSetImage(4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &sdsmShadowDepthInfo),
			RHIPushWriteDescriptorSetImage(5, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &gbufferB),
			RHIPushWriteDescriptorSetImage(6, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &gbufferS),
			RHIPushWriteDescriptorSetImage(7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &imageShadowMask),
		};

		// Compute depth range.
		{
			RHI::ScopePerframeMarker depthRangeComputeMarker(cmd, "DepthRangeCompute", { 1.0f, 0.0f, 0.0f, 1.0f });

			GPUDepthRange clearRangeValue = { .minDepth = ~0u, .maxDepth = 0u };
			vkCmdUpdateBuffer(cmd, *rangeBuffer->buffer.getBuffer(), 0, rangeBuffer->buffer.getBuffer()->getSize(), &clearRangeValue);
			std::array<VkBufferMemoryBarrier2, 1> fillBarriers
			{
				RHIBufferBarrier(rangeBuffer->buffer.getBuffer()->getVkBuffer(),
					VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT),
			};
			RHIPipelineBarrier(cmd, 0, (uint32_t)fillBarriers.size(), fillBarriers.data(), 0, nullptr);

			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->depthRangePipeline);

			// Push owner set #0.
			RHI::PushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->depthRangePipelineLayout, 0, uint32_t(writes.size()), writes.data());

			// Block dim is 3x3.
			vkCmdDispatch(cmd, 
				getGroupCount(sceneDepthZ.getExtent().width / 3 + 1, 8), 
				getGroupCount(sceneDepthZ.getExtent().height / 3 + 1, 8), 1);
			m_gpuTimer.getTimeStamp(cmd, "DepthRangeCompute");

			VkBufferMemoryBarrier2 endBufferBarrier = RHIBufferBarrier(rangeBuffer->buffer.getBuffer()->getVkBuffer(),
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
			RHIPipelineBarrier(cmd, 0, 1, &endBufferBarrier, 0, nullptr);
		}

		// Prepare cascade.
		{
			RHI::ScopePerframeMarker buildCascadeMarker(cmd, "PrepareCascadeInfo", { 1.0f, 0.0f, 0.0f, 1.0f });
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->cascadeBuildPipeline);

			RHI::PushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->cascadeBuildPipelineLayout, 0, uint32_t(writes.size()), writes.data());

			std::vector<VkDescriptorSet> compPassSets =
			{
				  viewData->buffer.getSet() 
				, frameData->buffer.getSet()       
			};

			// Set #1..2
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
				pass->cascadeBuildPipelineLayout, 1,
				(uint32_t)compPassSets.size(), compPassSets.data(),
				0, nullptr
			);

			vkCmdDispatch(cmd, getGroupCount(GMaxCascadePerDirectionalLight, 32), 1, 1);
			m_gpuTimer.getTimeStamp(cmd, "PrepareCascade");

			VkBufferMemoryBarrier2 endBufferBarrier = RHIBufferBarrier(cascadeInfoBuffer->buffer.getBuffer()->getVkBuffer(),
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
			RHIPipelineBarrier(cmd, 0, 1, &endBufferBarrier, 0, nullptr);
		}

		// Draw SDSM for directional lights.
		{
			const auto cullingCount = lightInfo.cascadeCount * staticMeshCount;

			auto indirectDrawCommandBuffer = getBuffers()->getIndirectStorage("SDSMMeshIndirectCommand",
				cullingCount * sizeof(GPUDrawIndirectCommand));

			auto indirectDrawCountBuffer = getBuffers()->getIndirectStorage("SDSMMeshIndirectCount",
				sizeof(GPUDrawIndirectCount) * lightInfo.cascadeCount);

			// Culling.
			{
				RHI::ScopePerframeMarker staticMeshGBufferCullingMarker(cmd, "SDSMCulling", { 1.0f, 0.0f, 0.0f, 1.0f });

				vkCmdFillBuffer(cmd, *indirectDrawCountBuffer->buffer.getBuffer(), 0, indirectDrawCountBuffer->buffer.getBuffer()->getSize(), 0u);
				vkCmdFillBuffer(cmd, *indirectDrawCommandBuffer->buffer.getBuffer(), 0, indirectDrawCommandBuffer->buffer.getBuffer()->getSize(), 0u);
				std::array<VkBufferMemoryBarrier2, 2> fillBarriers
				{
					RHIBufferBarrier(indirectDrawCommandBuffer->buffer.getBuffer()->getVkBuffer(),
						VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
						VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT),

					RHIBufferBarrier(indirectDrawCountBuffer->buffer.getBuffer()->getVkBuffer(),
						VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
						VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT),
				};
				RHIPipelineBarrier(cmd, 0, (uint32_t)fillBarriers.size(), fillBarriers.data(), 0, nullptr);

				CascadeCullingPushConst gpuPushConstant =
				{
					.cullCountPercascade = staticMeshCount,
					.cascadeCount = lightInfo.cascadeCount,
				};
				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->cullingPipeline);
				vkCmdPushConstants(cmd, pass->cullingPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(CascadeCullingPushConst), &gpuPushConstant);

				// Set #0.
				RHI::PushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->cullingPipelineLayout, 0, uint32_t(writes.size()), writes.data());
				
				std::vector<VkDescriptorSet> compPassSets =
				{
					  scene->getStaticMeshesObjectsPtr()->buffer.getSet() // objectDatas
					, indirectDrawCommandBuffer->buffer.getSet()          // indirectCommands
					, indirectDrawCountBuffer->buffer.getSet()            // drawCount
				};

				// Set #1..3
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
					pass->cullingPipelineLayout, 1,
					(uint32_t)compPassSets.size(), compPassSets.data(),
					0, nullptr
				);
				vkCmdDispatch(cmd, getGroupCount(cullingCount, 64), 1, 1);

				m_gpuTimer.getTimeStamp(cmd, "SDSM Culling");

				// End buffer barrier.
				std::array<VkBufferMemoryBarrier2, 2> endBufferBarriers
				{
					RHIBufferBarrier(indirectDrawCommandBuffer->buffer.getBuffer()->getVkBuffer(),
						VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
						VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT),

					RHIBufferBarrier(indirectDrawCountBuffer->buffer.getBuffer()->getVkBuffer(),
						VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
						VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT),
				};
				RHIPipelineBarrier(cmd, 0, (uint32_t)endBufferBarriers.size(), endBufferBarriers.data(), 0, nullptr);
			}

			// Render Depth.
			SDSMDepth->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));
			{
				RHI::ScopePerframeMarker staticMeshGBufferMarker(cmd, "SDSMShadowDepth", { 1.0f, 0.0f, 0.0f, 1.0f });

				VkRenderingAttachmentInfo depthAttachment = RHIRenderingAttachmentInfo(
					SDSMDepth->getImage().getView(RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT)),
					VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
					VK_ATTACHMENT_LOAD_OP_CLEAR,
					VK_ATTACHMENT_STORE_OP_STORE,
					VkClearValue{ .depthStencil = {0.0f, 1} }
				);


				const VkRenderingInfo renderInfo
				{
					.sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR,
					.renderArea = VkRect2D{.offset {0,0}, .extent {.width = SDSMDepth->getImage().getExtent().width, .height = SDSMDepth->getImage().getExtent().height }},
					.layerCount = 1,
					.colorAttachmentCount = 0,
					.pColorAttachments = nullptr,
					.pDepthAttachment = &depthAttachment,
				};

				vkCmdBeginRendering(cmd, &renderInfo);
				{
					vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pass->depthRenderPipeline);

					// Set #0.
					RHI::PushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pass->depthRenderPipelineLayout, 0, uint32_t(writes.size()), writes.data());

					std::vector<VkDescriptorSet> meshPassSets =
					{
						  MeshManager::get()->getBindlessVertexBuffers()->getSet() // verticesArray
						, MeshManager::get()->getBindlessIndexBuffers()->getSet() // indicesArray
						, Bindless::Texture->getSet()
						, Bindless::Sampler->getSet()
						, scene->getStaticMeshesObjectsPtr()->buffer.getSet() // objectDatas
						, indirectDrawCommandBuffer->buffer.getSet() // indirectCommands
					};
					// Set #1...#6
					vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pass->depthRenderPipelineLayout,
						1, (uint32_t)meshPassSets.size(), meshPassSets.data(), 0, nullptr);

					// Set depth bias for shadow depth rendering to avoid shadow artifact.
					vkCmdSetDepthBias(cmd, lightInfo.shadowBiasConst, 0, lightInfo.shadowBiasSlope);

					for (uint32_t cascadeIndex = 0; cascadeIndex < lightInfo.cascadeCount; cascadeIndex++)
					{
						VkRect2D scissor{};
						scissor.extent = { lightInfo.perCascadeXYDim, lightInfo.perCascadeXYDim };
						scissor.offset = { int32_t(lightInfo.perCascadeXYDim * cascadeIndex), 0 };

						VkViewport viewport{};
						viewport.minDepth = 0.0f;
						viewport.maxDepth = 1.0f;
						viewport.y = (float)lightInfo.perCascadeXYDim;
						viewport.height = -(float)lightInfo.perCascadeXYDim;
						viewport.x = (float)lightInfo.perCascadeXYDim * (float)cascadeIndex;
						viewport.width = (float)lightInfo.perCascadeXYDim;

						vkCmdSetScissor(cmd, 0, 1, &scissor);
						vkCmdSetViewport(cmd, 0, 1, &viewport);

						DepthDrawPushConst pushConst{ .cascadeId = cascadeIndex, .perCascadeMaxCount = staticMeshCount };
						vkCmdPushConstants(cmd, pass->depthRenderPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(DepthDrawPushConst), &pushConst);

						vkCmdDrawIndirectCount(cmd,
							indirectDrawCommandBuffer->buffer.getBuffer()->getVkBuffer(), 
							cascadeIndex * sizeof(GPUDrawIndirectCommand) * staticMeshCount,
							indirectDrawCountBuffer->buffer.getBuffer()->getVkBuffer(),
							cascadeIndex * sizeof(GPUDrawIndirectCount),
							staticMeshCount,
							sizeof(GPUDrawIndirectCommand)
						);
					}
					m_gpuTimer.getTimeStamp(cmd, "SDSMDepthRendering");

				}
				vkCmdEndRendering(cmd);
			}
		}

		// Evaluate soft shadow.
		SDSMDepth->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));
		SDSMMask->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		{
			RHI::ScopePerframeMarker marker(cmd, "SoftShadowEvaluate", { 1.0f, 0.0f, 0.0f, 1.0f });
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->softShadowEvaluatePipeline);

			RHI::PushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->softShadowEvaluatePipelineLayout, 0, uint32_t(writes.size()), writes.data());

			std::vector<VkDescriptorSet> compPassSets =
			{
				  viewData->buffer.getSet()
				, frameData->buffer.getSet()
				, RHI::SamplerManager->getCommonDescriptorSet()
			};

			// Set #1..2
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
				pass->softShadowEvaluatePipelineLayout, 1,
				(uint32_t)compPassSets.size(), compPassSets.data(),
				0, nullptr
			);

			vkCmdDispatch(cmd, getGroupCount(SDSMMask->getImage().getExtent().width, 8), getGroupCount(SDSMMask->getImage().getExtent().height, 8), 1);
			m_gpuTimer.getTimeStamp(cmd, "SoftShadowEvaluate");
		}
		SDSMMask->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
	}
}