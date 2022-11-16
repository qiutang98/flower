#include "Pch.h"
#include "../DeferredRenderer.h"
#include "../../Renderer.h"
#include "../../RenderSceneData.h"
#include "../../SceneTextures.h"

namespace Flower
{
	
	class AtmospherePass : public PassInterface
	{
	public:
		// Common set layout.
		VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;

		VkPipeline transmittanceLutPipeline = VK_NULL_HANDLE;
		VkPipelineLayout transmittanceLutPipelineLayout = VK_NULL_HANDLE;

		VkPipeline multiScatterLutPipeline = VK_NULL_HANDLE;
		VkPipelineLayout multiScatterLutPipelineLayout = VK_NULL_HANDLE;

		VkPipeline skyViewLutPipeline = VK_NULL_HANDLE;
		VkPipelineLayout skyViewLutPipelineLayout = VK_NULL_HANDLE;

		VkPipeline froxelLutPipeline = VK_NULL_HANDLE;
		VkPipelineLayout froxelLutPipelineLayout = VK_NULL_HANDLE;

		VkPipeline compositionPipeline = VK_NULL_HANDLE;
		VkPipelineLayout compositionPipelineLayout = VK_NULL_HANDLE;

		VkPipeline envCapturePipeline = VK_NULL_HANDLE;
		VkPipelineLayout envCapturePipelineLayout = VK_NULL_HANDLE;

	protected:
		virtual void init() override
		{
			RHI::get()->descriptorFactoryBegin()
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, GCommonShaderStage, 0) // imageTransmittanceLut
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, GCommonShaderStage, 1) // inTransmittanceLut
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, GCommonShaderStage, 2) // imageSkyViewLut
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, GCommonShaderStage, 3) // inSkyViewLut
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, GCommonShaderStage, 4) // imageMultiScatterLut
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, GCommonShaderStage, 5) // inMultiScatterLut
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, GCommonShaderStage, 6) // inDepth
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, GCommonShaderStage, 7) // imageFroxelScatterLut
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, GCommonShaderStage, 8) // inFroxelScatterLut
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, GCommonShaderStage, 9) // imageHdrSceneColor
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, GCommonShaderStage, 10) // inGBufferA
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, GCommonShaderStage, 11) // inSDSMShadowDepth
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, GCommonShaderStage, 12) // SSBOCascadeInfoBuffer
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, GCommonShaderStage, 13) // imageCaptureEnv
				.buildNoInfoPush(setLayout);

			std::vector<VkDescriptorSetLayout> setLayouts =
			{
				  setLayout // Owner layout.
				, GetLayoutStatic(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)  // viewData
				, GetLayoutStatic(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)  // frameData
				, RHI::SamplerManager->getCommonDescriptorSetLayout() // Common samplers
			};

			// Transmittance compute.
			{
				CHECK(transmittanceLutPipeline == VK_NULL_HANDLE);
				CHECK(transmittanceLutPipelineLayout == VK_NULL_HANDLE);
				
				auto shaderModule = RHI::ShaderManager->getShader("AtmosphereTransmittanceLut.comp.spv", true);

				// Vulkan build functions.
				VkPipelineLayoutCreateInfo plci = RHIPipelineLayoutCreateInfo();
				plci.setLayoutCount = (uint32_t)setLayouts.size();
				plci.pSetLayouts = setLayouts.data();
				transmittanceLutPipelineLayout = RHI::get()->createPipelineLayout(plci);

				VkPipelineShaderStageCreateInfo shaderStageCI{};
				shaderStageCI.module = shaderModule;
				shaderStageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				shaderStageCI.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				shaderStageCI.pName = "main";
				VkComputePipelineCreateInfo computePipelineCreateInfo{};
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.layout = transmittanceLutPipelineLayout;
				computePipelineCreateInfo.flags = 0;
				computePipelineCreateInfo.stage = shaderStageCI;
				RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &transmittanceLutPipeline));
			}

			// Multi scatter compute.
			{
				CHECK(multiScatterLutPipeline == VK_NULL_HANDLE);
				CHECK(multiScatterLutPipelineLayout == VK_NULL_HANDLE);

				auto shaderModule = RHI::ShaderManager->getShader("AtmosphereMultiScatterLut.comp.spv", true);

				// Vulkan build functions.
				VkPipelineLayoutCreateInfo plci = RHIPipelineLayoutCreateInfo();
				plci.setLayoutCount = (uint32_t)setLayouts.size();
				plci.pSetLayouts = setLayouts.data();
				multiScatterLutPipelineLayout = RHI::get()->createPipelineLayout(plci);

				VkPipelineShaderStageCreateInfo shaderStageCI{};
				shaderStageCI.module = shaderModule;
				shaderStageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				shaderStageCI.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				shaderStageCI.pName = "main";
				VkComputePipelineCreateInfo computePipelineCreateInfo{};
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.layout = multiScatterLutPipelineLayout;
				computePipelineCreateInfo.flags = 0;
				computePipelineCreateInfo.stage = shaderStageCI;
				RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &multiScatterLutPipeline));
			}

			// SkyView lut compute.
			{
				CHECK(skyViewLutPipeline == VK_NULL_HANDLE);
				CHECK(skyViewLutPipelineLayout == VK_NULL_HANDLE);

				auto shaderModule = RHI::ShaderManager->getShader("AtmosphereSkyViewLut.comp.spv", true);

				// Vulkan build functions.
				VkPipelineLayoutCreateInfo plci = RHIPipelineLayoutCreateInfo();
				plci.setLayoutCount = (uint32_t)setLayouts.size();
				plci.pSetLayouts = setLayouts.data();
				skyViewLutPipelineLayout = RHI::get()->createPipelineLayout(plci);

				VkPipelineShaderStageCreateInfo shaderStageCI{};
				shaderStageCI.module = shaderModule;
				shaderStageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				shaderStageCI.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				shaderStageCI.pName = "main";
				VkComputePipelineCreateInfo computePipelineCreateInfo{};
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.layout = skyViewLutPipelineLayout;
				computePipelineCreateInfo.flags = 0;
				computePipelineCreateInfo.stage = shaderStageCI;
				RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &skyViewLutPipeline));
			}

			// Froxel lut compute.
			{
				CHECK(froxelLutPipeline == VK_NULL_HANDLE);
				CHECK(froxelLutPipelineLayout == VK_NULL_HANDLE);

				auto shaderModule = RHI::ShaderManager->getShader("AtmosphereFroxelLut.comp.spv", true);

				// Vulkan build functions.
				VkPipelineLayoutCreateInfo plci = RHIPipelineLayoutCreateInfo();
				plci.setLayoutCount = (uint32_t)setLayouts.size();
				plci.pSetLayouts = setLayouts.data();
				froxelLutPipelineLayout = RHI::get()->createPipelineLayout(plci);

				VkPipelineShaderStageCreateInfo shaderStageCI{};
				shaderStageCI.module = shaderModule;
				shaderStageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				shaderStageCI.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				shaderStageCI.pName = "main";
				VkComputePipelineCreateInfo computePipelineCreateInfo{};
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.layout = froxelLutPipelineLayout;
				computePipelineCreateInfo.flags = 0;
				computePipelineCreateInfo.stage = shaderStageCI;
				RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &froxelLutPipeline));
			}

			// Composition pipeline compute.
			{
				CHECK(compositionPipeline == VK_NULL_HANDLE);
				CHECK(compositionPipelineLayout == VK_NULL_HANDLE);

				auto shaderModule = RHI::ShaderManager->getShader("AtmosphereComposition.comp.spv", true);

				// Vulkan build functions.
				VkPipelineLayoutCreateInfo plci = RHIPipelineLayoutCreateInfo();
				plci.setLayoutCount = (uint32_t)setLayouts.size();
				plci.pSetLayouts = setLayouts.data();
				compositionPipelineLayout = RHI::get()->createPipelineLayout(plci);

				VkPipelineShaderStageCreateInfo shaderStageCI{};
				shaderStageCI.module = shaderModule;
				shaderStageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				shaderStageCI.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				shaderStageCI.pName = "main";
				VkComputePipelineCreateInfo computePipelineCreateInfo{};
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.layout = compositionPipelineLayout;
				computePipelineCreateInfo.flags = 0;
				computePipelineCreateInfo.stage = shaderStageCI;
				RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &compositionPipeline));
			}

			// Env capture pipeline compute.
			{
				CHECK(envCapturePipeline == VK_NULL_HANDLE);
				CHECK(envCapturePipelineLayout == VK_NULL_HANDLE);

				auto shaderModule = RHI::ShaderManager->getShader("AtmosphereEnvironmentCapture.comp.spv", true);

				// Vulkan build functions.
				VkPipelineLayoutCreateInfo plci = RHIPipelineLayoutCreateInfo();
				plci.setLayoutCount = (uint32_t)setLayouts.size();
				plci.pSetLayouts = setLayouts.data();
				envCapturePipelineLayout = RHI::get()->createPipelineLayout(plci);

				VkPipelineShaderStageCreateInfo shaderStageCI{};
				shaderStageCI.module = shaderModule;
				shaderStageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				shaderStageCI.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				shaderStageCI.pName = "main";
				VkComputePipelineCreateInfo computePipelineCreateInfo{};
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.layout = envCapturePipelineLayout;
				computePipelineCreateInfo.flags = 0;
				computePipelineCreateInfo.stage = shaderStageCI;
				RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &envCapturePipeline));
			}
		}

		virtual void release() override
		{
			RHISafeRelease(transmittanceLutPipeline);
			RHISafeRelease(transmittanceLutPipelineLayout);

			RHISafeRelease(multiScatterLutPipeline);
			RHISafeRelease(multiScatterLutPipelineLayout);

			RHISafeRelease(skyViewLutPipeline);
			RHISafeRelease(skyViewLutPipelineLayout);

			RHISafeRelease(froxelLutPipeline);
			RHISafeRelease(froxelLutPipelineLayout);

			RHISafeRelease(compositionPipeline);
			RHISafeRelease(compositionPipelineLayout);

			RHISafeRelease(envCapturePipeline);
			RHISafeRelease(envCapturePipelineLayout);
		}
	};

	void DeferredRenderer::renderAtmosphere(
		VkCommandBuffer cmd, 
		Renderer* renderer, 
		SceneTextures* inTextures, 
		RenderSceneData* scene, 
		BufferParamRefPointer& viewData, 
		BufferParamRefPointer& frameData,
		bool bComposite)
	{
		// Skip if no directional light.
		if (scene->getImportanceLights().directionalLightCount <= 0)
		{
			return;
		}

		auto& tansmittanceLut = inTextures->getAtmosphereTransmittance()->getImage();
		auto& skyViewLut = inTextures->getAtmosphereSkyView()->getImage();
		auto& multiScatterLut = inTextures->getAtmosphereMultiScatter()->getImage();
		auto& froxelScatterLut = inTextures->getAtmosphereFroxelScatter()->getImage();
		auto& envCapture = inTextures->getAtmosphereEnvCapture()->getImage();

		auto& sceneDepthZ = inTextures->getDepth()->getImage();
		auto& sceneColorHdr = inTextures->getHdrSceneColor()->getImage();
		auto& gbufferA = inTextures->getGbufferA()->getImage();
		auto& sdsmShadowDepth = (m_cacheFrameData.bSdsmDraw > 0) ? inTextures->getSDSMDepth()->getImage() : inTextures->getDepth()->getImage();
		auto cascadeInfoBuffer = scene->getCascadeInfoPtr();

		auto* pass = getPasses()->getPass<AtmospherePass>();

		VkDescriptorImageInfo tansmittanceLutImageInfo = RHIDescriptorImageInfoStorage(tansmittanceLut.getView(buildBasicImageSubresource()));
		VkDescriptorImageInfo tansmittanceLutInfo = RHIDescriptorImageInfoSample(tansmittanceLut.getView(buildBasicImageSubresource()));
		VkDescriptorImageInfo skyViewLutImageInfo = RHIDescriptorImageInfoStorage(skyViewLut.getView(buildBasicImageSubresource()));
		VkDescriptorImageInfo skyViewLutInfo = RHIDescriptorImageInfoSample(skyViewLut.getView(buildBasicImageSubresource()));
		VkDescriptorImageInfo multiScatterLutImageInfo = RHIDescriptorImageInfoStorage(multiScatterLut.getView(buildBasicImageSubresource()));
		VkDescriptorImageInfo multiScatterLutInfo = RHIDescriptorImageInfoSample(multiScatterLut.getView(buildBasicImageSubresource()));
		VkDescriptorImageInfo sceneDepthZInfo = RHIDescriptorImageInfoSample(sceneDepthZ.getView(RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT)));
		VkDescriptorImageInfo froxelScatterImageInfo = RHIDescriptorImageInfoStorage(froxelScatterLut.getView(buildBasicImageSubresource(), VK_IMAGE_VIEW_TYPE_3D));
		VkDescriptorImageInfo froxelScatterLutInfo = RHIDescriptorImageInfoSample(froxelScatterLut.getView(buildBasicImageSubresource(), VK_IMAGE_VIEW_TYPE_3D));
		VkDescriptorImageInfo hdrSceneColorImageInfo = RHIDescriptorImageInfoStorage(sceneColorHdr.getView(buildBasicImageSubresource()));
		VkDescriptorImageInfo gbufferAInfo = RHIDescriptorImageInfoSample(gbufferA.getView(buildBasicImageSubresource()));
		VkDescriptorImageInfo sdsmDepthInfo = RHIDescriptorImageInfoSample(sdsmShadowDepth.getView(RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT)));
		VkDescriptorBufferInfo cascadeBufferInfo = VkDescriptorBufferInfo{ .buffer = cascadeInfoBuffer->buffer.getBuffer()->getVkBuffer(), .offset = 0, .range = cascadeInfoBuffer->buffer.getBuffer()->getSize() };

		auto captureViewRange = VkImageSubresourceRange{ 
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, 
			.baseMipLevel = 0, 
			.levelCount = 1, 
			.baseArrayLayer = 0,  
			.layerCount = 6 };

		VkDescriptorImageInfo envCaptureImageInfo = RHIDescriptorImageInfoStorage(envCapture.getView(captureViewRange, VK_IMAGE_VIEW_TYPE_CUBE));

		std::vector<VkWriteDescriptorSet> writes
		{
			RHIPushWriteDescriptorSetImage(0,  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &tansmittanceLutImageInfo),
			RHIPushWriteDescriptorSetImage(1,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &tansmittanceLutInfo),
			RHIPushWriteDescriptorSetImage(2,  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &skyViewLutImageInfo),
			RHIPushWriteDescriptorSetImage(3,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &skyViewLutInfo),
			RHIPushWriteDescriptorSetImage(4,  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &multiScatterLutImageInfo),
			RHIPushWriteDescriptorSetImage(5,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &multiScatterLutInfo),
			RHIPushWriteDescriptorSetImage(6,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &sceneDepthZInfo),
			RHIPushWriteDescriptorSetImage(7,  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &froxelScatterImageInfo),
			RHIPushWriteDescriptorSetImage(8,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &froxelScatterLutInfo),
			RHIPushWriteDescriptorSetImage(9,  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &hdrSceneColorImageInfo),
			RHIPushWriteDescriptorSetImage(10, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &gbufferAInfo),
			RHIPushWriteDescriptorSetImage(11, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &sdsmDepthInfo),
			RHIPushWriteDescriptorSetBuffer(12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &cascadeBufferInfo),
			RHIPushWriteDescriptorSetImage(13,  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &envCaptureImageInfo),
		};	

		std::vector<VkDescriptorSet> compPassSets =
		{
			  viewData->buffer.getSet()
			, frameData->buffer.getSet()
			, RHI::SamplerManager->getCommonDescriptorSet()
		};

		if (!bComposite)
		{
			RHI::ScopePerframeMarker atmosphereMarker(cmd, "Atmosphere", { 1.0f, 1.0f, 0.0f, 1.0f });

			// Pass #0. tansmittance lut.
			{
				tansmittanceLut.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
				sceneDepthZ.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));

				RHI::ScopePerframeMarker transmittanceMarker(cmd, "TransmittanceLut", { 1.0f, 1.0f, 0.0f, 1.0f });
				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->transmittanceLutPipeline);

				// Push owner set #0.
				RHI::PushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->transmittanceLutPipelineLayout, 0, uint32_t(writes.size()), writes.data());

				// Set #1..3
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
					pass->transmittanceLutPipelineLayout, 1,
					(uint32_t)compPassSets.size(), compPassSets.data(),
					0, nullptr
				);

				vkCmdDispatch(cmd, getGroupCount(tansmittanceLut.getExtent().width, 8), getGroupCount(tansmittanceLut.getExtent().height, 8), 1);
				tansmittanceLut.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
			}

			// Pass #1. multi scatter lut.
			{
				multiScatterLut.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

				RHI::ScopePerframeMarker marker(cmd, "MultiScatterLut", { 1.0f, 1.0f, 0.0f, 1.0f });
				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->multiScatterLutPipeline);

				// Push owner set #0.
				RHI::PushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->multiScatterLutPipelineLayout, 0, uint32_t(writes.size()), writes.data());

				// Set #1..3
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
					pass->multiScatterLutPipelineLayout, 1,
					(uint32_t)compPassSets.size(), compPassSets.data(),
					0, nullptr
				);

				vkCmdDispatch(cmd, multiScatterLut.getExtent().width, multiScatterLut.getExtent().height, 1);
				multiScatterLut.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
			}

			// Pass #2. sky view lut.
			{
				skyViewLut.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

				RHI::ScopePerframeMarker marker(cmd, "SkyVIewLut", { 1.0f, 1.0f, 0.0f, 1.0f });
				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->skyViewLutPipeline);

				// Push owner set #0.
				RHI::PushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->skyViewLutPipelineLayout, 0, uint32_t(writes.size()), writes.data());

				// Set #1..3
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
					pass->skyViewLutPipelineLayout, 1,
					(uint32_t)compPassSets.size(), compPassSets.data(),
					0, nullptr
				);

				vkCmdDispatch(cmd, getGroupCount(skyViewLut.getExtent().width, 8), getGroupCount(skyViewLut.getExtent().height, 8), 1);
				skyViewLut.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
			}

			// Pass #3. froxel lut.
			{
				froxelScatterLut.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
				RHI::ScopePerframeMarker marker(cmd, "FroxelLut", { 1.0f, 1.0f, 0.0f, 1.0f });
				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->froxelLutPipeline);

				// Push owner set #0.
				RHI::PushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->froxelLutPipelineLayout, 0, uint32_t(writes.size()), writes.data());

				// Set #1..3
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
					pass->froxelLutPipelineLayout, 1,
					(uint32_t)compPassSets.size(), compPassSets.data(),
					0, nullptr
				);

				vkCmdDispatch(cmd, getGroupCount(froxelScatterLut.getExtent().width, 8), getGroupCount(froxelScatterLut.getExtent().height, 8), froxelScatterLut.getExtent().depth);
				froxelScatterLut.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
			}

			// Capture pass.
			{
				envCapture.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, captureViewRange);
				RHI::ScopePerframeMarker marker(cmd, "EnvCapture", { 1.0f, 1.0f, 0.0f, 1.0f });
				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->envCapturePipeline);

				// Push owner set #0.
				RHI::PushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->envCapturePipelineLayout, 0, uint32_t(writes.size()), writes.data());

				// Set #1..3
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
					pass->envCapturePipelineLayout, 1,
					(uint32_t)compPassSets.size(), compPassSets.data(),
					0, nullptr
				);

				vkCmdDispatch(cmd, getGroupCount(envCapture.getExtent().width, 8), getGroupCount(envCapture.getExtent().height, 8), 6);
				envCapture.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, captureViewRange);
			}

			m_gpuTimer.getTimeStamp(cmd, "SkyPrepare");
		}
		else
		{
			// Pass #4. composite.
			sceneColorHdr.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

			RHI::ScopePerframeMarker marker(cmd, "SceneColorHdrLut", { 1.0f, 1.0f, 0.0f, 1.0f });
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->compositionPipeline);
			// Push owner set #0.
			RHI::PushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->compositionPipelineLayout, 0, uint32_t(writes.size()), writes.data());

			// Set #1..3
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
				pass->compositionPipelineLayout, 1,
				(uint32_t)compPassSets.size(), compPassSets.data(),
				0, nullptr
			);

			vkCmdDispatch(cmd, getGroupCount(sceneColorHdr.getExtent().width, 8), getGroupCount(sceneColorHdr.getExtent().height, 8), 1);
			sceneColorHdr.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());


			m_gpuTimer.getTimeStamp(cmd, "SkyComposition");
		}

	}
}