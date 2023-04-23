#include "../renderer_interface.h"
#include "../render_scene.h"
#include "../renderer.h"
#include "../scene_textures.h"

namespace engine
{
	class AtmospherePass : public PassInterface
	{
	public:
		VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;

		std::unique_ptr<ComputePipeResources> transmittanceLutPipe;
		std::unique_ptr<ComputePipeResources> multiScatterLutPipe;
		std::unique_ptr<ComputePipeResources> skyViewLutPipe;
		std::unique_ptr<ComputePipeResources> froxelLutPipe;
		std::unique_ptr<ComputePipeResources> compositionPipe;
		std::unique_ptr<ComputePipeResources> capturePipe;
	protected:
		virtual void onInit() override
		{
			getContext()->descriptorFactoryBegin()
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCommonShaderStage, 0) // imageTransmittanceLut
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCommonShaderStage, 1) // inTransmittanceLut
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCommonShaderStage, 2) // imageSkyViewLut
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCommonShaderStage, 3) // inSkyViewLut
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCommonShaderStage, 4) // imageMultiScatterLut
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCommonShaderStage, 5) // inMultiScatterLut
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCommonShaderStage, 6) // inDepth
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCommonShaderStage, 7) // imageFroxelScatterLut
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCommonShaderStage, 8) // inFroxelScatterLut
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCommonShaderStage, 9) // imageHdrSceneColor
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCommonShaderStage, 10) // inGBufferA
				.bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kCommonShaderStage, 11) // inFrameData
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCommonShaderStage, 12) // imageCaptureEnv
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCommonShaderStage, 13) // imageCaptureEnv
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCommonShaderStage, 14) // imageCaptureEnv
				.buildNoInfoPush(setLayout);

			std::vector<VkDescriptorSetLayout> setLayouts = { 
				setLayout, 
				m_context->getSamplerCache().getCommonDescriptorSetLayout(),
				getRenderer()->getBlueNoise().spp_1_buffer.setLayouts
			};
			
			transmittanceLutPipe = std::make_unique<ComputePipeResources>("shader/transmittance_lut.comp.spv", 0, setLayouts);
			multiScatterLutPipe = std::make_unique<ComputePipeResources>("shader/multi_scatter_lut.comp.spv", 0, setLayouts);
			skyViewLutPipe = std::make_unique<ComputePipeResources>("shader/skyview_lut.comp.spv", 0, setLayouts);
			froxelLutPipe = std::make_unique<ComputePipeResources>("shader/sky_air_perspective.comp.spv", 0, setLayouts);
			compositionPipe = std::make_unique<ComputePipeResources>("shader/sky_composition.comp.spv", 0, setLayouts);
			capturePipe = std::make_unique<ComputePipeResources>("shader/bake_capture.comp.spv", 0, setLayouts);
		}

		virtual void release() override
		{
			transmittanceLutPipe.reset();
			multiScatterLutPipe.reset();
			froxelLutPipe.reset();
			skyViewLutPipe.reset();
			compositionPipe.reset();
			capturePipe.reset();
		}
	};

	FSR2Context* RendererInterface::getFSR2()
	{
		if (m_fsr2 == nullptr)
		{
			m_fsr2 = std::make_unique<FSR2Context>();
		}

		return m_fsr2.get();
	}

	void RendererInterface::renderAtmosphere(
		VkCommandBuffer cmd,
		class GBufferTextures* inGBuffers,
		class RenderScene* scene,
		BufferParameterHandle perFrameGPU,
		AtmosphereTextures& inout,
		const SDSMInfos* sdsmInfos,
		bool bComposite)
	{
		if (!scene->isSkyExist())
		{
			return;
		}

		if (!inout.isValid())
		{
			inout.transmittance = getContext()->getRenderTargetPools().createPoolImage(
				"AtmosphereTransmittance",
				256, // Must can divide by 8.
				64,  // Must can divide by 8.
				VK_FORMAT_R16G16B16A16_SFLOAT,
				VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
			);

			inout.skyView = getContext()->getRenderTargetPools().createPoolImage(
				"AtmosphereSkyView",
				256,
				256,
				VK_FORMAT_R16G16B16A16_SFLOAT,
				VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
			);

			inout.multiScatter = getContext()->getRenderTargetPools().createPoolImage(
				"AtmosphereMultiScatter",
				32,  // Must can divide by 8.
				32,  // Must can divide by 8.
				VK_FORMAT_R16G16B16A16_SFLOAT,
				VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
			);

			inout.froxelScatter = getContext()->getRenderTargetPools().createPoolImage(
				"AtmosphereFroxelScatter",
				32,  // Must can divide by 8.
				32,  // Must can divide by 8.
				VK_FORMAT_R16G16B16A16_SFLOAT,
				VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
				1, // Mipmap count.
				32 // Depth
			);

			inout.envCapture = getContext()->getRenderTargetPools().createPoolCubeImage(
				"AtmosphereEnvCapture",
				128,  // Must can divide by 8.
				128,  // Must can divide by 8.
				VK_FORMAT_R16G16B16A16_SFLOAT,
				VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
				-1 // Need mipmaps.
			);
		}

		auto& tansmittanceLut = inout.transmittance->getImage();
		auto& skyViewLut = inout.skyView->getImage();
		auto& multiScatterLut = inout.multiScatter->getImage();
		auto& froxelScatterLut = inout.froxelScatter->getImage();
		auto& sceneDepthZ = inGBuffers->depthTexture->getImage();
		auto& sceneColorHdr = inGBuffers->hdrSceneColor->getImage();
		auto& gbufferA = inGBuffers->gbufferA->getImage();
		auto& envCapture = inout.envCapture->getImage();
		auto captureViewRange = VkImageSubresourceRange{ .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 6 };

		auto* pass = getContext()->getPasses().get<AtmospherePass>();
		PushSetBuilder setBuilder(cmd);
		setBuilder
			.addUAV(tansmittanceLut)
			.addSRV(tansmittanceLut)
			.addUAV(skyViewLut)
			.addSRV(skyViewLut)
			.addUAV(multiScatterLut)
			.addSRV(multiScatterLut)
			.addSRV(sceneDepthZ, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT))
			.addUAV(froxelScatterLut, buildBasicImageSubresource(), VK_IMAGE_VIEW_TYPE_3D)
			.addSRV(froxelScatterLut, buildBasicImageSubresource(), VK_IMAGE_VIEW_TYPE_3D)
			.addUAV(sceneColorHdr)
			.addSRV(gbufferA)
			.addBuffer(perFrameGPU)
			.addUAV(envCapture, captureViewRange, VK_IMAGE_VIEW_TYPE_CUBE)
			.addSRV(sdsmInfos->shadowDepths, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT))
			.addBuffer(sdsmInfos->cascadeInfoBuffer);

		sdsmInfos->shadowDepths->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));

		std::vector<VkDescriptorSet> additionalSets = 
		{
			m_context->getSamplerCache().getCommonDescriptorSet(),
			getRenderer()->getBlueNoise().spp_1_buffer.set
		};

		if (bComposite)
		{
			ScopePerframeMarker marker(cmd, "Atmosphere Composition", { 1.0f, 1.0f, 0.0f, 1.0f });

			// Pass #4. composite.
			{
				sceneColorHdr.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
				gbufferA.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

				pass->compositionPipe->bind(cmd);
				setBuilder.push(pass->compositionPipe.get());

				pass->compositionPipe->bindSet(cmd, additionalSets, 1);

				vkCmdDispatch(cmd, getGroupCount(sceneColorHdr.getExtent().width, 8), getGroupCount(sceneColorHdr.getExtent().height, 8), 1);
				sceneColorHdr.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
			}

			m_gpuTimer.getTimeStamp(cmd, "SkyComposition");

		}
		else
		{
			ScopePerframeMarker marker(cmd, "Atmosphere Luts", { 1.0f, 1.0f, 0.0f, 1.0f });


			// Pass #0. tansmittance lut.
			{
				tansmittanceLut.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
				sceneDepthZ.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));

				pass->transmittanceLutPipe->bind(cmd);
				setBuilder.push(pass->transmittanceLutPipe.get());
				pass->transmittanceLutPipe->bindSet(cmd, additionalSets, 1);
				vkCmdDispatch(cmd, getGroupCount(tansmittanceLut.getExtent().width, 8), getGroupCount(tansmittanceLut.getExtent().height, 8), 1);
				tansmittanceLut.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
			}

			// Pass #1. multi scatter lut.
			{
				multiScatterLut.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

				pass->multiScatterLutPipe->bind(cmd);
				setBuilder.push(pass->multiScatterLutPipe.get());
				pass->multiScatterLutPipe->bindSet(cmd, additionalSets, 1);
				vkCmdDispatch(cmd, multiScatterLut.getExtent().width, multiScatterLut.getExtent().height, 1);
				multiScatterLut.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
			}

			// Pass #2. sky view lut.
			{
				skyViewLut.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

				pass->skyViewLutPipe->bind(cmd);
				setBuilder.push(pass->skyViewLutPipe.get());
				pass->skyViewLutPipe->bindSet(cmd, additionalSets, 1);
				vkCmdDispatch(cmd, getGroupCount(skyViewLut.getExtent().width, 8), getGroupCount(skyViewLut.getExtent().height, 8), 1);
				skyViewLut.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
			}

			// Pass #3. froxel lut.
			{
				froxelScatterLut.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

				pass->froxelLutPipe->bind(cmd);
				setBuilder.push(pass->froxelLutPipe.get());
				pass->froxelLutPipe->bindSet(cmd, additionalSets, 1);

				vkCmdDispatch(cmd, getGroupCount(froxelScatterLut.getExtent().width, 8), getGroupCount(froxelScatterLut.getExtent().height, 8), froxelScatterLut.getExtent().depth);
				froxelScatterLut.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
			}

			// Capture pass.
			{

				envCapture.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, captureViewRange);

				pass->capturePipe->bind(cmd);
				setBuilder.push(pass->capturePipe.get());
				pass->capturePipe->bindSet(cmd, additionalSets, 1);

				vkCmdDispatch(cmd,
					getGroupCount(envCapture.getExtent().width, 8),
					getGroupCount(envCapture.getExtent().height, 8), 6);
				envCapture.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, captureViewRange);
			}

			m_gpuTimer.getTimeStamp(cmd, "SkyPrepare");
		}
	}

	struct GPUSkylightPush
	{
		uint32_t convolutionSampleCount; // 4096
		int  updateFaceIndex;
	};

	struct GPUSkyReflectionPush
	{
		float alphaRoughness;
		int samplesCount;         // 1024
		float maxBrightValue;     // 1000.0f
		float filterRoughnessMin; // 0.05f
		int updateFaceIndex;
	};

	class SkylightPass : public PassInterface
	{
	public:
		VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;

		std::unique_ptr<ComputePipeResources> radiancePipe;
		std::unique_ptr<ComputePipeResources> reflectionPipe;

	protected:
		virtual void onInit() override
		{
			getContext()->descriptorFactoryBegin()
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0) // out irradiance
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1) // sample cube
				.buildNoInfoPush(setLayout);

			std::vector<VkDescriptorSetLayout> setLayouts = { setLayout, m_context->getSamplerCache().getCommonDescriptorSetLayout() };

			radiancePipe = std::make_unique<ComputePipeResources>("shader/skylight.comp.spv", sizeof(GPUSkylightPush), setLayouts);
			reflectionPipe = std::make_unique<ComputePipeResources>("shader/skyreflection.comp.spv", sizeof(GPUSkyReflectionPush), setLayouts);
		}

		virtual void release() override
		{
			radiancePipe.reset();
			reflectionPipe.reset();
		}
	};

	void RendererInterface::renderSkylight(VkCommandBuffer cmd, const AtmosphereTextures& inAtmosphere)
	{
		// Update index of face id.
		m_skylightUpdateFaceIndex ++;
		m_skylightUpdateFaceIndex = m_skylightUpdateFaceIndex % 6;

		auto* pass = getContext()->getPasses().get<SkylightPass>();
		const auto& skyEnvCube = inAtmosphere.envCapture;

		const auto inCubeViewRangeAll = VkImageSubresourceRange
		{
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel = 0,
			.levelCount = skyEnvCube->getImage().getInfo().mipLevels,
			.baseArrayLayer = 0,
			.layerCount = 6
		};

		skyEnvCube->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, inCubeViewRangeAll);

		// Generate cubemap mips for filter.
		{
			auto cubemapViewRange = VkImageSubresourceRange
			{
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = m_skylightUpdateFaceIndex,
				.layerCount = 1
			};

			// Mip 0 as src input.
			skyEnvCube->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, cubemapViewRange);

			VkImageMemoryBarrier barrier{};
			barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			barrier.image = skyEnvCube->getImage().getImage();
			barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			barrier.subresourceRange.baseArrayLayer = m_skylightUpdateFaceIndex;
			barrier.subresourceRange.layerCount = 1;
			barrier.subresourceRange.levelCount = 1;

			// Generate cubemap mips.
			int32_t mipWidth = skyEnvCube->getImage().getExtent().width;
			int32_t mipHeight = skyEnvCube->getImage().getExtent().height;
			for (uint32_t i = 1; i < skyEnvCube->getImage().getInfo().mipLevels; i++)
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

				blit.srcSubresource.baseArrayLayer = m_skylightUpdateFaceIndex;
				blit.dstSubresource.baseArrayLayer = m_skylightUpdateFaceIndex;

				blit.srcSubresource.layerCount = 1; // Cube map, only update one face
				blit.dstSubresource.layerCount = 1;

				vkCmdBlitImage(cmd,
					skyEnvCube->getImage().getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
					skyEnvCube->getImage().getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
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

				if (mipWidth > 1) mipWidth /= 2;
				if (mipHeight > 1) mipHeight /= 2;
			}

			cubemapViewRange.levelCount = skyEnvCube->getImage().getInfo().mipLevels;
			skyEnvCube->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, cubemapViewRange);
		}

		skyEnvCube->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, inCubeViewRangeAll);

		// Irradiance Compute.
		{
			auto irradianceViewRange = VkImageSubresourceRange
			{
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 6
			};

			m_skylightRadiance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, irradianceViewRange);
			{
				GPUSkylightPush push
				{
					.convolutionSampleCount = 1024,
					.updateFaceIndex = int(m_skylightUpdateFaceIndex),
				};

				pass->radiancePipe->bindAndPushConst(cmd, &push);
				PushSetBuilder(cmd)
					.addUAV(m_skylightRadiance, irradianceViewRange, VK_IMAGE_VIEW_TYPE_CUBE)
					.addSRV(skyEnvCube, inCubeViewRangeAll, VK_IMAGE_VIEW_TYPE_CUBE)
					.push(pass->radiancePipe.get());

				pass->radiancePipe->bindSet(cmd, std::vector<VkDescriptorSet>{
					m_context->getSamplerCache().getCommonDescriptorSet()
				}, 1);

				vkCmdDispatch(cmd,
					m_skylightRadiance->getImage().getExtent().width,
					m_skylightRadiance->getImage().getExtent().height,
					1u
				);
			}
			m_skylightRadiance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, irradianceViewRange);
		}

		{
			pass->reflectionPipe->bind(cmd);
			pass->reflectionPipe->bindSet(cmd, std::vector<VkDescriptorSet>{
				m_context->getSamplerCache().getCommonDescriptorSet()
			}, 1);

			const float deltaRoughness = 1.0f / std::max(float(m_skylightReflection->getImage().getInfo().mipLevels), 1.0f);

			for (uint32_t i = 0; i < m_skylightReflection->getImage().getInfo().mipLevels; i++)
			{
				auto viewRange = VkImageSubresourceRange
				{
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.baseMipLevel = i,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 6
				};

				m_skylightReflection->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, viewRange);
				{
					PushSetBuilder(cmd)
						.addUAV(m_skylightReflection, viewRange, VK_IMAGE_VIEW_TYPE_CUBE)
						.addSRV(skyEnvCube, inCubeViewRangeAll, VK_IMAGE_VIEW_TYPE_CUBE)
						.push(pass->reflectionPipe.get());

					GPUSkyReflectionPush push
					{
						.alphaRoughness = float(i) * deltaRoughness,
						.samplesCount = 512,
						.maxBrightValue = 10000.0f,
						.filterRoughnessMin = 0.05f,  // Realtime filter all roughness.
						.updateFaceIndex = int(m_skylightUpdateFaceIndex),
					};

					pass->reflectionPipe->pushConst(cmd, &push);
					vkCmdDispatch(cmd,
						glm::max(1u, m_skylightReflection->getImage().getExtent().width >> i),
						glm::max(1u, m_skylightReflection->getImage().getExtent().height >> i),
						1u);
				}
				m_skylightReflection->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, viewRange);
			}
		}

		m_skylightRadiance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, inCubeViewRangeAll);
		m_skylightReflection->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, inCubeViewRangeAll);
	}
}