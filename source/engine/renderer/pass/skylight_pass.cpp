#include "../deferred_renderer.h"
#include "../render_scene.h"
#include "../renderer.h"
#include "../scene_textures.h"

namespace engine
{
	struct GPUSkyReflectionPush
	{
		float alphaRoughness;
		int   samplesCount;          // 1024
		float maxBrightValue;        // 1000.0f
		float filterRoughnessMin;    // 0.05f

		uint convolutionSampleCount; // 4096
	};

	class SkylightPass : public PassInterface
	{
	public:
		std::unique_ptr<ComputePipeResources> radiancePipe;
		std::unique_ptr<ComputePipeResources> reflectionPipe;

	protected:
		virtual void onInit() override
		{
			VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;

			getContext()->descriptorFactoryBegin()
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 0) 
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1) 
				.buildNoInfoPush(setLayout);

			std::vector<VkDescriptorSetLayout> setLayouts = 
			{ 
				setLayout, 
				m_context->getSamplerCache().getCommonDescriptorSetLayout() 
			};

			ShaderVariant shaderVariant("shader/skylight.glsl");
			shaderVariant.setStage(EShaderStage::eComputeShader);

			{
				auto copyVariant = shaderVariant;
				copyVariant.setMacro(L"SKYLIGHT_IRRADIANCE_PASS");
				radiancePipe = std::make_unique<ComputePipeResources>(copyVariant, sizeof(GPUSkyReflectionPush), setLayouts);
			}
			{
				auto copyVariant = shaderVariant;
				copyVariant.setMacro(L"SKYLIGHT_REFLECTION_PASS");
				reflectionPipe = std::make_unique<ComputePipeResources>(copyVariant, sizeof(GPUSkyReflectionPush), setLayouts);
			}
		}

		virtual void release() override
		{
			radiancePipe.reset();
			reflectionPipe.reset();
		}
	};

	void engine::buildCubemapReflection(
		VkCommandBuffer cmd,
		PoolImageSharedRef cube,
		PoolImageSharedRef& resultImage,
		uint dimension)
	{
		const auto inCubeViewRangeAll = VkImageSubresourceRange
		{
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel = 0,
			.levelCount = cube->getImage().getInfo().mipLevels,
			.baseArrayLayer = 0,
			.layerCount = 6
		};

		int32_t mipWidth = cube->getImage().getExtent().width;
		int32_t mipHeight = cube->getImage().getExtent().height;

		// Blit inner to generate full mipmaps.
		for (uint32_t i = 1; i < cube->getImage().getInfo().mipLevels; i++)
		{
			const auto& skyImage = cube->getImage().getImage();

			// Dest mip.
			VkImageMemoryBarrier barrier{};
			barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			barrier.image = skyImage;
			barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			barrier.subresourceRange.baseArrayLayer = 0;
			barrier.subresourceRange.layerCount = 6;
			barrier.subresourceRange.baseMipLevel = i;
			barrier.subresourceRange.levelCount = 1;
			barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			barrier.srcAccessMask = VK_ACCESS_NONE;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

			// Blit struct.
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
			blit.srcSubresource.layerCount = 6;
			blit.dstSubresource.layerCount = 6;

			vkCmdBlitImage(cmd, skyImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, skyImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

			// Layout for read sync.
			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
			vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

			if (mipWidth > 1) mipWidth /= 2;
			if (mipHeight > 1) mipHeight /= 2;
		}
		cube->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, inCubeViewRangeAll);

		const GPUSkyReflectionPush pushTemplate
		{
			.samplesCount = 128,
			.maxBrightValue = 10000.0f,
			.filterRoughnessMin = 0.05f,  // Realtime filter all roughness.
			.convolutionSampleCount = 256,
		};

		resultImage = getContext()->getRenderTargetPools().createPoolCubeImage(
			"sceneEnvCapture_convole",
			dimension,  // Must can divide by 8.
			dimension,  // Must can divide by 8.
			VK_FORMAT_R16G16B16A16_SFLOAT,
			VK_IMAGE_USAGE_STORAGE_BIT |
			VK_IMAGE_USAGE_SAMPLED_BIT |
			VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
			VK_IMAGE_USAGE_TRANSFER_DST_BIT,
			-1
		);

		{
			auto* pass = getContext()->getPasses().get<SkylightPass>();
			auto push = pushTemplate;

			pass->reflectionPipe->bind(cmd);
			pass->reflectionPipe->bindSet(cmd, std::vector<VkDescriptorSet>{
				getContext()->getSamplerCache().getCommonDescriptorSet()
			}, 1);

			const float deltaRoughness = 1.0f / std::max(float(resultImage->getImage().getInfo().mipLevels), 1.0f);

			for (uint32_t i = 0; i < resultImage->getImage().getInfo().mipLevels; i++)
			{
				auto viewRange = VkImageSubresourceRange
				{
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.baseMipLevel = i,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 6
				};

				resultImage->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, viewRange);
				{
					PushSetBuilder(cmd)
						.addUAV(resultImage, viewRange, VK_IMAGE_VIEW_TYPE_CUBE)
						.addSRV(cube, inCubeViewRangeAll, VK_IMAGE_VIEW_TYPE_CUBE)
						.push(pass->reflectionPipe.get());

					push.alphaRoughness = float(i) * deltaRoughness;
					pass->reflectionPipe->pushConst(cmd, &push);
					vkCmdDispatch(cmd,
						glm::max(1u, resultImage->getImage().getExtent().width >> i),
						glm::max(1u, resultImage->getImage().getExtent().height >> i),
						6U);
				}
			}
		}
		resultImage->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, inCubeViewRangeAll);
	}

	void engine::renderSkylight(
		VkCommandBuffer cmd,
		const struct AtmosphereTextures& inAtmosphere,
		const PerFrameData& perframe,
		RenderScene* scene,
		SkyLightRenderContext& context,
		ReflectionProbeContext& reflectionProbe,
		GPUTimestamps* timer)
	{
		if (!inAtmosphere.envCapture)
		{
			return;
		}

		auto inSkyEnvCube = inAtmosphere.envCapture;

		// Init resources.
		{
			CHECK (context.skylightRadiance == nullptr)
			{
				context.skylightRadiance = getContext()->getRenderTargetPools().createPoolCubeImage(
					"SkyIBLIrradiance",
					context.irradianceDim,  // Must can divide by 8.
					context.irradianceDim,  // Must can divide by 8.
					VK_FORMAT_R16G16B16A16_SFLOAT,
					VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
				);
				context.skylightRadiance->getImage().transitionLayout(
					cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresourceCube());
			}
			CHECK(context.skylightReflection == nullptr)
			{
				context.skylightReflection = getContext()->getRenderTargetPools().createPoolCubeImage(
					"SkyIBLPrefilter",
					inSkyEnvCube->getImage().getExtent().width,
					inSkyEnvCube->getImage().getExtent().height,
					VK_FORMAT_R16G16B16A16_SFLOAT,
					VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
					-1    // Need mipmaps.
				);
				context.skylightReflection->getImage().transitionLayout(
					cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresourceCube());
			}
		}

		auto* pass = getContext()->getPasses().get<SkylightPass>();

		auto skyEnvCube = getContext()->getRenderTargetPools().createPoolCubeImage(
			"SkyEnvCapture",
			inSkyEnvCube->getImage().getExtent().width,  // Must can divide by 8.
			inSkyEnvCube->getImage().getExtent().height,  // Must can divide by 8.
			VK_FORMAT_R16G16B16A16_SFLOAT,
			VK_IMAGE_USAGE_STORAGE_BIT |
			VK_IMAGE_USAGE_SAMPLED_BIT |
			VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
			VK_IMAGE_USAGE_TRANSFER_DST_BIT,
			-1
		);

		const auto inCubeViewRangeAll = VkImageSubresourceRange
		{
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel = 0,
			.levelCount = skyEnvCube->getImage().getInfo().mipLevels,
			.baseArrayLayer = 0,
			.layerCount = 6
		};

		int32_t mipWidth = skyEnvCube->getImage().getExtent().width;
		int32_t mipHeight = skyEnvCube->getImage().getExtent().height;

		// Blit mip 0 to skyEnvCube.
		{
			inSkyEnvCube->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, inCubeViewRangeAll);
			skyEnvCube->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, inCubeViewRangeAll);

			VkImageBlit blit{};
			blit.srcOffsets[0] = { 0, 0, 0 };
			blit.dstOffsets[0] = { 0, 0, 0 };
			blit.srcOffsets[1] = { mipWidth, mipHeight, 1};
			blit.dstOffsets[1] = { mipWidth, mipHeight, 1 };
			blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			blit.srcSubresource.mipLevel = 0;
			blit.dstSubresource.mipLevel = 0;
			blit.srcSubresource.baseArrayLayer = 0;
			blit.dstSubresource.baseArrayLayer = 0;
			blit.srcSubresource.layerCount = 6;
			blit.dstSubresource.layerCount = 6;

			vkCmdBlitImage(cmd, 
				inSkyEnvCube->getImage().getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				skyEnvCube->getImage().getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
		}
		skyEnvCube->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, inCubeViewRangeAll);


		// Blit inner to generate full mipmaps.
		for (uint32_t i = 1; i < skyEnvCube->getImage().getInfo().mipLevels; i++)
		{
			const auto& skyImage = skyEnvCube->getImage().getImage();

			// Dest mip.
			VkImageMemoryBarrier barrier{};
			barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			barrier.image = skyImage;
			barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			barrier.subresourceRange.baseArrayLayer = 0;
			barrier.subresourceRange.layerCount = 6;
			barrier.subresourceRange.baseMipLevel = i;
			barrier.subresourceRange.levelCount = 1;
			barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			barrier.srcAccessMask = VK_ACCESS_NONE;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

			// Blit struct.
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
			blit.srcSubresource.layerCount = 6; 
			blit.dstSubresource.layerCount = 6;

			vkCmdBlitImage(cmd, skyImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, skyImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

			// Layout for read sync.
			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
			vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

			if (mipWidth  > 1) mipWidth  /= 2;
			if (mipHeight > 1) mipHeight /= 2;
		}
		skyEnvCube->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, inCubeViewRangeAll);

		if (timer)
		{
			timer->getTimeStamp(cmd, "Skylight mipmap");
		}

		const GPUSkyReflectionPush pushTemplate
		{
			.samplesCount = (int)context.irradianceSampleCount,
			.maxBrightValue = 10000.0f,
			.filterRoughnessMin = 0.05f,  // Realtime filter all roughness.
			.convolutionSampleCount = context.reflectionSampleCount,
		};

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

			context.skylightRadiance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, irradianceViewRange);
			{
				pass->radiancePipe->bindAndPushConst(cmd, &pushTemplate);
				PushSetBuilder(cmd)
					.addUAV(context.skylightRadiance, irradianceViewRange, VK_IMAGE_VIEW_TYPE_CUBE)
					.addSRV(skyEnvCube, inCubeViewRangeAll, VK_IMAGE_VIEW_TYPE_CUBE)
					.push(pass->radiancePipe.get());

				pass->radiancePipe->bindSet(cmd, std::vector<VkDescriptorSet>{
					getContext()->getSamplerCache().getCommonDescriptorSet()
				}, 1);

				vkCmdDispatch(cmd,
					context.skylightRadiance->getImage().getExtent().width,
					context.skylightRadiance->getImage().getExtent().height,
					6U
				);
			}
		}

		if (timer)
		{
			timer->getTimeStamp(cmd, "Skylight irradiance");
		}


		{
			auto push = pushTemplate;

			pass->reflectionPipe->bind(cmd);
			pass->reflectionPipe->bindSet(cmd, std::vector<VkDescriptorSet>{
				getContext()->getSamplerCache().getCommonDescriptorSet()
			}, 1);

			const float deltaRoughness = 1.0f / std::max(float(context.skylightReflection->getImage().getInfo().mipLevels), 1.0f);

			for (uint32_t i = 0; i < context.skylightReflection->getImage().getInfo().mipLevels; i++)
			{
				auto viewRange = VkImageSubresourceRange
				{
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.baseMipLevel = i,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 6
				};

				context.skylightReflection->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, viewRange);
				{
					PushSetBuilder(cmd)
						.addUAV(context.skylightReflection, viewRange, VK_IMAGE_VIEW_TYPE_CUBE)
						.addSRV(skyEnvCube, inCubeViewRangeAll, VK_IMAGE_VIEW_TYPE_CUBE)
						.push(pass->reflectionPipe.get());

					push.alphaRoughness = float(i) * deltaRoughness;
					pass->reflectionPipe->pushConst(cmd, &push);
					vkCmdDispatch(cmd,
						glm::max(1u, context.skylightReflection->getImage().getExtent().width >> i),
						glm::max(1u, context.skylightReflection->getImage().getExtent().height >> i),
						6U);
				}
			}
		}
		context.skylightRadiance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, inCubeViewRangeAll);
		context.skylightReflection->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, inCubeViewRangeAll);


		if (timer)
		{
			timer->getTimeStamp(cmd, "Skylight conv");
		}
	}
}