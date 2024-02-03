#include "../deferred_renderer.h"
#include "../render_scene.h"
#include "../renderer.h"
#include "../scene_textures.h"

namespace engine
{
	struct VLPush
	{
		uint sdsmShadowDepthIndices[kMaxCascadeNum];
		uint cascadeCount;
	};

    class VolumetricLightPass : public PassInterface
    {
    public:
        std::unique_ptr<ComputePipeResources> injectPipe;
        std::unique_ptr<ComputePipeResources> accumulatePipe;
        std::unique_ptr<ComputePipeResources> compositePipe;

		virtual void release() override
		{
			injectPipe.reset();
			accumulatePipe.reset();
			compositePipe.reset();
		}

        virtual void onInit() override
        {
			VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
			getContext()->descriptorFactoryBegin()
				.bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0) // inFrameData
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1) // imageTransmittanceLut
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 2) // inTransmittanceLut
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3) // imageSkyViewLut
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 4) // inSkyViewLut
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5) // imageMultiScatterLut
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 6) // inMultiScatterLut
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 7) // imageTransmittanceLut
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 8) // inTransmittanceLut
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 9) // inTransmittanceLut
				.buildNoInfoPush(setLayout);

			std::vector<VkDescriptorSetLayout> setLayouts = {
				setLayout,
				m_context->getSamplerCache().getCommonDescriptorSetLayout(),
				getRenderer()->getBlueNoise().spp_1_buffer.setLayouts,
				m_context->getBindlessTextureSetLayout()
			};

			ShaderVariant shaderVariant("shader/volumetric_light.glsl");
			shaderVariant.setStage(EShaderStage::eComputeShader);

			{
				auto copyVariant = shaderVariant;
				copyVariant.setMacro(L"INJECT_LIGHTING_PASS");
				injectPipe = std::make_unique<ComputePipeResources>(copyVariant, sizeof(VLPush), setLayouts);
			}

			{
				auto copyVariant = shaderVariant;
				copyVariant.setMacro(L"ACCUMUALTE_PASS");
				accumulatePipe = std::make_unique<ComputePipeResources>(copyVariant, sizeof(VLPush), setLayouts);
			}

			{
				auto copyVariant = shaderVariant;
				copyVariant.setMacro(L"COMPOSITE_PASS");
				compositePipe = std::make_unique<ComputePipeResources>(copyVariant, sizeof(VLPush), setLayouts);
			}

        }
    };

	void DeferredRenderer::renderVolumetricFog(
		VkCommandBuffer cmd,
		GBufferTextures* inGBuffers,
		RenderScene* scene,
		BufferParameterHandle perFrameGPU,
		AtmosphereTextures& inAtmosphere,
		const PerFrameData& perframe,
		const SkyLightRenderContext& skyContext,
		const SDSMInfos& sunSDSMInfos)
	{
		if (scene->getSkyComponent() == nullptr)
		{
			return;
		}

		auto& sceneDepthZ = inGBuffers->depthTexture->getImage();
		auto& sceneColorHdr = inGBuffers->hdrSceneColor->getImage();

		auto* pass = getContext()->getPasses().get<VolumetricLightPass>();
		auto* rtPool = &getContext()->getRenderTargetPools();

		constexpr int kWidth  = 160;
		constexpr int kHeight = 88;
		constexpr int kDepth  = 64;

		auto froxelScatter = getContext()->getRenderTargetPools().createPoolImage(
			"FogFroxelScatter",
			kWidth,  // Must can divide by 8.
			kHeight,  // Must can divide by 8.
			VK_FORMAT_R16G16B16A16_SFLOAT,
			VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
			1, // Mipmap count.
			kDepth // Depth
		);

		auto froxelAccumulate = getContext()->getRenderTargetPools().createPoolImage(
			"FogFroxelAccumulate",
			kWidth,  // Must can divide by 8.
			kHeight,  // Must can divide by 8.
			VK_FORMAT_R16G16B16A16_SFLOAT,
			VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
			1, // Mipmap count.
			kDepth // Depth
		);

		PushSetBuilder setBuilder(cmd);
		setBuilder
			.addBuffer(perFrameGPU)
			.addUAV(froxelScatter, buildBasicImageSubresource(), VK_IMAGE_VIEW_TYPE_3D)
			.addSRV(froxelScatter, buildBasicImageSubresource(), VK_IMAGE_VIEW_TYPE_3D)
			.addUAV(sceneColorHdr)
			.addSRV(sceneDepthZ, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT))
			.addBuffer(sunSDSMInfos.cascadeInfoBuffer)
			.addSRV(m_history.cloudShadowDepthHistory != nullptr ? m_history.cloudShadowDepthHistory->getImage() : 
				getContext()->getBuiltinTextureTranslucent()->getSelfImage())
			.addUAV(froxelAccumulate, buildBasicImageSubresource(), VK_IMAGE_VIEW_TYPE_3D)
			.addSRV(froxelAccumulate, buildBasicImageSubresource(), VK_IMAGE_VIEW_TYPE_3D)
			.addSRV(m_history.volumetricFogScatterIntensity != nullptr ? m_history.volumetricFogScatterIntensity : inAtmosphere.froxelScatter, buildBasicImageSubresource(), VK_IMAGE_VIEW_TYPE_3D)
			.push(pass->injectPipe.get());

		std::vector<VkDescriptorSet> additionalSets =
		{
			getContext()->getSamplerCache().getCommonDescriptorSet(),
			getRenderer()->getBlueNoise().spp_1_buffer.set,
			getContext()->getBindlessTexture().getSet()
		};
		pass->injectPipe->bindSet(cmd, additionalSets, 1);

		VLPush push{};
		push.cascadeCount = perframe.sunLightInfo.cascadeConfig.cascadeCount;

		for (int i = sunSDSMInfos.shadowDepths.size() - 1; i >= 0; i--)
		{
			push.sdsmShadowDepthIndices[i] =
				sunSDSMInfos.shadowDepths[i]->getImage().getOrCreateView(
					RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT)).srvBindless;
		}

		{
			froxelScatter->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

			pass->injectPipe->bindAndPushConst(cmd, &push);


			vkCmdDispatch(cmd, getGroupCount(kWidth, 8), getGroupCount(kHeight, 8), kDepth);

			froxelScatter->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
		}

		{
			froxelAccumulate->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

			pass->accumulatePipe->bindAndPushConst(cmd, &push);


			vkCmdDispatch(cmd, getGroupCount(kWidth, 8), getGroupCount(kHeight, 8), kDepth);

			froxelAccumulate->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
		}

		{
			sceneColorHdr.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

			pass->compositePipe->bindAndPushConst(cmd, &push);


			vkCmdDispatch(cmd,
				getGroupCount(sceneColorHdr.getExtent().width, 8), 
				getGroupCount(sceneColorHdr.getExtent().height, 8), 1);

			sceneColorHdr.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
		}

		m_history.volumetricFogScatterIntensity = froxelScatter;

		m_gpuTimer.getTimeStamp(cmd, "Volumetric Voxel fog");
	}



}