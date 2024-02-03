#include "../deferred_renderer.h"
#include "../render_scene.h"
#include "../renderer.h"
#include "../scene_textures.h"
#include "../../scene/component/reflection_probe_component.h"

namespace engine
{
	class AtmospherePass : public PassInterface
	{
	public:
		std::unique_ptr<ComputePipeResources> transmittanceLutPipe;
		std::unique_ptr<ComputePipeResources> multiScatterLutPipe;
		std::unique_ptr<ComputePipeResources> skyViewLutPipe;
		std::unique_ptr<ComputePipeResources> froxelLutPipe;
		std::unique_ptr<ComputePipeResources> compositionPipe;
		std::unique_ptr<ComputePipeResources> capturePipe;
		std::unique_ptr<ComputePipeResources> distantPipe;
		std::unique_ptr<ComputePipeResources> distantGridPipe;
	protected:
		virtual void onInit() override
		{
			VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
			getContext()->descriptorFactoryBegin()
				.bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0) // inFrameData
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1) // imageTransmittanceLut
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 2) // inTransmittanceLut
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3) // imageSkyViewLut
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 4) // inSkyViewLut
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 5) // imageMultiScatterLut
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 6) // inMultiScatterLut
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 7) // inDepth
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 8) // imageFroxelScatterLut
				.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 9) // inFroxelScatterLut
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 10) // imageHdrSceneColor
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 11) // sceneCapture
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 12) // sceneCapture
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 13) // sceneCapture
				.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 14) // sceneCapture
				.buildNoInfoPush(setLayout);

			std::vector<VkDescriptorSetLayout> setLayouts = {
				setLayout,
				m_context->getSamplerCache().getCommonDescriptorSetLayout()
			};

			ShaderVariant shaderVariant("shader/sky_render.glsl");
			shaderVariant.setStage(EShaderStage::eComputeShader);

			{
				auto copyVariant = shaderVariant;
				copyVariant.setMacro(L"TRANSMITTANCE_LUT_PASS");
				copyVariant.setMacro(L"NO_MULTISCATAPPROX_ENABLED");
				transmittanceLutPipe = std::make_unique<ComputePipeResources>(copyVariant, 0, setLayouts);
			}

			{
				auto copyVariant = shaderVariant;
				copyVariant.setMacro(L"MULTI_SCATTER_PASS");
				copyVariant.setMacro(L"NO_MULTISCATAPPROX_ENABLED");
				multiScatterLutPipe = std::make_unique<ComputePipeResources>(copyVariant, 0, setLayouts);
			}

			{
				auto copyVariant = shaderVariant;
				copyVariant.setMacro(L"SKY_LUT_PASS");
				skyViewLutPipe = std::make_unique<ComputePipeResources>(copyVariant, 0, setLayouts);
			}

			{
				auto copyVariant = shaderVariant;
				copyVariant.setMacro(L"AIR_PERSPECTIVE_PASS");
				froxelLutPipe = std::make_unique<ComputePipeResources>(copyVariant, 0, setLayouts);
			}

			{
				auto copyVariant = shaderVariant;
				copyVariant.setMacro(L"COMPOSITE_SKY_PASS");
				compositionPipe = std::make_unique<ComputePipeResources>(copyVariant, 0, setLayouts);
			}

			{
				auto copyVariant = shaderVariant;
				copyVariant.setMacro(L"SKY_CAPTURE_PASS");
				capturePipe = std::make_unique<ComputePipeResources>(copyVariant, 0, setLayouts);
			}
			{
				auto copyVariant = shaderVariant;
				copyVariant.setMacro(L"SKY_DISTANCE_LIT_CLOUD_PASS");
				distantPipe = std::make_unique<ComputePipeResources>(copyVariant, 0, setLayouts);
			}

			{
				auto copyVariant = shaderVariant;
				copyVariant.setMacro(L"SKY_DISTANCE_GRID_LIT_PASS");
				distantGridPipe = std::make_unique<ComputePipeResources>(copyVariant, 0, setLayouts);
			}
		}

		virtual void release() override
		{
			transmittanceLutPipe.reset();
			multiScatterLutPipe.reset();
			froxelLutPipe.reset();
			skyViewLutPipe.reset();
			compositionPipe.reset();
			capturePipe.reset();
			distantPipe.reset();
			distantGridPipe.reset();
		}
	};

	void engine::renderAtmosphere(
		VkCommandBuffer cmd,
		GBufferTextures* inGBuffers,
		RenderScene* scene,
		BufferParameterHandle perFrameGPU,
		const PerFrameData& perframe,
		AtmosphereTextures& inout,
		bool bComposite,
		GPUTimestamps* timer)
	{
		if (!scene->getSkyComponent())
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

			// Sky capture which is a low frequency texture use 64x64 is enough.
			inout.envCapture = getContext()->getRenderTargetPools().createPoolCubeImage(
				"AtmosphereEnvCapture",
				64,  // Must can divide by 8.
				64,  // Must can divide by 8.
				VK_FORMAT_R16G16B16A16_SFLOAT,
				VK_IMAGE_USAGE_STORAGE_BIT |
				VK_IMAGE_USAGE_SAMPLED_BIT |
				VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
				VK_IMAGE_USAGE_TRANSFER_DST_BIT
			);

			inout.distant = getContext()->getRenderTargetPools().createPoolImage(
				"AtmosphereDistantSH",
				64, 
				1,
				VK_FORMAT_R16G16B16A16_SFLOAT,
				VK_IMAGE_USAGE_STORAGE_BIT |
				VK_IMAGE_USAGE_SAMPLED_BIT |
				VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
				VK_IMAGE_USAGE_TRANSFER_DST_BIT
			);

			// Froxy distant scatter with cloud shadow.
			inout.distantGrid = getContext()->getRenderTargetPools().createPoolImage(
				"AtmosphereDistantSH-Grid",
				32,  // Must can divide by 8.
				32,  // Must can divide by 8.
				VK_FORMAT_R16G16B16A16_SFLOAT,
				VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
				1, // Mipmap count.
				32 // Depth
			);
		}

		auto& tansmittanceLut  = inout.transmittance->getImage();
		auto& skyViewLut       = inout.skyView->getImage();
		auto& multiScatterLut  = inout.multiScatter->getImage();
		auto& froxelScatterLut = inout.froxelScatter->getImage();

		auto& sceneDepthZ   = inGBuffers->depthTexture->getImage();
		auto& sceneColorHdr = inGBuffers->hdrSceneColor->getImage();
		auto& idTexture = inGBuffers->gbufferId->getImage();
		auto& envCapture    = inout.envCapture->getImage();
		auto& distantImage = inout.distant->getImage();
		auto& distantGridImage = inout.distantGrid->getImage();

		auto captureViewRange = VkImageSubresourceRange
		{ 
			.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT, 
			.baseMipLevel   = 0, 
			.levelCount     = 1, 
			.baseArrayLayer = 0, 
			.layerCount     = 6,
		};

		auto* pass = getContext()->getPasses().get<AtmospherePass>();
		PushSetBuilder setBuilder(cmd);
		setBuilder
			.addBuffer(perFrameGPU)
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
			.addUAV(envCapture, captureViewRange, VK_IMAGE_VIEW_TYPE_CUBE)
			.addUAV(idTexture)
			.addUAV(distantImage)
			.addUAV(distantGridImage, buildBasicImageSubresource(), VK_IMAGE_VIEW_TYPE_3D);

		std::vector<VkDescriptorSet> additionalSets =
		{
			getContext()->getSamplerCache().getCommonDescriptorSet(),
		};

		if (bComposite)
		{
			ScopePerframeMarker marker(cmd, "Atmosphere Composition", { 1.0f, 1.0f, 0.0f, 1.0f }, timer);

			// Pass #5. composite.
			{
				idTexture.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
				sceneColorHdr.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
				sceneDepthZ.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));

				pass->compositionPipe->bind(cmd);
				setBuilder.push(pass->compositionPipe.get());

				pass->compositionPipe->bindSet(cmd, additionalSets, 1);

				vkCmdDispatch(cmd, getGroupCount(sceneColorHdr.getExtent().width, 8), getGroupCount(sceneColorHdr.getExtent().height, 8), 1);
				sceneColorHdr.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
				idTexture.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
			}

		}
		else
		{
			ScopePerframeMarker marker(cmd, "Atmosphere Luts", { 1.0f, 1.0f, 0.0f, 1.0f }, timer);


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

				vkCmdDispatch(cmd, 
					getGroupCount(froxelScatterLut.getExtent().width,  8), 
					getGroupCount(froxelScatterLut.getExtent().height, 8), 
					froxelScatterLut.getExtent().depth);
				froxelScatterLut.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
			}

			// Pass #4 Capture pass.
			{
				envCapture.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, captureViewRange);

				pass->capturePipe->bind(cmd);
				setBuilder.push(pass->capturePipe.get());
				pass->capturePipe->bindSet(cmd, additionalSets, 1);
				vkCmdDispatch(cmd,
					getGroupCount(envCapture.getExtent().width, 8),
					getGroupCount(envCapture.getExtent().height, 8), 6);

			}

			{
				distantImage.transitionGeneral(cmd);
				pass->distantPipe->bind(cmd);
				setBuilder.push(pass->distantPipe.get());
				pass->distantPipe->bindSet(cmd, additionalSets, 1);
				vkCmdDispatch(cmd, 1, 1, inout.distant->getImage().getExtent().width);
			}

			{
				distantGridImage.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

				pass->distantGridPipe->bind(cmd);
				setBuilder.push(pass->distantGridPipe.get());
				pass->distantGridPipe->bindSet(cmd, additionalSets, 1);

				vkCmdDispatch(cmd,
					distantGridImage.getExtent().width,
					distantGridImage.getExtent().height,
					distantGridImage.getExtent().depth);

				distantGridImage.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
			}


			envCapture.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, captureViewRange);
		}
	}
}