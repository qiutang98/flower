#include "../deferred_renderer.h"
#include "../render_scene.h"
#include "../renderer.h"
#include "../scene_textures.h"
#include "../../scene/component/landscape_component.h"
#include <asset/asset_texture.h>
#include <asset/asset_manager.h>

namespace engine
{

	static AutoCVarFloat cVarTerrainLodContinueCoefficient(
		"r.terrain.lod.continue", "coefficient of lod continue.", "Terrain", 1.5f, CVarFlags::ReadAndWrite);

	static AutoCVarBool cVarTerrainDebugFrame(
		"r.terrain.debugFrameware", "terrain render frameware.", "Terrain", false, CVarFlags::ReadAndWrite);

	struct TerrainLODPreparePush
	{
		uint lodIndex; // from (frameData.landscape.lodCount - 1) to 1.
		float coefficientLodContinue;
		uint maxLodMipmap;
	};

	struct GPUDispatchIndirectCommand
	{
		uint32_t x;
		uint32_t y;
		uint32_t z;
		uint32_t pad;
	};
	static_assert(sizeof(GPUDispatchIndirectCommand) % (4 * sizeof(float)) == 0);

	struct HeightmapHzbPassPush
	{
		uint32_t kLoadSrcLevel = 1;
	};

	struct TerrainShadowDepthSDSMPush
	{
		uint32_t inCascadeId;
	};

	class TerrainPass : public PassInterface
	{
	public:
		std::unique_ptr<ComputePipeResources> lodPrepare;
		std::unique_ptr<ComputePipeResources> lodArgs;
		std::unique_ptr<ComputePipeResources> patchArgs;

		std::unique_ptr<ComputePipeResources> patchPrepare;
		std::unique_ptr<ComputePipeResources> gbufferDrawArgs;

		std::unique_ptr<GraphicPipeResources> gbuffer;

		std::unique_ptr<ComputePipeResources> loadMap;
		std::unique_ptr<ComputePipeResources> heightmapHzb;

		std::unique_ptr<ComputePipeResources> heightmap2NormalMap;

		std::unique_ptr<GraphicPipeResources> sdsmShadowDepth;

		virtual void release() override
		{
			gbuffer.reset();
			lodPrepare.reset();
			patchPrepare.reset();
			lodArgs.reset();
			patchArgs.reset();
			gbufferDrawArgs.reset();
			heightmapHzb.reset();
			loadMap.reset();
			heightmap2NormalMap.reset();
			sdsmShadowDepth.reset();
		}

		virtual void onInit() override
		{
			{
				VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
				getContext()->descriptorFactoryBegin()
					.bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0) // frameData
					.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1) // 
					.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  2)
					.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3) // 
					.buildNoInfoPush(setLayout);

				ShaderVariant shaderVariant("shader/terrain_shadow_depth.glsl");
				shaderVariant.setStage(EShaderStage::eVertexShader);

				ShaderVariant emptyFrag{ };

				sdsmShadowDepth = std::make_unique<GraphicPipeResources>(
					shaderVariant,
					emptyFrag,
                    std::vector<VkDescriptorSetLayout>
                    {
						setLayout,
                        m_context->getSamplerCache().getCommonDescriptorSetLayout(),
                    },
                    sizeof(TerrainShadowDepthSDSMPush),
                    std::vector<VkFormat>{ },
                    std::vector<VkPipelineColorBlendAttachmentState>{ },
                    GBufferTextures::depthTextureFormat(),
                    VK_CULL_MODE_NONE,
					VK_COMPARE_OP_GREATER, true, true,
					std::vector<VkVertexInputAttributeDescription>
					{
						{ 0, 0, VK_FORMAT_R32G32_SFLOAT, 0 },
					},
					sizeof(float) * 2);
			}

			{
				VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
				getContext()->descriptorFactoryBegin()
					.bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0) // frameData
					.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1) // 
					.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2) // 
					.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3) // 
					.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4) // 
					.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5) // 
					.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6) // 
					.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7) // 
					.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 8)
					.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 9) // 
					.buildNoInfoPush(setLayout);

				std::vector<VkDescriptorSetLayout> setLayouts = {
					setLayout,
				};


				{
					ShaderVariant shaderVariant("shader/terrain_lod.glsl");
					shaderVariant.setStage(EShaderStage::eComputeShader).setMacro(L"LOD_PREPARE_PASS");

					lodPrepare = std::make_unique<ComputePipeResources>(
						shaderVariant,
						sizeof(TerrainLODPreparePush),
						setLayouts);
				}

				{
					ShaderVariant shaderVariant("shader/terrain_lod.glsl");
					shaderVariant.setStage(EShaderStage::eComputeShader).setMacro(L"LOD_ARGS_PASS");

					lodArgs = std::make_unique<ComputePipeResources>(
						shaderVariant,
						sizeof(TerrainLODPreparePush),
						setLayouts);
				}

				{
					ShaderVariant shaderVariant("shader/terrain_lod.glsl");
					shaderVariant.setStage(EShaderStage::eComputeShader).setMacro(L"LOD_PATH_PASS");

					patchArgs = std::make_unique<ComputePipeResources>(
						shaderVariant,
						sizeof(TerrainLODPreparePush),
						setLayouts);
				}
			}

			{
				// Config code.
				VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
				getContext()->descriptorFactoryBegin()
					.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 0) // inDepth
					.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1) // hizFurthestImage
					.buildNoInfoPush(setLayout);

				heightmapHzb = std::make_unique<ComputePipeResources>("shader/terrain_heightMap_MinMaxGen.glsl", 
					(uint32_t)sizeof(HeightmapHzbPassPush), std::vector<VkDescriptorSetLayout>{ setLayout });
			}

			{
				// Config code.
				VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
				getContext()->descriptorFactoryBegin()
					.bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0) // frameData
					.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1) // 
					.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2) // hizFurthestImage
					.buildNoInfoPush(setLayout);

				loadMap = std::make_unique<ComputePipeResources>("shader/terrain_lod_patch_map.glsl", 0, std::vector<VkDescriptorSetLayout>{ setLayout });
			}

			{
				// Config code.
				VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
				getContext()->descriptorFactoryBegin()
					.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 0) // 
					.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1) // hizFurthestImage
					.buildNoInfoPush(setLayout);

				heightmap2NormalMap = std::make_unique<ComputePipeResources>("shader/heightfield2normal.glsl", 0, std::vector<VkDescriptorSetLayout>{ setLayout });
			}

			{
				VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
				getContext()->descriptorFactoryBegin()
					.bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0) // frameData
					.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1) // 
					.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2) // 
					.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3) // 
					.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4) // 
					.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5) // 
					.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  6)
					.buildNoInfoPush(setLayout);

				std::vector<VkDescriptorSetLayout> setLayouts = {
					setLayout,
				};

				{
					ShaderVariant shaderVariant("shader/terrain_patch.glsl");
					shaderVariant.setStage(EShaderStage::eComputeShader).setMacro(L"PATCH_CULL_PASS");

					patchPrepare = std::make_unique<ComputePipeResources>(
						shaderVariant,
						0,
						setLayouts);
				}

				{
					ShaderVariant shaderVariant("shader/terrain_patch.glsl");
					shaderVariant.setStage(EShaderStage::eComputeShader).setMacro(L"DRAW_COMMAND_PASS");

					gbufferDrawArgs = std::make_unique<ComputePipeResources>(
						shaderVariant,
						0,
						setLayouts);
				}

			}


			{
				VkDescriptorSetLayout layout = VK_NULL_HANDLE;
				getContext()->descriptorFactoryBegin()
					.bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0) // frameData
					.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1) // 
					.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  2)
					.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  3)
					.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  4)
					.buildNoInfoPush(layout);

				gbuffer = std::make_unique<GraphicPipeResources>(
					"shader/terrain_gbuffer.glsl",
					std::vector<VkDescriptorSetLayout>{ 
						layout,
						getContext()->getSamplerCache().getCommonDescriptorSetLayout(),
					},
					0,
					std::vector<VkFormat>
					{
						GBufferTextures::hdrSceneColorFormat(),
						GBufferTextures::gbufferAFormat(),
						GBufferTextures::gbufferBFormat(),
						GBufferTextures::gbufferSFormat(),
						GBufferTextures::gbufferVFormat(),
						GBufferTextures::gbufferIdFormat(),
					},
					std::vector<VkPipelineColorBlendAttachmentState>
					{
						RHIColorBlendAttachmentOpauqeState(),
						RHIColorBlendAttachmentOpauqeState(),
						RHIColorBlendAttachmentOpauqeState(),
						RHIColorBlendAttachmentOpauqeState(),
						RHIColorBlendAttachmentOpauqeState(),
						RHIColorBlendAttachmentOpauqeState(),
					},
					GBufferTextures::depthTextureFormat(),
					VK_CULL_MODE_FRONT_BIT,
					VK_COMPARE_OP_GREATER, true, true,
					std::vector<VkVertexInputAttributeDescription>
					{
						{ 0, 0, VK_FORMAT_R32G32_SFLOAT, 0 },
					},
					sizeof(float) * 2);
			}
		}
	};

	void LandscapeComponent::clearCache()
	{
		m_heightmapTextureUUID = { };
		m_heightmapImage = nullptr;

		m_heightMapHzb = nullptr;

		m_maxHeight = 400.0f;
		m_minHeight = -10.0f;
	}


	void LandscapeComponent::buildCache()
	{
		// Load height map image.
		auto asset = std::dynamic_pointer_cast<AssetTexture>(getAssetManager()->getAsset(m_heightmapTextureUUID));
		m_heightmapImage = asset->getGPUImage().lock();
	}


	bool LandscapeComponent::collectLandscape(RenderScene& renderScene, VkCommandBuffer cmd)
	{
		if (!m_heightmapTextureUUID.empty())
		{
			// Need load heightmap image.
			if (m_heightmapImage == nullptr)
			{
				buildCache();
			}
		}

		// Pre-generate mipmap.
		if (m_heightmapImage != nullptr && m_heightmapImage->isAssetReady())
		{
			// Generate heightmap hzb when change here.
			auto* pass = getContext()->getPasses().get<TerrainPass>();
			auto* rtPool = &getContext()->getRenderTargetPools();

			auto* image = m_heightmapImage->getReadyImage();

			if (m_normalMap == nullptr)
			{
				m_normalMap = rtPool->createPoolImage(
					"terrain_normalMap",
					image->getExtent().width,
					image->getExtent().height,
					VK_FORMAT_A2B10G10R10_UNORM_PACK32,
					VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);

				m_normalMap->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL);

				{
					pass->heightmap2NormalMap->bind(cmd);

					PushSetBuilder(cmd)
						.addSRV(*image)
						.addUAV(m_normalMap)
						.push(pass->heightmap2NormalMap.get());

					vkCmdDispatch(cmd, getGroupCount(image->getExtent().width, 8), getGroupCount(image->getExtent().height, 8), 1);
				}

				m_normalMap->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
			}

			// Need generate heightmap hzb.
			if(m_heightMapHzb == nullptr)
			{
				uint32_t mipStartWidth  = image->getExtent().width  / 2;
				uint32_t mipStartHeight = image->getExtent().height / 2;

				m_heightMapHzb = rtPool->createPoolImage(
					"HeightmapHzb",
					mipStartWidth,
					mipStartHeight,
					VK_FORMAT_R16G16_UNORM,
					VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
					kRenderTextureFullMip);

				// Build from src.
				HeightmapHzbPassPush push{ .kLoadSrcLevel = 1 };
				{
					pass->heightmapHzb->bind(cmd);
					pass->heightmapHzb->pushConst(cmd, &push);
					{
						VkImageSubresourceRange rangeMip0 
						{ 
							.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, 
							.baseMipLevel = 0, 
							.levelCount = 1, 
							.baseArrayLayer = 0, 
							.layerCount = 1 
						};

						m_heightMapHzb->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, rangeMip0);

						PushSetBuilder(cmd)
							.addSRV(m_heightmapImage->getSelfImage(), rangeMip0)
							.addUAV(m_heightMapHzb, rangeMip0)
							.push(pass->heightmapHzb.get());

						vkCmdDispatch(cmd, getGroupCount(mipStartWidth, 8), getGroupCount(mipStartHeight, 8), 1);

						m_heightMapHzb->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, rangeMip0);
					}
				}

				// Build from hiz.
				push.kLoadSrcLevel = 0;
				pass->heightmapHzb->pushConst(cmd, &push);
				if (m_heightMapHzb->getImage().getInfo().mipLevels > 1)
				{
					uint32_t loopWidth  = mipStartWidth;
					uint32_t loopHeight = mipStartHeight;

					for (uint32_t i = 1; i < m_heightMapHzb->getImage().getInfo().mipLevels; i++)
					{
						VkImageSubresourceRange rangeMipN_1{ 
							.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = i - 1, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1 };

						VkImageSubresourceRange rangeMipN{ 
							.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = i, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1 };

						m_heightMapHzb->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, rangeMipN_1);
						m_heightMapHzb->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, rangeMipN);

						PushSetBuilder(cmd)
							.addSRV(m_heightMapHzb, rangeMipN_1)
							.addUAV(m_heightMapHzb, rangeMipN)
							.push(pass->heightmapHzb.get());


						loopWidth  = math::max(1u, loopWidth / 2);
						loopHeight = math::max(1u, loopHeight / 2);
						vkCmdDispatch(cmd, getGroupCount(loopWidth, 8), getGroupCount(loopHeight, 8), 1);
					}
				}


				m_heightMapHzb->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
			}
		}

		// Only render when heightmap image exist.
		return (m_heightmapImage != nullptr) && (m_heightMapHzb != nullptr) && m_heightmapImage->isAssetReady();
	}

	void engine::renderTerrainSDSMDepth(
		VkCommandBuffer cmd,
		BufferParameterHandle perFrameGPU,
		GBufferTextures* inGBuffers,
		RenderScene* scene,
		SDSMInfos& sdsmInfo,
		uint32_t cascadeId)
	{
		auto* landscape = scene->getLandscape();
		if (landscape == nullptr)
		{
			return;
		}

		auto* rtPool = &getContext()->getRenderTargetPools();
		auto* pass = getContext()->getPasses().get<TerrainPass>();

		auto& cascadeBuffer = sdsmInfo.cascadeInfoBuffer;
		{


			CHECK(inGBuffers->terrainPathDispatchBuffer != nullptr);
			CHECK(inGBuffers->terrainLodCountBuffer != nullptr);
			CHECK(inGBuffers->terrainLodNodeBuffer != nullptr);

			TerrainShadowDepthSDSMPush push{};
			push.inCascadeId = cascadeId;

			pass->sdsmShadowDepth->bindAndPushConst(cmd, &push);

			auto& patchBuffer = inGBuffers->terrainPatchBufferMainView;
			auto& drawArgs = inGBuffers->terrainDrawArgsMainView;

			PushSetBuilder(cmd)
				.addBuffer(perFrameGPU)
				.addBuffer(patchBuffer)
				.addSRV(*landscape->getGPUImage()->getReadyImage())
				.addBuffer(cascadeBuffer)
				.push(pass->sdsmShadowDepth.get());

			pass->sdsmShadowDepth->bindSet(cmd, std::vector<VkDescriptorSet>{
				getContext()->getSamplerCache().getCommonDescriptorSet()
			}, 1);

			auto vB = getRenderer()->getSharedTextures().uniformGridVertices16x16->getVkBuffer();
			const VkDeviceSize vBOffset = 0;
			vkCmdBindVertexBuffers(cmd, 0, 1, &vB, &vBOffset);

			vkCmdDrawIndirect(cmd, drawArgs->getBuffer()->getVkBuffer(),
				0, 1, sizeof(uvec4));
		}
	}

	void engine::prepareTerrainLODS(
		VkCommandBuffer cmd, 
		GBufferTextures* inGBuffers, 
		RenderScene* scene, 
		BufferParameterHandle perFrameGPU, 
		GPUTimestamps* timer)
	{
		auto* landscape = scene->getLandscape();
		if (landscape == nullptr)
		{
			return;
		}

		auto* rtPool = &getContext()->getRenderTargetPools();
		auto* pass = getContext()->getPasses().get<TerrainPass>();

		uint lodCount = landscape->getLODCount();
		uint maxLODCount = kTerrainCoarseNodeDim * kTerrainCoarseNodeDim * math::pow(4, int(lodCount) - 1);

		auto readyLodNodeListBuffer = getContext()->getBufferParameters().getStaticStorage("ReadyTerrainLODNodeList", sizeof(uint) * 3 * maxLODCount);
		auto readyLodNodeCount = getContext()->getBufferParameters().getStaticStorage("ReadyTerrainLODCount", sizeof(uint));

		auto lodContinueCountersBuffer = getContext()->getBufferParameters().getStaticStorage("lodContinueCounters", sizeof(uint) * lodCount);

		auto lodContinueBuffer0 = getContext()->getBufferParameters().getStaticStorage("lodContinueBuffer0", sizeof(uint) * 2 * maxLODCount);
		auto lodContinueBuffer1 = getContext()->getBufferParameters().getStaticStorage("lodContinueBuffer1", sizeof(uint) * 2 * maxLODCount);

		auto lodCmdBuffer = getContext()->getBufferParameters().getIndirectStorage("lodCmdBuffer", sizeof(GPUDispatchIndirectCommand));
		auto patchCmdBuffer = getContext()->getBufferParameters().getIndirectStorage("patchCmdBuffer", sizeof(GPUDispatchIndirectCommand));

		int lodNodeIdCount = (16 * (1 - pow(4, lodCount))) / -3;
		auto lodNodeContinue = getContext()->getBufferParameters().getStaticStorage("lodNodeContinue", sizeof(uint) * lodNodeIdCount);

		// Clear counter buffers.
		vkCmdFillBuffer(cmd, *readyLodNodeCount->getBuffer(),         0, readyLodNodeCount->getBuffer()->getSize(),         0u);
		vkCmdFillBuffer(cmd, *lodContinueCountersBuffer->getBuffer(), 0, lodContinueCountersBuffer->getBuffer()->getSize(), 0u);

		{
			std::array<VkBufferMemoryBarrier2, 2> fillBarriers
			{
				RHIBufferBarrier(readyLodNodeCount->getBuffer()->getVkBuffer(), 
					VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT),
				RHIBufferBarrier(readyLodNodeCount->getBuffer()->getVkBuffer(), 
					VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT),
			};
			RHIPipelineBarrier(cmd, 0, (uint32_t)fillBarriers.size(), fillBarriers.data(), 0, nullptr);

		}

		uint32_t dim = math::pow(2, lodCount - 1) * 4;

		auto patchLodMap = rtPool->createPoolImage(
			"patchLodMap",
			dim,
			dim,
			VK_FORMAT_R8_UNORM,
			VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);


		// Get push const.
		TerrainLODPreparePush pushConst { };
		pushConst.coefficientLodContinue = cVarTerrainLodContinueCoefficient.get();
		pushConst.maxLodMipmap = landscape->getHeightMapHZB()->getImage().getInfo().mipLevels - 1;

		pass->lodPrepare->bind(cmd);

		// Loop lods.
		auto readBuffer = lodContinueBuffer0;
		auto writeBuffer = lodContinueBuffer1;
		bool bFirstPass = true;
		for (int i = lodCount - 1; i >= 1; i--)
		{
			pushConst.lodIndex = i;

			bool bFinalPass = i == 1;

			// Add prepare args pass.
			if (!bFirstPass)
			{

				PushSetBuilder(cmd)
					.addBuffer(perFrameGPU)
					.addBuffer(readyLodNodeListBuffer)
					.addBuffer(readyLodNodeCount)
					.addBuffer(lodContinueCountersBuffer)
					.addBuffer(readBuffer)
					.addBuffer(writeBuffer)
					.addBuffer(lodCmdBuffer)
					.addBuffer(patchCmdBuffer)
					.addSRV(landscape->getHeightMapHZB())
					.addBuffer(lodNodeContinue)
					.push(pass->lodArgs.get());

				pass->lodArgs->bindAndPushConst(cmd, &pushConst);



				vkCmdDispatch(cmd, 1, 1, 1);

				auto barrier = RHIBufferBarrier(lodCmdBuffer->getBuffer()->getVkBuffer(),
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
					VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT);

				RHIPipelineBarrier(cmd, 0, 1, &barrier, 0, nullptr);
			}

			{
				// push set.
				PushSetBuilder(cmd)
					.addBuffer(perFrameGPU)
					.addBuffer(readyLodNodeListBuffer)
					.addBuffer(readyLodNodeCount)
					.addBuffer(lodContinueCountersBuffer)
					.addBuffer(readBuffer)
					.addBuffer(writeBuffer)
					.addBuffer(lodCmdBuffer)
					.addBuffer(patchCmdBuffer)
					.addSRV(landscape->getHeightMapHZB())
					.addBuffer(lodNodeContinue)
					.push(pass->lodPrepare.get());

				pass->lodPrepare->bindAndPushConst(cmd, &pushConst);



				if (bFirstPass)
				{
					vkCmdDispatch(cmd, getGroupCount(kTerrainCoarseNodeDim * kTerrainCoarseNodeDim, 64), 1, 1);
				}
				else
				{
					vkCmdDispatchIndirect(cmd, lodCmdBuffer->getBuffer()->getVkBuffer(), 0);
				}

				std::array<VkBufferMemoryBarrier2, 2> endBufferBarriers
				{
					RHIBufferBarrier(readBuffer->getBuffer()->getVkBuffer(),
						VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_MEMORY_READ_BIT,
						VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT),

					RHIBufferBarrier(writeBuffer->getBuffer()->getVkBuffer(),
						VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
						VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT),
				};
				RHIPipelineBarrier(cmd, 0, (uint32_t)endBufferBarriers.size(), endBufferBarriers.data(), 0, nullptr);


				if (!bFirstPass && (!bFinalPass))
				{
					auto barrier = RHIBufferBarrier(lodCmdBuffer->getBuffer()->getVkBuffer(),
						VK_ACCESS_INDIRECT_COMMAND_READ_BIT, VK_ACCESS_MEMORY_READ_BIT,
						VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_MEMORY_WRITE_BIT);

					RHIPipelineBarrier(cmd, 0, 1, &barrier, 0, nullptr);
				}

				if (bFinalPass)
				{
					pass->patchArgs->bindAndPushConst(cmd, &pushConst);

					PushSetBuilder(cmd)
						.addBuffer(perFrameGPU)
						.addBuffer(readyLodNodeListBuffer)
						.addBuffer(readyLodNodeCount)
						.addBuffer(lodContinueCountersBuffer)
						.addBuffer(readBuffer)
						.addBuffer(writeBuffer)
						.addBuffer(lodCmdBuffer)
						.addBuffer(patchCmdBuffer)
						.addSRV(landscape->getHeightMapHZB())
						.addBuffer(lodNodeContinue)
						.push(pass->patchArgs.get());





					vkCmdDispatch(cmd, 1, 1, 1);

					auto barrier = RHIBufferBarrier(patchCmdBuffer->getBuffer()->getVkBuffer(),
						VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
						VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT);

					RHIPipelineBarrier(cmd, 0, 1, &barrier, 0, nullptr);
				}
			}



			// Swap.
			auto tempBuffer = readBuffer;
			readBuffer = writeBuffer;
			writeBuffer = tempBuffer;
			bFirstPass = false;
		}

		// build patch lod map.
		{

			pass->loadMap->bind(cmd);

			patchLodMap->getImage().transitionGeneral(cmd);

			PushSetBuilder(cmd)
				.addBuffer(perFrameGPU)
				.addBuffer(lodNodeContinue)
				.addUAV(patchLodMap)
				.push(pass->loadMap.get());

			vkCmdDispatch(cmd, 
				getGroupCount(patchLodMap->getImage().getExtent().width, 8), 
				getGroupCount(patchLodMap->getImage().getExtent().height, 8), 1);

			patchLodMap->getImage().transitionShaderReadOnly(cmd);
		}



		inGBuffers->terrainLodNodeBuffer = readyLodNodeListBuffer;
		inGBuffers->terrainLodCountBuffer = readyLodNodeCount;
		inGBuffers->terrainPathDispatchBuffer = patchCmdBuffer;
		inGBuffers->terrainLODPatchMap = patchLodMap;

		{
			auto patchCountBuffer = getContext()->getBufferParameters().getStaticStorage("patchCountBuffer", sizeof(uint));
			auto patchBuffer = getContext()->getBufferParameters().getStaticStorage("patchBuffer", sizeof(TerrainPatch) * maxLODCount * 64);



			// Clear counter buffers.
			{

				vkCmdFillBuffer(cmd, *patchCountBuffer->getBuffer(), 0, patchCountBuffer->getBuffer()->getSize(), 0u);

				std::array<VkBufferMemoryBarrier2, 1> fillBarriers
				{
					RHIBufferBarrier(patchCountBuffer->getBuffer()->getVkBuffer(),
						VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
						VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT),
				};
				RHIPipelineBarrier(cmd, 0, (uint32_t)fillBarriers.size(), fillBarriers.data(), 0, nullptr);
			}


			auto gbufferDrawArgs = getContext()->getBufferParameters().getIndirectStorage("gbufferDrawArgs", sizeof(uint) * 4);

			// Patch build for main view.
			{
				pass->patchPrepare->bind(cmd);

				PushSetBuilder(cmd)
					.addBuffer(perFrameGPU)
					.addBuffer(inGBuffers->terrainLodNodeBuffer)
					.addBuffer(inGBuffers->terrainLodCountBuffer)
					.addBuffer(patchCountBuffer)
					.addBuffer(patchBuffer)
					.addBuffer(gbufferDrawArgs)
					.addSRV(inGBuffers->terrainLODPatchMap)
					.push(pass->patchPrepare.get());

				vkCmdDispatchIndirect(cmd, inGBuffers->terrainPathDispatchBuffer->getBuffer()->getVkBuffer(), 0);


				std::array<VkBufferMemoryBarrier2, 2> computeBarriers
				{
					RHIBufferBarrier(patchCountBuffer->getBuffer()->getVkBuffer(),
						VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
						VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT),
					RHIBufferBarrier(patchBuffer->getBuffer()->getVkBuffer(),
						VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
						VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT),
				};
				RHIPipelineBarrier(cmd, 0, (uint32_t)computeBarriers.size(), computeBarriers.data(), 0, nullptr);
			}


			{
				pass->gbufferDrawArgs->bind(cmd);
				PushSetBuilder(cmd)
					.addBuffer(perFrameGPU)
					.addBuffer(inGBuffers->terrainLodNodeBuffer)
					.addBuffer(inGBuffers->terrainLodCountBuffer)
					.addBuffer(patchCountBuffer)
					.addBuffer(patchBuffer)
					.addBuffer(gbufferDrawArgs)
					.addSRV(inGBuffers->terrainLODPatchMap)
					.push(pass->gbufferDrawArgs.get());

				vkCmdDispatch(cmd, 1, 1, 1);

				std::array<VkBufferMemoryBarrier2, 1> computeBarriers
				{
					RHIBufferBarrier(gbufferDrawArgs->getBuffer()->getVkBuffer(),
						VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
						VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT),
				};
				RHIPipelineBarrier(cmd, 0, (uint32_t)computeBarriers.size(), computeBarriers.data(), 0, nullptr);
			}

			inGBuffers->terrainPatchBufferMainView = patchBuffer;
			inGBuffers->terrainDrawArgsMainView = gbufferDrawArgs;
		}

	}

	// Render terrain.
	void engine::renderTerrainGbuffer(
		VkCommandBuffer cmd, 
		GBufferTextures* inGBuffers, 
		RenderScene* scene, 
		BufferParameterHandle perFrameGPU, 
		GPUTimestamps* timer)
	{

		auto* landscape = scene->getLandscape();
		if (landscape == nullptr)
		{
			return;
		}

		CHECK(inGBuffers->terrainPathDispatchBuffer != nullptr);
		CHECK(inGBuffers->terrainLodCountBuffer != nullptr);
		CHECK(inGBuffers->terrainLodNodeBuffer != nullptr);

		auto* pass = getContext()->getPasses().get<TerrainPass>();
		uint lodCount = landscape->getLODCount();
		uint maxLODCount = kTerrainCoarseNodeDim * kTerrainCoarseNodeDim * math::exp2(lodCount - 1);


		auto& hdrSceneColor = inGBuffers->hdrSceneColor->getImage();
		auto& gbufferA = inGBuffers->gbufferA->getImage();
		auto& gbufferB = inGBuffers->gbufferB->getImage();
		auto& gbufferS = inGBuffers->gbufferS->getImage();
		auto& gbufferV = inGBuffers->gbufferV->getImage();
		auto& gbufferId = inGBuffers->gbufferId->getImage();
		auto& sceneDepthZ = inGBuffers->depthTexture->getImage();

		hdrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		gbufferA.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		gbufferB.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		gbufferS.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		gbufferV.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		gbufferId.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
		sceneDepthZ.transitionLayout(cmd, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));

		std::vector<VkRenderingAttachmentInfo> colorAttachments = ColorAttachmentsBuilder()
			.add(hdrSceneColor, VK_ATTACHMENT_LOAD_OP_LOAD)
			.add(gbufferA, VK_ATTACHMENT_LOAD_OP_LOAD)
			.add(gbufferB, VK_ATTACHMENT_LOAD_OP_LOAD)
			.add(gbufferS, VK_ATTACHMENT_LOAD_OP_LOAD)
			.add(gbufferV, VK_ATTACHMENT_LOAD_OP_LOAD)
			.add(gbufferId, VK_ATTACHMENT_LOAD_OP_LOAD)
			.result;

		VkRenderingAttachmentInfo depthAttachment = getDepthAttachment(sceneDepthZ, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);


		{
			ScopeRenderCmdObject renderCmdScope(cmd, timer, "Terrain GBuffer", sceneDepthZ, colorAttachments, depthAttachment);


			pass->gbuffer->bind(cmd);
			PushSetBuilder(cmd)
				.addBuffer(perFrameGPU)
				.addBuffer(inGBuffers->terrainPatchBufferMainView)
				.addSRV(*landscape->getGPUImage()->getReadyImage())
				.addSRV(inGBuffers->terrainLODPatchMap)
				.addSRV(landscape->getNormalMap())
				.push(pass->gbuffer.get());

			pass->gbuffer->bindSet(cmd, std::vector<VkDescriptorSet>{
				getContext()->getSamplerCache().getCommonDescriptorSet()
			}, 1);

			auto vB = getRenderer()->getSharedTextures().uniformGridVertices16x16->getVkBuffer();
			const VkDeviceSize vBOffset = 0;
			vkCmdBindVertexBuffers(cmd, 0, 1, &vB, &vBOffset);

			// Draw first.
			if(!cVarTerrainDebugFrame.get())
			{
				cmdSetPolygonFillMode(cmd, VK_POLYGON_MODE_FILL);
				vkCmdDrawIndirect(cmd, inGBuffers->terrainDrawArgsMainView->getBuffer()->getVkBuffer(), 
					0, 1, sizeof(uvec4));
			}
			else
			{
				cmdSetPolygonFillMode(cmd, VK_POLYGON_MODE_LINE);
				vkCmdDrawIndirect(cmd, inGBuffers->terrainDrawArgsMainView->getBuffer()->getVkBuffer(), 
					0, 1, sizeof(uvec4));
			}
		}
	}
}

