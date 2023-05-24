#include "../renderer_interface.h"
#include "../render_scene.h"
#include "../renderer.h"
#include "../scene_textures.h"

#define CBT_IMPLEMENTATION
#include "../../../shader/cbt/cbt.h"

#define LEB_IMPLEMENTATION
#include "../../../shader/leb/leb.h"

namespace engine
{
    // TODO: Terrain prez.
    //       id pick. selection.

    struct TerrainBatcherPassPush
    {
        int u_CbtID = 0;
        int u_MeshletIndexCount = 0;
    };

    struct TerrainRenderUniform
    {
        math::mat4 prevModel;
        uint32_t maskTexId;
        uint32_t pad0;
        uint32_t pad1;
        uint32_t pad2;
    };

    class CbtPass : public PassInterface
    {
    public:
        struct SharedPush
        {
            int u_CbtID  = 0;
            int u_PassID = 0;
        };

        VkDescriptorSetLayout sharedSetLayout = VK_NULL_HANDLE;
        std::unique_ptr<ComputePipeResources> sumReductionPipe;
        std::unique_ptr<ComputePipeResources> sumReductionPreparePipe;

        virtual void onInit() override
        {
            getContext()->descriptorFactoryBegin()
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0)
                .buildNoInfoPush(sharedSetLayout);

            std::vector<VkDescriptorSetLayout> layouts = 
            {
                sharedSetLayout
            };

            sumReductionPipe = std::make_unique<ComputePipeResources>("shader/cbt_sumReduction.comp.spv", (uint32_t)sizeof(SharedPush), layouts);
            sumReductionPreparePipe = std::make_unique<ComputePipeResources>("shader/cbt_sumReductionPrepass.comp.spv", (uint32_t)sizeof(SharedPush), layouts);
        }

        virtual void release() override
        {
            sumReductionPipe.reset();
            sumReductionPreparePipe.reset();
        }
    };

    class TerrainPass : public PassInterface
    {
    public:
        VkDescriptorSetLayout commonSetLayout = VK_NULL_HANDLE;
        VkDescriptorSetLayout batcherSetLayout = VK_NULL_HANDLE;

        std::unique_ptr<ComputePipeResources> batcherPipe;
        std::unique_ptr<ComputePipeResources> splitPipe;
        std::unique_ptr<ComputePipeResources> mergePipe;
        std::unique_ptr<GraphicPipeResources> renderPipe;
        std::unique_ptr<GraphicPipeResources> renderSDSMDepthPipe;

        virtual void onInit() override
        {
            getContext()->descriptorFactoryBegin()
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0)
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1)
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 2)
                .buildNoInfoPush(batcherSetLayout);
            std::vector<VkDescriptorSetLayout> batcherLayouts =
            {
                batcherSetLayout
            };
            batcherPipe = std::make_unique<ComputePipeResources>("shader/terrain_batcher.comp.spv", (uint32_t)sizeof(TerrainBatcherPassPush), batcherLayouts);


            getContext()->descriptorFactoryBegin()
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCommonShaderStage, 0)
                .bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kCommonShaderStage, 1) // uniform
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCommonShaderStage, 2)
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCommonShaderStage, 3)
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCommonShaderStage, 4)
                .buildNoInfoPush(commonSetLayout);
            std::vector<VkDescriptorSetLayout> commonLayouts = 
            {
                commonSetLayout,
                getContext()->getSamplerCache().getCommonDescriptorSetLayout(),
                getContext()->getBindlessTextureSetLayout(),
                getContext()->getDynamicUniformBuffers().getSetlayout(),
            };
            splitPipe = std::make_unique<ComputePipeResources>("shader/terrain_split.comp.spv", (uint32_t)sizeof(TerrainCommonPassPush), commonLayouts);
            mergePipe = std::make_unique<ComputePipeResources>("shader/terrain_merge.comp.spv", (uint32_t)sizeof(TerrainCommonPassPush), commonLayouts);

            renderPipe = std::make_unique<GraphicPipeResources>(
                "shader/terrain_render.vert.spv", 
                "shader/terrain_render.frag.spv",
                commonLayouts,
                (uint32_t)sizeof(TerrainCommonPassPush), 
                std::vector<VkFormat>
                {
                    GBufferTextures::hdrSceneColorFormat(),
                    GBufferTextures::gbufferAFormat(),
                    GBufferTextures::gbufferBFormat(),
                    GBufferTextures::gbufferSFormat(),
                    GBufferTextures::gbufferVFormat(),
                    GBufferTextures::getIdTextureFormat()
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
                VK_COMPARE_OP_GREATER, false, false,
                std::vector<VkVertexInputAttributeDescription>
                {
                    { 0, 0, VK_FORMAT_R32G32_SFLOAT, 0 }, // i_VertexPos
                },
                sizeof(float) * 2);

            renderSDSMDepthPipe = std::make_unique<GraphicPipeResources>(
                "shader/render_sdsm_depth.vert.spv",
                "shader/render_sdsm_depth.frag.spv",
                commonLayouts,
                (uint32_t)sizeof(TerrainCommonPassPush),
                std::vector<VkFormat>{ },
                std::vector<VkPipelineColorBlendAttachmentState>{ },
                GBufferTextures::depthTextureFormat(),
                VK_CULL_MODE_FRONT_BIT,
                VK_COMPARE_OP_GREATER, true, true,
                std::vector<VkVertexInputAttributeDescription>
                {
                    { 0, 0, VK_FORMAT_R32G32_SFLOAT, 0 }, // i_VertexPos
                },
                sizeof(float) * 2);
        }

        virtual void release() override
        {
            batcherPipe.reset();
            splitPipe.reset();
            mergePipe.reset();
            renderPipe.reset();
            renderSDSMDepthPipe.reset();
        }
    };



    void TerrainComponent::reductionLeb(VkCommandBuffer cmd, BufferParameterHandle perFrameGPU, GBufferTextures* inGBuffers, RenderScene* scene, class RendererInterface* renderer)
    {
        ScopePerframeMarker marker(cmd, "Leb reduction", { 1.0f, 1.0f, 0.0f, 1.0f });

        int it = m_setting.maxDepth;

        CbtPass::SharedPush push{ };
        push.u_CbtID = 0;
        push.u_PassID = 0;

        auto* pass = getContext()->getPasses().get<CbtPass>();

        PushSetBuilder(cmd)
            .addBuffer(m_lebBuffer)
            .push(pass->sumReductionPreparePipe.get());

        VkBufferMemoryBarrier2 endBufferBarrier = RHIBufferBarrier(m_lebBuffer->getBuffer()->getVkBuffer(),
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

        {
            int cnt = ((1 << it) >> 5); // / 2;
            int numGroup = (cnt >= 256) ? (cnt >> 8) : 1;

            push.u_PassID = it;
            pass->sumReductionPreparePipe->bindAndPushConst(cmd, &push);

            vkCmdDispatch(cmd, numGroup, 1, 1);

            RHIPipelineBarrier(cmd, 0, 1, &endBufferBarrier, 0, nullptr);
            it -= 5;
        }

        pass->sumReductionPipe->bind(cmd);
        while (--it >= 0)
        {
            int cnt = 1 << it;
            int numGroup = (cnt >= 256) ? (cnt >> 8) : 1;

            push.u_PassID = it;
            pass->sumReductionPipe->pushConst(cmd, &push);
            vkCmdDispatch(cmd, numGroup, 1, 1);

            RHIPipelineBarrier(cmd, 0, 1, &endBufferBarrier, 0, nullptr);
        }
    }

    void TerrainComponent::renderSDSMDepth(
        VkCommandBuffer cmd, 
        BufferParameterHandle perFrameGPU, 
        GBufferTextures* inGBuffers, 
        RenderScene* scene, 
        RendererInterface* renderer, 
        SDSMInfos& sdsmInfo,
        uint32_t cascadeId)
    {
        auto* pass = getContext()->getPasses().get<TerrainPass>();

        auto& sdsmDepth = sdsmInfo.shadowDepths;
        auto& cascadeBuffer = sdsmInfo.cascadeInfoBuffer;
        auto& selectionMask = inGBuffers->selectionOutlineMask->getImage();

        pass->renderSDSMDepthPipe->bind(cmd);
        PushSetBuilder(cmd)
            .addBuffer(m_lebBuffer)
            .addBuffer(perFrameGPU)
            .addSRV(m_renderContext.heightFieldImage->getImage())
            .addUAV(selectionMask)
            .addBuffer(cascadeBuffer)
            .push(pass->renderSDSMDepthPipe.get());
        pass->renderPipe->bindSet(cmd, std::vector<VkDescriptorSet>{getContext()->getSamplerCache().getCommonDescriptorSet() }, 1);

        auto vB = m_verticesBuffer->getBuffer()->getVkBuffer();
        const VkDeviceSize vBOffset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &vB, &vBOffset);
        vkCmdBindIndexBuffer(cmd, m_indicesBuffer->getBuffer()->getVkBuffer(), 0, VK_INDEX_TYPE_UINT16);

        m_renderContext.commonPushConst.cascadeId = cascadeId;
        pass->renderSDSMDepthPipe->pushConst(cmd, &m_renderContext.commonPushConst);

        vkCmdDrawIndexedIndirect(cmd,
            m_terrainDrawCmdBuffer->getBuffer()->getVkBuffer(),
            0,
            1,
            sizeof(float) * 8);
    }
    
    void TerrainComponent::updateLeb(VkCommandBuffer cmd, BufferParameterHandle perFrameGPU, GBufferTextures* inGBuffers, RenderScene* scene, class RendererInterface* renderer)
    {
        auto* pass = getContext()->getPasses().get<TerrainPass>();
        ScopePerframeMarker marker(cmd, "Leb update", { 1.0f, 1.0f, 0.0f, 1.0f });

        PushSetBuilder(cmd)
            .addBuffer(m_lebBuffer)
            .addBuffer(perFrameGPU)
            .addSRV(m_renderContext.heightFieldImage->getImage())
            .push(pass->splitPipe.get());

        pass->splitPipe->bindSet(cmd, std::vector<VkDescriptorSet>{getContext()->getSamplerCache().getCommonDescriptorSet() }, 1);

        if (m_renderContext.lebPingpong == 0)
        {
            pass->splitPipe->bindAndPushConst(cmd, &m_renderContext.commonPushConst);
        }
        else
        {
            pass->mergePipe->bindAndPushConst(cmd, &m_renderContext.commonPushConst);
        }
        vkCmdDispatchIndirect(cmd, m_dispatchCmdBuffer->getBuffer()->getVkBuffer(), 0);

        VkBufferMemoryBarrier2 endBufferBarrier = RHIBufferBarrier(m_lebBuffer->getBuffer()->getVkBuffer(),
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
        RHIPipelineBarrier(cmd, 0, 1, &endBufferBarrier, 0, nullptr);

        m_renderContext.lebPingpong = 1 - m_renderContext.lebPingpong;
    }

    void RendererInterface::renderTerrainGBuffer(
        VkCommandBuffer cmd,
        GBufferTextures* inGBuffers,
        BufferParameterHandle perFrameGPU,
        RenderScene* scene)
    {
        if (!scene->isTerrainExist())
        {
            return;
        }
        auto& terrains = scene->getTerrains();

        for (auto& terrain : terrains)
        {
            if (auto comp = terrain.lock())
            {
                comp->render(cmd, perFrameGPU, inGBuffers, scene, this);
            }
        }
    }

    void TerrainComponent::batchLeb(VkCommandBuffer cmd, BufferParameterHandle perFrameGPU, GBufferTextures* inGBuffers, RenderScene* scene, class RendererInterface* renderer)
    {
        auto* pass = getContext()->getPasses().get<TerrainPass>();
        ScopePerframeMarker marker(cmd, "Leb batcher", { 1.0f, 1.0f, 0.0f, 1.0f });

        PushSetBuilder(cmd)
            .addBuffer(m_lebBuffer)
            .addBuffer(m_terrainDrawCmdBuffer)
            .addBuffer(m_dispatchCmdBuffer)
            .push(pass->batcherPipe.get());

        TerrainBatcherPassPush pushConst{};
        pushConst.u_CbtID = 0;
        pushConst.u_MeshletIndexCount = 3 << (2 * m_setting.gpuSubd);

        pass->batcherPipe->bindAndPushConst(cmd, &pushConst);
        vkCmdDispatch(cmd, 1, 1, 1);

        std::vector<VkBufferMemoryBarrier2> endBufferBarriers =
        {
            RHIBufferBarrier(
                m_terrainDrawCmdBuffer->getBuffer()->getVkBuffer(),
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
                VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT),
            RHIBufferBarrier(
                m_dispatchCmdBuffer->getBuffer()->getVkBuffer(),
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
                VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_MEMORY_READ_BIT),
        };
        RHIPipelineBarrier(cmd, 0, (uint32_t)endBufferBarriers.size(), endBufferBarriers.data(), 0, nullptr);
    }

    void TerrainComponent::renderLeb(VkCommandBuffer cmd, BufferParameterHandle perFrameGPU, GBufferTextures* inGBuffers, RenderScene* scene, class RendererInterface* renderer)
    {
        auto* pass = getContext()->getPasses().get<TerrainPass>();

        auto& hdrSceneColor = inGBuffers->hdrSceneColor->getImage();
        auto& sceneDepthZ = inGBuffers->depthTexture->getImage();
        auto& gbufferA = inGBuffers->gbufferA->getImage();
        auto& gbufferB = inGBuffers->gbufferB->getImage();
        auto& gbufferS = inGBuffers->gbufferS->getImage();
        auto& gbufferV = inGBuffers->gbufferV->getImage();
        auto& idTexture = inGBuffers->idTexture->getImage();

        hdrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
        gbufferA.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
        gbufferB.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
        gbufferS.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
        gbufferV.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
        idTexture.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
        sceneDepthZ.transitionLayout(cmd, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));

        std::vector<VkRenderingAttachmentInfo> colorAttachments = ColorAttachmentsBuilder()
            .add(hdrSceneColor, VK_ATTACHMENT_LOAD_OP_LOAD)
            .add(gbufferA, VK_ATTACHMENT_LOAD_OP_LOAD)
            .add(gbufferB, VK_ATTACHMENT_LOAD_OP_LOAD)
            .add(gbufferS, VK_ATTACHMENT_LOAD_OP_LOAD)
            .add(gbufferV, VK_ATTACHMENT_LOAD_OP_LOAD)
            .add(idTexture, VK_ATTACHMENT_LOAD_OP_LOAD)
            .result;
        VkRenderingAttachmentInfo depthAttachment = getDepthAttachment(sceneDepthZ, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);

        auto& selectionMask = inGBuffers->selectionOutlineMask->getImage();
        selectionMask.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));

        {
            ScopeRenderCmdObject renderCmdScope(cmd, "TerrainGBuffer", sceneDepthZ, colorAttachments, depthAttachment);

            pass->renderPipe->bindAndPushConst(cmd, &m_renderContext.commonPushConst);
            TerrainRenderUniform params{};

            // params.prevModel = getNode()->getTransform()->getPrevWorldMatrix() * m_localMatrixPrev;
            params.prevModel = m_localMatrixPrev;
            auto fallbackId = getContext()->getEngineTextureWhite()->getBindlessIndex();

            params.maskTexId = isMaskSet() ? m_renderContext.grassSandMudMaskImage->getBindlessIndex() : fallbackId;
            {
                uint32_t dynamicOffset = getContext()->getDynamicUniformBuffers().alloc(sizeof(TerrainRenderUniform));
                memcpy((char*)(getContext()->getDynamicUniformBuffers().getBuffer()->getMapped()) + dynamicOffset, &params, sizeof(params));
                auto set = getContext()->getDynamicUniformBuffers().getSet();
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pass->renderPipe->pipelineLayout, 3, 1, &set, 1, &dynamicOffset);
            }

            PushSetBuilder(cmd)
                .addBuffer(m_lebBuffer)
                .addBuffer(perFrameGPU)
                .addSRV(m_renderContext.heightFieldImage->getImage())
                .addUAV(selectionMask)
                .push(pass->renderPipe.get());

            pass->renderPipe->bindSet(cmd, std::vector<VkDescriptorSet>{
                getContext()->getSamplerCache().getCommonDescriptorSet(),
                getContext()->getBindlessTextureSet(),
            }, 1);

            auto vB = m_verticesBuffer->getBuffer()->getVkBuffer();
            const VkDeviceSize vBOffset = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &vB, &vBOffset);
            vkCmdBindIndexBuffer(cmd, m_indicesBuffer->getBuffer()->getVkBuffer(), 0, VK_INDEX_TYPE_UINT16);

            vkCmdDrawIndexedIndirect(cmd,
                m_terrainDrawCmdBuffer->getBuffer()->getVkBuffer(),
                0,
                1,
                sizeof(float) * 8);
        }
    }

    void TerrainComponent::loadTexturesByUUID(bool bSync)
    {
        if (isHeightfieldSet())
        {
            m_renderContext.heightFieldImage = getContext()->getOrCreateTextureAsset(m_terrainHeightfieldId);

        }
        if (isMaskSet())
        {
            m_renderContext.grassSandMudMaskImage = getContext()->getOrCreateTextureAsset(m_terrainGrassSandMudMaskId);
        }

        if (bSync)
        {
            getContext()->waitDeviceIdle();
        }
    }

    void TerrainComponent::render(VkCommandBuffer cmd, BufferParameterHandle perFrameGPU, GBufferTextures* inGBuffers, RenderScene* scene, class RendererInterface* renderer)
    {
        if (isHeightfieldSet())
        {
            if (m_renderContext.heightFieldImage == nullptr)
            {
                loadTexturesByUUID(true);
            }
        }
        else
        {
            return;
        }

        if (!m_lebBuffer) loadLebBuffer();
        if (!m_cbtNodeCountBuffer) loadCbtNodeCountBuffer();
        if (!m_dispatchCmdBuffer || !m_terrainDrawCmdBuffer) loadRenderCmdBuffer();
        if (!m_verticesBuffer || !m_indicesBuffer) loadMeshletBuffers();

        // update common push const.
        m_renderContext.commonPushConst.u_DmapFactor = m_setting.dumpFactor;
        // m_renderContext.commonPushConst.u_ModelMatrix = getNode()->getTransform()->getWorldMatrix();
        {
            float width = m_renderContext.heightFieldImage->getImage().getExtent().width;
            float height = m_renderContext.heightFieldImage->getImage().getExtent().height;
            math::vec3 scale = math::vec3(width, 1.0f, height);
            
            m_localMatrixPrev = m_localMatrix;
            m_localMatrix =
                math::translate(glm::mat4(1.0f), math::vec3{-width * 0.5f, 0.0f, height * 0.5f }) *
                math::scale(glm::mat4(1.0f), scale)*
                math::toMat4(glm::quat(math::vec3{-math::pi<float>() * 0.5f, 0.0f, 0.0f}));
        }
        m_renderContext.commonPushConst.u_ModelMatrix = m_localMatrix;
        {
            float tmp = 2.0f * math::tan(renderer->getFrameData().camInfo.x / 2.0f) / inGBuffers->gbufferA->getImage().getExtent().height * (1 << m_setting.gpuSubd) * m_setting.primitivePixelLengthTarget;
            m_renderContext.commonPushConst.u_LodFactor = -2.0f * math::log2(tmp) + 2.0f;
        }
        m_renderContext.commonPushConst.u_MinLodVariance = m_setting.minLodStdev / 64.0f / m_renderContext.commonPushConst.u_DmapFactor;
        m_renderContext.commonPushConst.u_MinLodVariance *= m_renderContext.commonPushConst.u_MinLodVariance;
        m_renderContext.commonPushConst.bSelected = Editor::get()->getSceneNodeSelections().isSelected(SceneNodeSelctor(getNode())) ? 1U : 0U;
        m_renderContext.commonPushConst.sceneNodeId = getNode()->getId();

        // update leb.
        updateLeb(cmd, perFrameGPU, inGBuffers, scene, renderer);
        reductionLeb(cmd, perFrameGPU, inGBuffers, scene, renderer);
        batchLeb(cmd, perFrameGPU, inGBuffers, scene, renderer);
        renderLeb(cmd, perFrameGPU, inGBuffers, scene, renderer);
    }

    void TerrainComponent::setHeightField(const UUID& in)
    {
        if (in != m_terrainHeightfieldId)
        {
            m_terrainHeightfieldId = in;
            loadTexturesByUUID(true);
            markDirty();
        }
    }

    void TerrainComponent::setMask(const UUID& in)
    {
        if (in != m_terrainGrassSandMudMaskId)
        {
            m_terrainGrassSandMudMaskId = in;
            loadTexturesByUUID(false);
            markDirty();
        }
    }

    bool TerrainComponent::allBufferValid() const
    {
        return
            m_lebBuffer != nullptr &&
            m_cbtNodeCountBuffer != nullptr &&
            m_terrainDrawCmdBuffer != nullptr &&
            m_dispatchCmdBuffer != nullptr &&
            m_verticesBuffer != nullptr &&
            m_indicesBuffer != nullptr;
    }

    bool TerrainComponent::loadBuffers()
    {
        bool v = true;

        if (v) v &= loadLebBuffer();
        if (v) v &= loadRenderCmdBuffer();
        if (v) v &= loadMeshletBuffers();
        if (v) v &= loadCbtNodeCountBuffer();

        return v;
    }

    bool TerrainComponent::loadLebBuffer()
    {
        cbt_Tree* cbt = cbt_CreateAtDepth(m_setting.maxDepth, 1);
        LOG_TRACE("Loading leb sudivide buffer.");
        {
            const auto bufferSize = cbt_HeapByteSize(cbt);
            const char* data = cbt_GetHeap(cbt);
            std::string name = m_node.lock()->getName() + " leb";

            // Create new buffer.
            m_lebBuffer = getContext()->getBufferParameters().getStaticStorage(
                name.c_str(),
                bufferSize,
                (void*)data);
        }

        cbt_Release(cbt);
        return true;
    }

    bool TerrainComponent::loadCbtNodeCountBuffer()
    {
        LOG_TRACE("Loading cbt node count buffer.");
        std::string name = m_node.lock()->getName() + " cbt node count";
        m_cbtNodeCountBuffer = getContext()->getBufferParameters().getStaticStorageGPUOnly(name.c_str(), sizeof(int32_t));

        return true;
    }

    bool TerrainComponent::loadRenderCmdBuffer()
    {
        LOG_TRACE("Loading terrain cmd.");
        /*
        typedef struct VkDrawIndexedIndirectCommand {
            uint32_t    indexCount;
            uint32_t    instanceCount;
            uint32_t    firstIndex;
            int32_t     vertexOffset;
            uint32_t    firstInstance;
        } VkDrawIndexedIndirectCommand;
        */
        uint32_t drawElementsCmd[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

        /*
        typedef struct VkDispatchIndirectCommand {
            uint32_t    x;
            uint32_t    y;
            uint32_t    z;
        } VkDispatchIndirectCommand;
        */
        uint32_t dispatchCmd[8] = { 2, 1, 1, 0, 0, 0, 0, 0 };

        std::string drawElementsCmdName = m_node.lock()->getName() + " drawElementsCmd";
        m_terrainDrawCmdBuffer = getContext()->getBufferParameters().getParameter(
            drawElementsCmdName.c_str(), sizeof(drawElementsCmd),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VulkanBuffer::getStageCopyForUploadBufferFlags(),
            drawElementsCmd);

        std::string dispatchName = m_node.lock()->getName() + " dispatch cmd";
        m_dispatchCmdBuffer = getContext()->getBufferParameters().getParameter(
            dispatchName.c_str(), sizeof(dispatchCmd),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VulkanBuffer::getStageCopyForUploadBufferFlags(),
            dispatchCmd);

        return true;
    }

    bool TerrainComponent::loadMeshletBuffers()
    {
        LOG_TRACE("Loading terrain mesh lets.");

        std::vector<uint16_t> indexBuffer;
        std::vector<math::vec2> vertexBuffer;
        std::map<uint32_t, uint16_t> hashMap;

        int lebDepth = 2 * m_setting.gpuSubd;
        int triangleCount = 1 << lebDepth;
        int edgeTessellationFactor = 1 << m_setting.gpuSubd;

        // compute index and vertex buffer
        for (int i = 0; i < triangleCount; ++i)
        {
            cbt_Node node = { (uint64_t)(triangleCount + i), 2 * m_setting.gpuSubd };
            float attribArray[][3] = { {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f} };

            leb_DecodeNodeAttributeArray(node, 2, attribArray);

            for (int j = 0; j < 3; ++j)
            {
                uint32_t vertexID = attribArray[0][j] * (edgeTessellationFactor + 1)
                    + attribArray[1][j] * (edgeTessellationFactor + 1) * (edgeTessellationFactor + 1);
                auto it = hashMap.find(vertexID);

                if (it != hashMap.end())
                {
                    indexBuffer.push_back(it->second);
                }
                else
                {
                    uint16_t newIndex = (uint16_t)vertexBuffer.size();

                    indexBuffer.push_back(newIndex);
                    hashMap.insert(std::pair<uint32_t, uint16_t>(vertexID, newIndex));
                    vertexBuffer.push_back(math::vec2(attribArray[0][j], attribArray[1][j]));
                }
            }
        }

        std::string indicesName = m_node.lock()->getName() + " indices";
        m_indicesBuffer = getContext()->getBufferParameters().getParameter(
            indicesName.c_str(), sizeof(indexBuffer[0]) * indexBuffer.size(),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VulkanBuffer::getStageCopyForUploadBufferFlags(),
            &indexBuffer[0]);

        std::string verticesName = m_node.lock()->getName() + " vertices";
        m_verticesBuffer = getContext()->getBufferParameters().getParameter(
            verticesName.c_str(), sizeof(vertexBuffer[0]) * vertexBuffer.size(),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VulkanBuffer::getStageCopyForUploadBufferFlags(),
            &vertexBuffer[0]);

        return true;
    }
}