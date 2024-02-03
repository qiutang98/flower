#include "../deferred_renderer.h"
#include "../render_scene.h"
#include "../renderer.h"
#include "../scene_textures.h"

namespace engine
{
    struct GPUCullingPrepassPushConstants
    {
        uint32_t cullCount;
    };

    struct GPUCullingGbufferPushConstants
    {
        uint32_t cullCount;
        uint32_t hzbMipCount;
        glm::vec2 hzbSrcSize;
    };

    class StaticMeshPass : public PassInterface
    {
    public:
        std::unique_ptr<ComputePipeResources> prepass_cull;
        std::unique_ptr<GraphicPipeResources> prepass;



        std::unique_ptr<ComputePipeResources> gbuffer_cull;
        std::unique_ptr<GraphicPipeResources> gbuffer;



    protected:
        virtual void onInit() override
        {
            {
                VkDescriptorSetLayout prepassCullSetLayout = VK_NULL_HANDLE;
                getContext()->descriptorFactoryBegin()
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0) // frameData
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1) // objectDatas
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2) // indirectCommands
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3) // drawCount
                    .buildNoInfoPush(prepassCullSetLayout);

                ShaderVariant shaderVariant("shader/static_mesh.glsl");
                shaderVariant.setStage(EShaderStage::eComputeShader).setMacro(L"STATIC_MESH_PREPASS_CULL_PASS");

                prepass_cull = std::make_unique<ComputePipeResources>(
                    shaderVariant,
                    (uint32_t)sizeof(GPUCullingPrepassPushConstants),
                    std::vector<VkDescriptorSetLayout>{ prepassCullSetLayout });
            }


            {
                VkDescriptorSetLayout prepassSetLayout = VK_NULL_HANDLE;
                getContext()->descriptorFactoryBegin()
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0) // frameData
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1) // objectDatas
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2) // indirectCommands
                    .buildNoInfoPush(prepassSetLayout);

                ShaderVariant vertexShaderVariant("shader/static_mesh.glsl");
                vertexShaderVariant.setStage(EShaderStage::eVertexShader).setMacro(L"STATIC_MESH_PREPASS");

                ShaderVariant fragmentShaderVariant("shader/static_mesh.glsl");
                fragmentShaderVariant.setStage(EShaderStage::ePixelShader).setMacro(L"STATIC_MESH_PREPASS");

                prepass = std::make_unique<GraphicPipeResources>(
                    vertexShaderVariant,
                    fragmentShaderVariant,
                    std::vector<VkDescriptorSetLayout>
                    {
                        prepassSetLayout, 
                        m_context->getBindlessSSBOSetLayout(), 
                        m_context->getBindlessSSBOSetLayout(), 
                        m_context->getBindlessTextureSetLayout(),   
                        m_context->getBindlessSamplerSetLayout(),
                    },
                    0,
                    std::vector<VkFormat>{ },
                    std::vector<VkPipelineColorBlendAttachmentState>{ },
                    GBufferTextures::depthTextureFormat(),
                    VK_CULL_MODE_NONE);
            }

            {
                VkDescriptorSetLayout gbufferCullSetLayout = VK_NULL_HANDLE;
                getContext()->descriptorFactoryBegin()
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0) // frameData
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1) // objectDatas
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2) // indirectCommands
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3) // drawCount
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 4) // inHzb
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5) // indirectCommands
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6) // drawCount
                    .buildNoInfoPush(gbufferCullSetLayout);

                ShaderVariant shaderVariant("shader/static_mesh.glsl");
                shaderVariant.setStage(EShaderStage::eComputeShader).setMacro(L"STATIC_MESH_GBUFFER_CULL_PASS");

                gbuffer_cull = std::make_unique<ComputePipeResources>(
                    shaderVariant,
                    (uint32_t)sizeof(GPUCullingGbufferPushConstants),
                    std::vector<VkDescriptorSetLayout>{ gbufferCullSetLayout });
            }

            {
                VkDescriptorSetLayout gbufferSetLayout = VK_NULL_HANDLE;
                getContext()->descriptorFactoryBegin()
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0) // frameData
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1) // objectDatas
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2) // indirectCommands
                    .buildNoInfoPush(gbufferSetLayout);

                ShaderVariant vertexShaderVariant("shader/static_mesh.glsl");
                vertexShaderVariant.setStage(EShaderStage::eVertexShader).setMacro(L"STATIC_MESH_GBUFFER_PASS");

                ShaderVariant fragmentShaderVariant("shader/static_mesh.glsl");
                fragmentShaderVariant.setStage(EShaderStage::ePixelShader).setMacro(L"STATIC_MESH_GBUFFER_PASS");

                gbuffer = std::make_unique<GraphicPipeResources>(
                    vertexShaderVariant,
                    fragmentShaderVariant,
                    std::vector<VkDescriptorSetLayout>
                    {
                        gbufferSetLayout, 
                        m_context->getBindlessSSBOSetLayout(), 
                        m_context->getBindlessSSBOSetLayout(), 
                        m_context->getBindlessTextureSetLayout(), 
                        m_context->getBindlessSamplerSetLayout(),
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
                    VK_CULL_MODE_NONE,
                    VK_COMPARE_OP_GREATER_OR_EQUAL);
            }
        }

        virtual void release() override
        {
            prepass_cull.reset();
            prepass.reset();

            gbuffer_cull.reset();
            gbuffer.reset();
        }
    };

    void engine::renderStaticMeshPrepass(
        VkCommandBuffer cmd, 
        GBufferTextures* inGBuffers, 
        RenderScene* scene, 
        BufferParameterHandle perFrameGPU,
        GPUTimestamps* timer)
    {
        const uint32_t objectCount = (uint32_t)scene->getObjectCollector().size();
        if (objectCount <= 0)
        {
            return;
        }

        auto& sceneDepthZ = inGBuffers->depthTexture->getImage();
        VkRenderingAttachmentInfo depthAttachment = getDepthAttachment(sceneDepthZ);

        auto indirectDrawCommandBuffer = getContext()->getBufferParameters().getIndirectStorage(
            "StaticMeshIndirectCommand_Prepass", sizeof(StaticMeshDrawCommand) * objectCount);

        auto indirectDrawCountBuffer = getContext()->getBufferParameters().getIndirectStorage(
            "StaticMeshIndirectCount_Prepass", sizeof(uint32_t));

        auto* pass = getContext()->getPasses().get<StaticMeshPass>();

        // Culling.
        {
            ScopePerframeMarker staticMeshGBufferCullingMarker(cmd, "StaticMeshCulling_prepass", { 1.0f, 0.0f, 0.0f, 1.0f }, timer);

            vkCmdFillBuffer(cmd, *indirectDrawCountBuffer->getBuffer(), 0, indirectDrawCountBuffer->getBuffer()->getSize(), 0u);

            auto fillBarriers = RHIBufferBarrier(indirectDrawCountBuffer->getBuffer()->getVkBuffer(),
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            RHIPipelineBarrier(cmd, 0, 1, &fillBarriers, 0, nullptr);


            GPUCullingPrepassPushConstants gpuPushConstant =
            {
                .cullCount = objectCount,
            };

            pass->prepass_cull->bindAndPushConst(cmd, &gpuPushConstant);

            PushSetBuilder(cmd)
                .addBuffer(perFrameGPU)
                .addBuffer(scene->getObjectBufferGPU())
                .addBuffer(indirectDrawCommandBuffer)
                .addBuffer(indirectDrawCountBuffer)
                .push(pass->prepass_cull.get());

            vkCmdDispatch(cmd, getGroupCount(objectCount, 64), 1, 1);

            // End buffer barrier.
            std::array<VkBufferMemoryBarrier2, 2> endBufferBarriers
            {
                RHIBufferBarrier(indirectDrawCommandBuffer->getBuffer()->getVkBuffer(),
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT),
                RHIBufferBarrier(indirectDrawCountBuffer->getBuffer()->getVkBuffer(),
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT),
            };
            RHIPipelineBarrier(cmd, 0, (uint32_t)endBufferBarriers.size(), endBufferBarriers.data(), 0, nullptr);
        }

        sceneDepthZ.transitionLayout(cmd, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));
        {
            ScopeRenderCmdObject renderCmdScope(cmd, timer, "StaticMesh_Prepass", sceneDepthZ, {}, depthAttachment);

            pass->prepass->bind(cmd);
            PushSetBuilder(cmd)
                .addBuffer(perFrameGPU)
                .addBuffer(scene->getObjectBufferGPU())
                .addBuffer(indirectDrawCommandBuffer)
                .push(pass->prepass.get());

            pass->prepass->bindSet(cmd, std::vector<VkDescriptorSet>{
                getContext()->getBindlessSSBOSet(), 
                getContext()->getBindlessSSBOSet(), 
                getContext()->getBindlessTextureSet(), 
                getContext()->getBindlessSamplerSet()
            }, 1);

            vkCmdDrawIndirectCount(cmd,
                indirectDrawCommandBuffer->getBuffer()->getVkBuffer(), 0,
                indirectDrawCountBuffer->getBuffer()->getVkBuffer(),
                0,
                objectCount,
                sizeof(StaticMeshDrawCommand)
            );
        }
    }


    void engine::renderStaticMeshGBuffer(
        VkCommandBuffer cmd,
        GBufferTextures* inGBuffers,
        RenderScene* scene,
        BufferParameterHandle perFrameGPU,
        PoolImageSharedRef hzbFurthest,
        GPUTimestamps* timer,
        DebugLineDrawContext* debugLiner)
    {
        auto& hdrSceneColor = inGBuffers->hdrSceneColor->getImage();
        auto& gbufferA = inGBuffers->gbufferA->getImage();
        auto& gbufferB = inGBuffers->gbufferB->getImage();
        auto& gbufferS = inGBuffers->gbufferS->getImage();
        auto& gbufferV = inGBuffers->gbufferV->getImage();
        auto& gbufferId = inGBuffers->gbufferId->getImage();
        auto& sceneDepthZ = inGBuffers->depthTexture->getImage();

        auto rtsLayout2Attachment = [&]()
        {
            hdrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
            gbufferA.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
            gbufferB.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
            gbufferS.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
            gbufferV.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
            gbufferId.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
            sceneDepthZ.transitionLayout(cmd, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));
        };

        std::vector<VkRenderingAttachmentInfo> colorAttachments = ColorAttachmentsBuilder()
            .add(hdrSceneColor, VK_ATTACHMENT_LOAD_OP_LOAD)
            .add(gbufferA, VK_ATTACHMENT_LOAD_OP_LOAD)
            .add(gbufferB, VK_ATTACHMENT_LOAD_OP_LOAD)
            .add(gbufferS, VK_ATTACHMENT_LOAD_OP_LOAD)
            .add(gbufferV, VK_ATTACHMENT_LOAD_OP_LOAD)
            .add(gbufferId, VK_ATTACHMENT_LOAD_OP_LOAD)
            .result;

        VkRenderingAttachmentInfo depthAttachment = getDepthAttachment(sceneDepthZ, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);

        const uint32_t objectCount = (uint32_t)scene->getObjectCollector().size();
        if (objectCount <= 0)
        {
            rtsLayout2Attachment();
            ScopeRenderCmdObject renderCmdScope(cmd, timer, "StaticMeshGBuffer", sceneDepthZ, colorAttachments, depthAttachment);

            // Pre-return if no static mesh can use.
            return;
        }

        auto indirectDrawCommandBuffer = getContext()->getBufferParameters().getIndirectStorage(
            "StaticMeshIndirectCommand", 
            sizeof(StaticMeshDrawCommand) * objectCount);

        auto indirectDrawCountBuffer = getContext()->getBufferParameters().getIndirectStorage(
            "StaticMeshIndirectCount", sizeof(uint32_t));

        auto* pass = getContext()->getPasses().get<StaticMeshPass>();

        // Culling.
        {
            ScopePerframeMarker staticMeshGBufferCullingMarker(cmd, "StaticMeshGBufferCulling", { 1.0f, 0.0f, 0.0f, 1.0f }, timer);

            auto beginBarriers = RHIBufferBarrier(
                indirectDrawCountBuffer->getBuffer()->getVkBuffer(),
                VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            RHIPipelineBarrier(cmd, 0, 1, &beginBarriers, 0, nullptr);

            vkCmdFillBuffer(cmd, *indirectDrawCountBuffer->getBuffer(), 0, indirectDrawCountBuffer->getBuffer()->getSize(), 0u);

            std::array<VkBufferMemoryBarrier2, 2> fillBarriers
            {
                RHIBufferBarrier(indirectDrawCommandBuffer->getBuffer()->getVkBuffer(),
                    VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT),

                RHIBufferBarrier(indirectDrawCountBuffer->getBuffer()->getVkBuffer(),
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT),
            };
            RHIPipelineBarrier(cmd, 0, (uint32_t)fillBarriers.size(), fillBarriers.data(), 0, nullptr);

            GPUCullingGbufferPushConstants gpuPushConstant =
            {
                .cullCount = objectCount,
                .hzbMipCount = hzbFurthest->getImage().getInfo().mipLevels,
                .hzbSrcSize = math::vec2(hzbFurthest->getImage().getExtent().width, hzbFurthest->getImage().getExtent().height)
            };

            hzbFurthest->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

            if (debugLiner)
            {
                debugLiner->beforeRecord(cmd);
            }
            

            pass->gbuffer_cull->bindAndPushConst(cmd, &gpuPushConstant);
            PushSetBuilder(cmd)
                .addBuffer(perFrameGPU)
                .addBuffer(scene->getObjectBufferGPU())
                .addBuffer(indirectDrawCommandBuffer)
                .addBuffer(indirectDrawCountBuffer)
                .addSRV(hzbFurthest)
                .addBuffer(debugLiner ? *debugLiner->verticesGPU->getBuffer() : getRenderer()->getSSBODump(), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .addBuffer(debugLiner ? *debugLiner->verticesCount->getBuffer() : getRenderer()->getSSBODump(), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .push(pass->gbuffer_cull.get());

            vkCmdDispatch(cmd, getGroupCount(objectCount, 64), 1, 1);

            // End buffer barrier.
            std::array<VkBufferMemoryBarrier2, 2> endBufferBarriers
            {
                RHIBufferBarrier(indirectDrawCommandBuffer->getBuffer()->getVkBuffer(),
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT),

                RHIBufferBarrier(indirectDrawCountBuffer->getBuffer()->getVkBuffer(),
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT),
            };
            RHIPipelineBarrier(cmd, 0, (uint32_t)endBufferBarriers.size(), endBufferBarriers.data(), 0, nullptr);

            if (debugLiner)
            {
                debugLiner->endRecord(cmd);
            }
        }

        rtsLayout2Attachment();

        {
            ScopeRenderCmdObject renderCmdScope(cmd, timer, "StaticMeshGBuffer", sceneDepthZ, colorAttachments, depthAttachment);

            pass->gbuffer->bind(cmd);
            PushSetBuilder(cmd)
                .addBuffer(perFrameGPU)
                .addBuffer(scene->getObjectBufferGPU())
                .addBuffer(indirectDrawCommandBuffer)
                .push(pass->gbuffer.get());

            pass->gbuffer->bindSet(cmd, std::vector<VkDescriptorSet>{
                getContext()->getBindlessSSBOSet(), 
                getContext()->getBindlessSSBOSet(), 
                getContext()->getBindlessTextureSet(), 
                getContext()->getBindlessSamplerSet()
            }, 1);

            vkCmdDrawIndirectCount(cmd,
                indirectDrawCommandBuffer->getBuffer()->getVkBuffer(), 0,
                indirectDrawCountBuffer->getBuffer()->getVkBuffer(),
                0,
                objectCount,
                sizeof(StaticMeshDrawCommand)
            );
        }

    }

}