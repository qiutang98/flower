#include "../renderer_interface.h"
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
        VkDescriptorSetLayout prepassCullSetLayout = VK_NULL_HANDLE;
        VkDescriptorSetLayout prepassSetLayout = VK_NULL_HANDLE;

        std::unique_ptr<ComputePipeResources> gbuffer_cull;
        std::unique_ptr<GraphicPipeResources> gbuffer;
        VkDescriptorSetLayout gbufferCullSetLayout = VK_NULL_HANDLE;
        VkDescriptorSetLayout gbufferSetLayout = VK_NULL_HANDLE;

    protected:
        virtual void onInit() override
        {
            getContext()->descriptorFactoryBegin()
                .bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kCommonShaderStage, 0) // frameData
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCommonShaderStage, 1) // objectDatas
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCommonShaderStage, 2) // indirectCommands
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCommonShaderStage, 3) // drawCount
                .buildNoInfoPush(prepassCullSetLayout);
            prepass_cull = std::make_unique<ComputePipeResources>("shader/staticmesh_prepass_cull.comp.spv", (uint32_t)sizeof(GPUCullingPrepassPushConstants),
                std::vector<VkDescriptorSetLayout>
                {
                    prepassCullSetLayout
                });

            getContext()->descriptorFactoryBegin()
                .bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kCommonShaderStage, 0) // frameData
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCommonShaderStage, 1) // objectDatas
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCommonShaderStage, 2) // indirectCommands
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  kCommonShaderStage, 3)
                .buildNoInfoPush(prepassSetLayout);
            prepass = std::make_unique<GraphicPipeResources>(
                "shader/staticmesh_prepass.vert.spv",
                "shader/staticmesh_prepass.frag.spv",
                std::vector<VkDescriptorSetLayout>
                {
                      prepassSetLayout
                    , m_context->getBindlessSSBOSetLayout()
                    , m_context->getBindlessSSBOSetLayout()
                    , m_context->getBindlessTextureSetLayout()
                    , m_context->getBindlessSamplerSetLayout()
                },
                0,
                std::vector<VkFormat>{ },
                std::vector<VkPipelineColorBlendAttachmentState>{ },
                GBufferTextures::depthTextureFormat());

            getContext()->descriptorFactoryBegin()
                .bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kCommonShaderStage, 0) // frameData
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCommonShaderStage, 1) // objectDatas
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCommonShaderStage, 2) // indirectCommands
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCommonShaderStage, 3) // drawCount
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  kCommonShaderStage, 4) // inHzb
                .buildNoInfoPush(gbufferCullSetLayout);
            gbuffer_cull = std::make_unique<ComputePipeResources>("shader/staticmesh_cull.comp.spv", (uint32_t)sizeof(GPUCullingGbufferPushConstants),
                std::vector<VkDescriptorSetLayout>
                {
                    gbufferCullSetLayout
                });

            getContext()->descriptorFactoryBegin()
                .bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kCommonShaderStage, 0) // frameData
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCommonShaderStage, 1) // objectDatas
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCommonShaderStage, 2) // indirectCommands
                .buildNoInfoPush(gbufferSetLayout);
            gbuffer = std::make_unique<GraphicPipeResources>(
                "shader/staticmesh_gbuffer.vert.spv",
                "shader/staticmesh_gbuffer.frag.spv",
                std::vector<VkDescriptorSetLayout>
                {
                      gbufferSetLayout
                    , m_context->getBindlessSSBOSetLayout()
                    , m_context->getBindlessSSBOSetLayout()
                    , m_context->getBindlessTextureSetLayout()
                    , m_context->getBindlessSamplerSetLayout()
                },
                0,
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
                VK_COMPARE_OP_EQUAL);
        }

        virtual void release() override
        {
            prepass_cull.reset();
            prepass.reset();

            gbuffer_cull.reset();
            gbuffer.reset();
        }
    };

    void RendererInterface::renderStaticMeshPrepass(VkCommandBuffer cmd, GBufferTextures* inGBuffers, RenderScene* scene, BufferParameterHandle perFrameGPU)
    {
        const uint32_t staticMeshCount = (uint32_t)scene->getStaticMeshObjects().size();
        if (staticMeshCount <= 0)
        {
            return;
        }

        auto& sceneDepthZ = inGBuffers->depthTexture->getImage();
        VkRenderingAttachmentInfo depthAttachment = getDepthAttachment(sceneDepthZ);

        auto indirectDrawCommandBuffer = m_context->getBufferParameters().getIndirectStorage("StaticMeshIndirectCommand_Prepass", sizeof(GPUStaticMeshDrawCommand) * staticMeshCount);
        auto indirectDrawCountBuffer = m_context->getBufferParameters().getIndirectStorage("StaticMeshIndirectCount_Prepass", sizeof(uint32_t));

        auto* pass = m_context->getPasses().get<StaticMeshPass>();

        // Culling.
        {
            ScopePerframeMarker staticMeshGBufferCullingMarker(cmd, "StaticMeshCulling_prepass", { 1.0f, 0.0f, 0.0f, 1.0f });

            vkCmdFillBuffer(cmd, *indirectDrawCountBuffer->getBuffer(), 0, indirectDrawCountBuffer->getBuffer()->getSize(), 0u);

            auto fillBarriers = RHIBufferBarrier(indirectDrawCountBuffer->getBuffer()->getVkBuffer(),
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            RHIPipelineBarrier(cmd, 0, 1, &fillBarriers, 0, nullptr);


            GPUCullingPrepassPushConstants gpuPushConstant =
            {
                .cullCount = staticMeshCount,
            };

            pass->prepass_cull->bindAndPushConst(cmd, &gpuPushConstant);

            PushSetBuilder(cmd)
                .addBuffer(perFrameGPU)
                .addBuffer(scene->getStaticMeshObjectsGPU())
                .addBuffer(indirectDrawCommandBuffer)
                .addBuffer(indirectDrawCountBuffer)
                .push(pass->prepass_cull.get());

            vkCmdDispatch(cmd, getGroupCount(staticMeshCount, 64), 1, 1);

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
        m_gpuTimer.getTimeStamp(cmd, "StaticMesh Culling");

        auto& selectionMask = inGBuffers->selectionOutlineMask->getImage();
        sceneDepthZ.transitionLayout(cmd, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));
        selectionMask.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
        {
            ScopeRenderCmdObject renderCmdScope(cmd, "StaticMesh_Prepass", sceneDepthZ, {}, depthAttachment);

            pass->prepass->bind(cmd);
            PushSetBuilder(cmd)
                .addBuffer(perFrameGPU)
                .addBuffer(scene->getStaticMeshObjectsGPU())
                .addBuffer(indirectDrawCommandBuffer)
                .addUAV(selectionMask)
                .push(pass->prepass.get());

            pass->prepass->bindSet(cmd, std::vector<VkDescriptorSet>{
                 m_context->getBindlessSSBOSet()
               , m_context->getBindlessSSBOSet()
               , m_context->getBindlessTextureSet()
               , m_context->getBindlessSamplerSet()
            }, 1);

            vkCmdDrawIndirectCount(cmd,
                indirectDrawCommandBuffer->getBuffer()->getVkBuffer(), 0,
                indirectDrawCountBuffer->getBuffer()->getVkBuffer(),
                0,
                staticMeshCount,
                sizeof(GPUStaticMeshDrawCommand)
            );
        }
        m_gpuTimer.getTimeStamp(cmd, "StaticMesh Prepass Rendering");
    }


	void RendererInterface::renderStaticMeshGBuffer(
        VkCommandBuffer cmd, 
        GBufferTextures* inGBuffers, 
        RenderScene* scene, 
        BufferParameterHandle perFrameGPU, 
        PoolImageSharedRef hzbFurthest)
	{
        auto& hdrSceneColor = inGBuffers->hdrSceneColor->getImage();
        auto& gbufferA = inGBuffers->gbufferA->getImage();
        auto& gbufferB = inGBuffers->gbufferB->getImage();
        auto& gbufferS = inGBuffers->gbufferS->getImage();
        auto& gbufferV = inGBuffers->gbufferV->getImage();
        auto& sceneDepthZ = inGBuffers->depthTexture->getImage();
        auto& idTexture = inGBuffers->idTexture->getImage();

        auto rtsLayout2Attachment = [&]()
        {
            hdrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
            gbufferA.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
            gbufferB.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
            gbufferS.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
            gbufferV.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
            idTexture.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
            sceneDepthZ.transitionLayout(cmd, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));
        };

        std::vector<VkRenderingAttachmentInfo> colorAttachments = ColorAttachmentsBuilder()
            .add(hdrSceneColor, VK_ATTACHMENT_LOAD_OP_LOAD)
            .add(gbufferA, VK_ATTACHMENT_LOAD_OP_LOAD)
            .add(gbufferB, VK_ATTACHMENT_LOAD_OP_LOAD)
            .add(gbufferS, VK_ATTACHMENT_LOAD_OP_LOAD)
            .add(gbufferV, VK_ATTACHMENT_LOAD_OP_LOAD)
            .add(idTexture, VK_ATTACHMENT_LOAD_OP_LOAD)
            .result;

        VkRenderingAttachmentInfo depthAttachment = getDepthAttachment(sceneDepthZ, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);

        const uint32_t staticMeshCount = (uint32_t)scene->getStaticMeshObjects().size();
        if (staticMeshCount <= 0)
        {
            rtsLayout2Attachment();
            ScopeRenderCmdObject renderCmdScope(cmd, "StaticMeshGBuffer", sceneDepthZ, colorAttachments, depthAttachment);

            // Pre-return if no static mesh can use.
            return;
        }

        auto indirectDrawCommandBuffer = m_context->getBufferParameters().getIndirectStorage("StaticMeshIndirectCommand", sizeof(GPUStaticMeshDrawCommand) * staticMeshCount);
        auto indirectDrawCountBuffer = m_context->getBufferParameters().getIndirectStorage("StaticMeshIndirectCount", sizeof(uint32_t));

        auto* pass = m_context->getPasses().get<StaticMeshPass>();

        // Culling.
        {
            ScopePerframeMarker staticMeshGBufferCullingMarker(cmd, "StaticMeshGBufferCulling", { 1.0f, 0.0f, 0.0f, 1.0f });

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
                .cullCount = staticMeshCount,
                .hzbMipCount = hzbFurthest->getImage().getInfo().mipLevels,
                .hzbSrcSize = math::vec2( hzbFurthest->getImage().getExtent().width, hzbFurthest->getImage().getExtent().height)
            };

            hzbFurthest->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

            pass->gbuffer_cull->bindAndPushConst(cmd, &gpuPushConstant);
            PushSetBuilder(cmd)
                .addBuffer(perFrameGPU)
                .addBuffer(scene->getStaticMeshObjectsGPU())
                .addBuffer(indirectDrawCommandBuffer)
                .addBuffer(indirectDrawCountBuffer)
                .addSRV(hzbFurthest)
                .push(pass->gbuffer_cull.get());

            vkCmdDispatch(cmd, getGroupCount(staticMeshCount, 64), 1, 1);

            m_gpuTimer.getTimeStamp(cmd, "StaticMesh Culling");

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

        rtsLayout2Attachment();

        {
            ScopeRenderCmdObject renderCmdScope(cmd, "StaticMeshGBuffer", sceneDepthZ, colorAttachments, depthAttachment);

            pass->gbuffer->bind(cmd);
            PushSetBuilder(cmd)
                .addBuffer(perFrameGPU)
                .addBuffer(scene->getStaticMeshObjectsGPU())
                .addBuffer(indirectDrawCommandBuffer)
                .push(pass->gbuffer.get());

            pass->gbuffer->bindSet(cmd, std::vector<VkDescriptorSet>{
                m_context->getBindlessSSBOSet()
              , m_context->getBindlessSSBOSet()
              , m_context->getBindlessTextureSet()
              , m_context->getBindlessSamplerSet()
            }, 1);

            vkCmdDrawIndirectCount(cmd,
                indirectDrawCommandBuffer->getBuffer()->getVkBuffer(), 0,
                indirectDrawCountBuffer->getBuffer()->getVkBuffer(),
                0,
                staticMeshCount,
                sizeof(GPUStaticMeshDrawCommand)
            );

            m_gpuTimer.getTimeStamp(cmd, "StaticMesh Rendering");
        }

	}

}