#include "../deferred_renderer.h"
#include "../render_scene.h"
#include "../renderer.h"
#include "../scene_textures.h"

namespace engine
{
    static AutoCVarBool cVarEnableDebugLine(
        "r.DebugLine.Enable",
        "Enable draw debug line or not.",
        "DebugLine",
        true,
        CVarFlags::ReadAndWrite
    );

    static AutoCVarInt32 cVarDebugLineMaxCount(
        "r.DebugLine.MaxCount",
        "Enable draw debug line max point count.",
        "DebugLine",
        1024 * 1024,
        CVarFlags::ReadAndWrite
    );

    bool isDebugLineEnable()
    {
        return cVarEnableDebugLine.get();
    }


    void DebugLineDrawContext::beforeRecord(VkCommandBuffer cmd)
    {
        if (!isDebugLineEnable())
        {
            return;
        }

        std::array<VkBufferMemoryBarrier2, 2> endBufferBarriers
        {
            RHIBufferBarrier(verticesGPU->getBuffer()->getVkBuffer(),
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT),

            RHIBufferBarrier(verticesCount->getBuffer()->getVkBuffer(),
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT),
        };
        RHIPipelineBarrier(cmd, 0, (uint32_t)endBufferBarriers.size(), endBufferBarriers.data(), 0, nullptr);
    }

    void DebugLineDrawContext::endRecord(VkCommandBuffer cmd)
    {
        if (!isDebugLineEnable())
        {
            return;
        }

        std::array<VkBufferMemoryBarrier2, 2> endBufferBarriers
        {
            RHIBufferBarrier(verticesGPU->getBuffer()->getVkBuffer(),
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT),

            RHIBufferBarrier(verticesCount->getBuffer()->getVkBuffer(),
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT),
        };
        RHIPipelineBarrier(cmd, 0, (uint32_t)endBufferBarriers.size(), endBufferBarriers.data(), 0, nullptr);
    }

    void DebugLineDrawContext::reinit(VkCommandBuffer cmd)
    {
        maxCount = isDebugLineEnable() ? cVarDebugLineMaxCount.get() * 2 : 2;

        verticesDrawCmd = getContext()->getBufferParameters().getIndirectStorage(
            "DebugLineDrawCmd", sizeof(VkDrawIndirectCommand));
        verticesCount = getContext()->getBufferParameters().getIndirectStorage(
            "DebugLineDrawCount", sizeof(uint32_t));

        verticesGPU = getContext()->getBufferParameters().getStaticStorageGPUOnly(
            "DebugLineVertices",
            sizeof(LineDrawVertex) * maxCount);

        // Clear count buffer.
        {
            auto beginBarriers = RHIBufferBarrier(verticesCount->getBuffer()->getVkBuffer(),
                VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            RHIPipelineBarrier(cmd, 0, 1, &beginBarriers, 0, nullptr);

            vkCmdFillBuffer(cmd, *verticesCount->getBuffer(), 0, verticesCount->getBuffer()->getSize(), 0u);

            auto endBarrier = RHIBufferBarrier(verticesCount->getBuffer()->getVkBuffer(),
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            RHIPipelineBarrier(cmd, 0, 1, &endBarrier, 0, nullptr);
        }
    }

    class LineDebugPass : public PassInterface
    {
    public:
        std::unique_ptr<GraphicPipeResources> draw;
        std::unique_ptr<ComputePipeResources> prepareDraw;
        std::unique_ptr<GraphicPipeResources> drawCPU;
    protected:
        virtual void onInit() override
        {
            VkDescriptorSetLayout drawSetLayout = VK_NULL_HANDLE;
            getContext()->descriptorFactoryBegin()
                .bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0) // frameData
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1)
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2)
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3)
                .buildNoInfoPush(drawSetLayout);

            auto layouts = std::vector<VkDescriptorSetLayout>
            {
                drawSetLayout,
            };

            {
                ShaderVariant shader("shader/debug_line.glsl");
                shader.setStage(EShaderStage::eComputeShader).setMacro(L"PREPARE_DRAW_CMD_PASS");

                prepareDraw = std::make_unique<ComputePipeResources>(shader, 0, layouts);
            }

            {


                ShaderVariant vertexShaderVariant("shader/debug_line.glsl");
                vertexShaderVariant.setStage(EShaderStage::eVertexShader).setMacro(L"DRAW_PASS");

                ShaderVariant fragmentShaderVariant("shader/debug_line.glsl");
                fragmentShaderVariant.setStage(EShaderStage::ePixelShader).setMacro(L"DRAW_PASS");

                VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
                colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
                colorBlendAttachment.blendEnable = VK_TRUE;
                colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
                colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
                colorBlendAttachment.colorBlendOp        = VK_BLEND_OP_ADD;
                colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
                colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
                colorBlendAttachment.alphaBlendOp        = VK_BLEND_OP_ADD;

                draw = std::make_unique<GraphicPipeResources>(
                    vertexShaderVariant,
                    fragmentShaderVariant,
                    layouts,
                    0,
                    std::vector<VkFormat>
                    {
                        getContext()->getSwapchain().getImageFormat(),
                    },
                    std::vector<VkPipelineColorBlendAttachmentState>
                    {
                        colorBlendAttachment,
                    },
                    GBufferTextures::depthTextureFormat(),
                    VK_CULL_MODE_NONE,
                    VK_COMPARE_OP_GREATER_OR_EQUAL,
                    false,
                    false,
                    std::vector<VkVertexInputAttributeDescription>{},
                    0U,
                    false,
                    VK_PRIMITIVE_TOPOLOGY_LINE_LIST);
            }

            {
                ShaderVariant vertexShaderVariant("shader/debug_line.glsl");
                vertexShaderVariant.setStage(EShaderStage::eVertexShader).setMacro(L"WORLD_DRAW_LINE_BY_CPU");

                ShaderVariant fragmentShaderVariant("shader/debug_line.glsl");
                fragmentShaderVariant.setStage(EShaderStage::ePixelShader).setMacro(L"WORLD_DRAW_LINE_BY_CPU");

                VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
                colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
                colorBlendAttachment.blendEnable = VK_TRUE;
                colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
                colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
                colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
                colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
                colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
                colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

                VkVertexInputAttributeDescription inputs
                {
                    .location = 0,
                    .binding = 0,
                    .format  = VK_FORMAT_R32G32B32A32_SFLOAT,
                    .offset  = 0,
                };
                drawCPU = std::make_unique<GraphicPipeResources>(
                    vertexShaderVariant,
                    fragmentShaderVariant,
                    layouts,
                    0,
                    std::vector<VkFormat>
                    {
                        getContext()->getSwapchain().getImageFormat(),
                    },
                    std::vector<VkPipelineColorBlendAttachmentState>
                    {
                        colorBlendAttachment,
                    },
                    GBufferTextures::depthTextureFormat(),
                    VK_CULL_MODE_NONE,
                    VK_COMPARE_OP_GREATER_OR_EQUAL,
                    false,
                    false,
                    std::vector<VkVertexInputAttributeDescription>{ inputs },
                    sizeof(vec4),
                    false,
                    VK_PRIMITIVE_TOPOLOGY_LINE_LIST);
            }
        }

        virtual void release() override
        {
            draw.reset();
            prepareDraw.reset();
            drawCPU.reset();
        }
    };

	void DeferredRenderer::renderDebugLine(
		VkCommandBuffer cmd,
		GBufferTextures* inGBuffers,
		RenderScene* scene,
		BufferParameterHandle perFrameGPU)
	{
        if (!isDebugLineEnable())
        {
            return;
        }

        auto* pass = getContext()->getPasses().get<LineDebugPass>();
        auto& ldrSceneColor = getOutput()->getImage();
        auto& sceneDepthZ = inGBuffers->depthTexture->getImage();

        PushSetBuilder setBuilder(cmd);
        setBuilder
            .addBuffer(perFrameGPU)
            .addBuffer(m_debugLine.verticesGPU)
            .addBuffer(m_debugLine.verticesDrawCmd)
            .addBuffer(m_debugLine.verticesCount)
            .push(pass->prepareDraw.get());

        // Prepare draw.
        {
            std::array<VkBufferMemoryBarrier2, 2> prepareBarriers
            {
                RHIBufferBarrier(m_debugLine.verticesGPU->getBuffer()->getVkBuffer(),
                    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT),

                RHIBufferBarrier(m_debugLine.verticesCount->getBuffer()->getVkBuffer(),
                    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT),
            };
            RHIPipelineBarrier(cmd, 0, (uint32_t)prepareBarriers.size(), prepareBarriers.data(), 0, nullptr);


            pass->prepareDraw->bind(cmd);
            vkCmdDispatch(cmd, 1, 1, 1);


            std::array<VkBufferMemoryBarrier2, 2> endBufferBarriers
            {
                RHIBufferBarrier(m_debugLine.verticesDrawCmd->getBuffer()->getVkBuffer(),
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT),

                RHIBufferBarrier(m_debugLine.verticesCount->getBuffer()->getVkBuffer(),
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT),
            };
            RHIPipelineBarrier(cmd, 0, (uint32_t)endBufferBarriers.size(), endBufferBarriers.data(), 0, nullptr);
        }

        sceneDepthZ.transitionLayout(cmd, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));
        ldrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));

        std::vector<VkRenderingAttachmentInfo> colorAttachments = ColorAttachmentsBuilder()
            .add(ldrSceneColor, VK_ATTACHMENT_LOAD_OP_LOAD)
            .result;

        VkRenderingAttachmentInfo depthAttachment = getDepthAttachment(sceneDepthZ, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);


        {
            ScopeRenderCmdObject renderCmdScope(cmd, &m_gpuTimer, "lineVerticesGPU", sceneDepthZ, colorAttachments, depthAttachment);

            cmdSetPolygonFillMode(cmd, VK_POLYGON_MODE_LINE);

            setBuilder.push(pass->draw.get());
            pass->draw->bind(cmd);
            vkCmdDrawIndirectCount(cmd,
                m_debugLine.verticesDrawCmd->getBuffer()->getVkBuffer(), 0,
                m_debugLine.verticesCount->getBuffer()->getVkBuffer(), 0,
                1,
                sizeof(VkDrawIndirectCommand));

            if (!scene->getDebugLineDrawer().empty())
            {
                auto vertexBuffer = getContext()->getBufferParameters().getParameter(
                    "debuglineCPU buffer",
                    sizeof(vec4) * scene->getDebugLineDrawer().size(),
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    VulkanBuffer::getStageCopyForUploadBufferFlags(),
                    scene->getDebugLineDrawer().data());

                pass->drawCPU->bind(cmd);

                auto vB = vertexBuffer->getBuffer()->getVkBuffer();
                const VkDeviceSize vBOffset = 0;
                vkCmdBindVertexBuffers(cmd, 0, 1, &vB, &vBOffset);

                vkCmdDraw(cmd, scene->getDebugLineDrawer().size(), 1, 0, 0);
            }
        }
	}


}