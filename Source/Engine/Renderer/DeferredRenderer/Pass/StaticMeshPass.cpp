#include "Pch.h"
#include "../DeferredRenderer.h"
#include "../../Renderer.h"
#include "../../RenderSceneData.h"
#include "../../../AssetSystem/MeshManager.h"
#include "../../RendererTextures.h"
#include "../../SceneTextures.h"
#include "../../../AssetSystem/TextureManager.h"

namespace Flower
{
    struct GPUCullingPushConstants
    {
        uint32_t cullCount;
    };

    class StaticMeshPass : public PassInterface
    {
    public:
        VkPipeline cullingPipeline = VK_NULL_HANDLE;
        VkPipelineLayout cullingPipelineLayout = VK_NULL_HANDLE;
        VkPipeline gbufferPipeline = VK_NULL_HANDLE;
        VkPipelineLayout gbufferPipelineLayout = VK_NULL_HANDLE;

    protected:
        virtual void init() override
        {
            initCulling();
            initGBuffer();
        }

        virtual void release() override
        {
            RHISafeRelease(cullingPipeline);
            RHISafeRelease(cullingPipelineLayout);

            RHISafeRelease(gbufferPipeline);
            RHISafeRelease(gbufferPipelineLayout);
        }

    private:
        void initCulling()
        {
            CHECK(cullingPipeline == VK_NULL_HANDLE);
            CHECK(cullingPipelineLayout == VK_NULL_HANDLE);

            // Config.
            auto shaderModule = RHI::ShaderManager->getShader("StaticMeshCulling.comp.spv", true);
            std::vector<VkDescriptorSetLayout> setLayouts =
            {
                  GetLayoutStatic(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) // objectDatas
                , GetLayoutStatic(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) // indirectCommands
                , GetLayoutStatic(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) // drawCount
                , GetLayoutStatic(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) // viewData
            };

            // Vulkan build functions.
            VkPipelineLayoutCreateInfo plci = RHIPipelineLayoutCreateInfo();
            VkPushConstantRange push_constant{};
            push_constant.offset = 0;
            push_constant.size = sizeof(GPUCullingPushConstants);
            push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            plci.pPushConstantRanges = &push_constant;
            plci.pushConstantRangeCount = 1;
            plci.setLayoutCount = (uint32_t)setLayouts.size();
            plci.pSetLayouts = setLayouts.data();
            cullingPipelineLayout = RHI::get()->createPipelineLayout(plci);
            VkPipelineShaderStageCreateInfo shaderStageCI{};
            shaderStageCI.module = shaderModule;
            shaderStageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            shaderStageCI.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            shaderStageCI.pName = "main";
            VkComputePipelineCreateInfo computePipelineCreateInfo{};
            computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            computePipelineCreateInfo.layout = cullingPipelineLayout;
            computePipelineCreateInfo.flags = 0;
            computePipelineCreateInfo.stage = shaderStageCI;
            RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &cullingPipeline));
        }

        void initGBuffer()
        {
            CHECK(gbufferPipeline == VK_NULL_HANDLE);
            CHECK(gbufferPipelineLayout == VK_NULL_HANDLE);

            auto vertShader = RHI::ShaderManager->getShader("StaticMeshGBuffer.vert.spv", true);
            auto fragShader = RHI::ShaderManager->getShader("StaticMeshGBuffer.frag.spv", true);

            std::vector<VkDescriptorSetLayout> setLayouts =
            {
                  GetLayoutStatic(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) // viewData
                , GetLayoutStatic(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) // frameData
                , MeshManager::get()->getBindlessVertexBuffers()->getSetLayout() // verticesArray
                , MeshManager::get()->getBindlessIndexBuffers()->getSetLayout()  // indicesArray
                , Bindless::Texture->getSetLayout() // texture2D array
                , Bindless::Sampler->getSetLayout() // sampler2D array
                , GetLayoutStatic(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) // objectDatas
                , GetLayoutStatic(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) // indirectCommands
            };

            std::vector<VkPipelineShaderStageCreateInfo> shaderStages =
            {
                RHIPipelineShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, vertShader),
                RHIPipelineShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, fragShader),
            };

            std::vector<VkFormat> colorAttachmentFormats =
            {
                RTFormats::hdrSceneColor(),
                RTFormats::gbufferA(),
                RTFormats::gbufferB(),
                RTFormats::gbufferS(),
                RTFormats::gbufferV(),
            };

            std::vector<VkPipelineColorBlendAttachmentState> attachmentBlends =
            {
                RHIColorBlendAttachmentOpauqeState(),
                RHIColorBlendAttachmentOpauqeState(),
                RHIColorBlendAttachmentOpauqeState(),
                RHIColorBlendAttachmentOpauqeState(),
                RHIColorBlendAttachmentOpauqeState(),
            };

            const VkPipelineRenderingCreateInfo pipelineRenderingCreateInfo
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
                .colorAttachmentCount = (uint32_t)colorAttachmentFormats.size(),
                .pColorAttachmentFormats = colorAttachmentFormats.data(),
                .depthAttachmentFormat = RTFormats::depth(),
            };
            VkPipelineColorBlendStateCreateInfo colorBlending
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                .logicOpEnable = VK_FALSE,
                .logicOp = VK_LOGIC_OP_COPY,
                .attachmentCount = uint32_t(attachmentBlends.size()),
                .pAttachments = attachmentBlends.data(),
            };

            auto defaultViewport = RHIDefaultViewportState();
            const auto& deafultDynamicState = RHIDefaultDynamicStateCreateInfo();
            auto vertexInputState = RHIVertexInputStateCreateInfo();
            auto assemblyCreateInfo = RHIInputAssemblyCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
            auto rasterState = RHIRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL);
            rasterState.cullMode = VK_CULL_MODE_FRONT_BIT;
            auto multiSampleState = RHIMultisamplingStateCreateInfo();
            auto depthStencilState = RHIDepthStencilCreateInfo(true, true, VK_COMPARE_OP_GREATER); // Reverse z.

            VkPipelineLayoutCreateInfo plci = RHIPipelineLayoutCreateInfo();
            plci.setLayoutCount = (uint32_t)setLayouts.size();
            plci.pSetLayouts = setLayouts.data();
            gbufferPipelineLayout = RHI::get()->createPipelineLayout(plci);
            VkGraphicsPipelineCreateInfo pipelineCreateInfo
            {
                .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                .pNext = &pipelineRenderingCreateInfo,
                .stageCount = uint32_t(shaderStages.size()),
                .pStages = shaderStages.data(),
                .pVertexInputState = &vertexInputState,
                .pInputAssemblyState = &assemblyCreateInfo,
                .pViewportState = &defaultViewport,
                .pRasterizationState = &rasterState,
                .pMultisampleState = &multiSampleState,
                .pDepthStencilState = &depthStencilState,
                .pColorBlendState = &colorBlending,
                .pDynamicState = &deafultDynamicState,
                .layout = gbufferPipelineLayout,
            };
            RHICheck(vkCreateGraphicsPipelines(RHI::Device, nullptr, 1, &pipelineCreateInfo, nullptr, &gbufferPipeline));
        }
    };

    
    void DeferredRenderer::renderStaticMeshGBuffer(
        VkCommandBuffer cmd, 
        Renderer* renderer, 
        SceneTextures* inTextures,
        RenderSceneData* scene, 
        BufferParamRefPointer& viewData, 
        BufferParamRefPointer& frameData)
    {
        uint32_t staticMeshCount = (uint32_t)scene->getCollectStaticMeshes().size();

        auto& hdrSceneColor = inTextures->getHdrSceneColor()->getImage();
        auto& gbufferA = inTextures->getGbufferA()->getImage();
        auto& gbufferB = inTextures->getGbufferB()->getImage();
        auto& gbufferS = inTextures->getGbufferS()->getImage();
        auto& gbufferV = inTextures->getGbufferV()->getImage();
        auto& sceneDepthZ = inTextures->getDepth()->getImage();

        auto rtsLayout2Attachment = [&]()
        {
            hdrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
            gbufferA.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
            gbufferB.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
            gbufferS.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
            gbufferV.transitionLayout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT));
            sceneDepthZ.transitionLayout(cmd, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));
        };

        std::vector<VkRenderingAttachmentInfo> colorAttachments =
        {
            // Hdr scene color.
            RHIRenderingAttachmentInfo(
                inTextures->getHdrSceneColor()->getImage().getView(buildBasicImageSubresource()),
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, 
                VkClearValue{.color = {0.0f, 0.0f, 0.0f, 0.0f}}),

            // Gbuffer A
            RHIRenderingAttachmentInfo(
                inTextures->getGbufferA()->getImage().getView(buildBasicImageSubresource()),
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE,
                VkClearValue{.color = {0.0f, 0.0f, 0.0f, 0.0f}}),

            // Gbuffer B
            RHIRenderingAttachmentInfo(
                inTextures->getGbufferB()->getImage().getView(buildBasicImageSubresource()),
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE,
                VkClearValue{.color = {0.0f, 0.0f, 0.0f, 0.0f}}),

            // Gbuffer S
            RHIRenderingAttachmentInfo(
                inTextures->getGbufferS()->getImage().getView(buildBasicImageSubresource()),
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE,
                VkClearValue{.color = {0.0f, 0.0f, 0.0f, 0.0f}}),

            // Gbuffer V
            RHIRenderingAttachmentInfo(
                inTextures->getGbufferV()->getImage().getView(buildBasicImageSubresource()),
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE,
                VkClearValue{.color = {0.0f, 0.0f, 0.0f, 0.0f}}),
        };

        VkRenderingAttachmentInfo depthAttachment = RHIRenderingAttachmentInfo(
            inTextures->getDepth()->getImage().getView(RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT)),
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            VK_ATTACHMENT_LOAD_OP_CLEAR,
            VK_ATTACHMENT_STORE_OP_STORE,
            VkClearValue{ .depthStencil = {0.0f, 1} }
        );

        uint32_t renderWidth = inTextures->getHdrSceneColor()->getImage().getExtent().width;
        uint32_t renderHeight = inTextures->getHdrSceneColor()->getImage().getExtent().height;

        const VkRenderingInfo renderInfo
        {
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR,
            .renderArea = VkRect2D{.offset {0,0}, .extent {renderWidth, renderHeight}},
            .layerCount = 1,
            .colorAttachmentCount = uint32_t(colorAttachments.size()),
            .pColorAttachments = colorAttachments.data(),
            .pDepthAttachment = &depthAttachment,
        };

        VkRect2D scissor{ .offset{ 0,0 }, .extent {renderWidth, renderHeight} };
        VkViewport viewport
        {
            .x = 0.0f, .y = (float)m_renderHeight,
            .width = (float)renderWidth, .height = -(float)renderHeight,
            .minDepth = 0.0f, .maxDepth = 1.0f,
        };

       

        if (staticMeshCount <= 0)
        {
            rtsLayout2Attachment();
            
            RHI::ScopePerframeMarker staticMeshGBufferMarker(cmd, "StaticMeshGBuffer", { 1.0f, 0.0f, 0.0f, 1.0f });
            vkCmdBeginRendering(cmd, &renderInfo);
            {
                vkCmdSetScissor(cmd, 0, 1, &scissor);
                vkCmdSetViewport(cmd, 0, 1, &viewport);
                vkCmdSetDepthBias(cmd, 0, 0, 0);
            }
            vkCmdEndRendering(cmd);
            // Pre-return if no static mesh can use.
            return;
        }

        auto indirectDrawCommandBuffer = getBuffers()->getIndirectStorage("StaticMeshIndirectCommand", sizeof(GPUDrawIndirectCommand) * staticMeshCount);
        auto indirectDrawCountBuffer = getBuffers()->getIndirectStorage("StaticMeshIndirectCount", sizeof(GPUDrawIndirectCount));

        auto* pass = getPasses()->getPass<StaticMeshPass>();

        // Culling.
        {
            RHI::ScopePerframeMarker staticMeshGBufferCullingMarker(cmd, "StaticMeshGBufferCulling", { 1.0f, 0.0f, 0.0f, 1.0f });

            vkCmdFillBuffer(cmd, *indirectDrawCountBuffer->buffer.getBuffer(), 0, indirectDrawCountBuffer->buffer.getBuffer()->getSize(), 0u);
            vkCmdFillBuffer(cmd, *indirectDrawCommandBuffer->buffer.getBuffer(), 0, indirectDrawCommandBuffer->buffer.getBuffer()->getSize(), 0u);
            std::array<VkBufferMemoryBarrier2, 2> fillBarriers
            {
                RHIBufferBarrier(indirectDrawCommandBuffer->buffer.getBuffer()->getVkBuffer(),
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT),

                RHIBufferBarrier(indirectDrawCountBuffer->buffer.getBuffer()->getVkBuffer(),
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT),
            };
            RHIPipelineBarrier(cmd, 0, (uint32_t)fillBarriers.size(), fillBarriers.data(), 0, nullptr);


            GPUCullingPushConstants gpuPushConstant =
            {
                .cullCount = staticMeshCount,
            };

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->cullingPipeline);
            vkCmdPushConstants(cmd, pass->cullingPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(GPUCullingPushConstants), &gpuPushConstant);

            std::vector<VkDescriptorSet> compPassSets =
            {
                  scene->getStaticMeshesObjectsPtr()->buffer.getSet() // objectDatas
                , indirectDrawCommandBuffer->buffer.getSet()          // indirectCommands
                , indirectDrawCountBuffer->buffer.getSet()            // drawCount
                , viewData->buffer.getSet()                           // viewData
            };

            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                pass->cullingPipelineLayout, 0,
                (uint32_t)compPassSets.size(), compPassSets.data(),
                0, nullptr
            );
            vkCmdDispatch(cmd, getGroupCount(staticMeshCount, 64), 1, 1);

            m_gpuTimer.getTimeStamp(cmd, "StaticMesh Culling");

            // End buffer barrier.
            std::array<VkBufferMemoryBarrier2, 2> endBufferBarriers
            {
                RHIBufferBarrier(indirectDrawCommandBuffer->buffer.getBuffer()->getVkBuffer(),
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
                    VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT),

                RHIBufferBarrier(indirectDrawCountBuffer->buffer.getBuffer()->getVkBuffer(),
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
                    VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT),
            };
            RHIPipelineBarrier(cmd, 0, (uint32_t)endBufferBarriers.size(), endBufferBarriers.data(), 0, nullptr);
        }

        rtsLayout2Attachment();

        {
            RHI::ScopePerframeMarker staticMeshGBufferMarker(cmd, "StaticMeshGBuffer", { 1.0f, 0.0f, 0.0f, 1.0f });
            vkCmdBeginRendering(cmd, &renderInfo);
            {
                vkCmdSetScissor(cmd, 0, 1, &scissor);
                vkCmdSetViewport(cmd, 0, 1, &viewport);
                vkCmdSetDepthBias(cmd, 0, 0, 0);

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pass->gbufferPipeline);

                std::vector<VkDescriptorSet> meshPassSets =
                {
                      viewData->buffer.getSet()  // viewData
                    , frameData->buffer.getSet() // frameData
                    , MeshManager::get()->getBindlessVertexBuffers()->getSet() // verticesArray
                    , MeshManager::get()->getBindlessIndexBuffers()->getSet() // indicesArray
                    , Bindless::Texture->getSet()
                    , Bindless::Sampler->getSet()
                    , scene->getStaticMeshesObjectsPtr()->buffer.getSet() // objectDatas
                    , indirectDrawCommandBuffer->buffer.getSet() // indirectCommands
                };

                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pass->gbufferPipelineLayout,
                    0, (uint32_t)meshPassSets.size(), meshPassSets.data(), 0, nullptr);

                vkCmdDrawIndirectCount(cmd,
                    indirectDrawCommandBuffer->buffer.getBuffer()->getVkBuffer(), 0,
                    indirectDrawCountBuffer->buffer.getBuffer()->getVkBuffer(),
                    0,
                    staticMeshCount,
                    sizeof(GPUDrawIndirectCommand)
                );

                m_gpuTimer.getTimeStamp(cmd, "StaticMesh Rendering");
            }
            vkCmdEndRendering(cmd);
        }
    }
}