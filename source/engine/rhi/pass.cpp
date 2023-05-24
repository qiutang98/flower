#include "pass.h"
#include "rhi.h"

namespace engine
{
	PassCollector::PassCollector(VulkanContext* context)
		: m_context(context)
	{

	}

	void PassCollector::updateAllPasses()
	{
		m_context->waitDeviceIdle();

		for (auto& pair : m_passMap)
		{
			pair.second->release();
			pair.second->init(m_context);
		}
	}

	PassCollector::~PassCollector()
	{
		m_context->waitDeviceIdle();

		for (auto& pair : m_passMap)
		{
			pair.second->release();
		}
	}

	ComputePipeResources::ComputePipeResources(const std::string& shaderPath, uint32_t pushConstSize, const std::vector<VkDescriptorSetLayout>& inSetLayout)
	{
        const std::vector<VkDescriptorSetLayout>& setLayouts = inSetLayout;
		auto shaderModule = getContext()->getShaderCache().getShader(shaderPath, true);
		

		VkPipelineLayoutCreateInfo plci = RHIPipelineLayoutCreateInfo();
		VkPushConstantRange pushRange{ .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = pushConstSize };

		if (pushConstSize > 0)
		{
			plci.pushConstantRangeCount = 1;
			plci.pPushConstantRanges = &pushRange;
		}

		plci.setLayoutCount = (uint32_t)setLayouts.size();
		plci.pSetLayouts = setLayouts.data();
		pipelineLayout = getContext()->createPipelineLayout(plci);
		VkPipelineShaderStageCreateInfo shaderStageCI{};
		shaderStageCI.module = shaderModule;
		shaderStageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		shaderStageCI.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderStageCI.pName = "main";
		VkComputePipelineCreateInfo computePipelineCreateInfo{};
		computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		computePipelineCreateInfo.layout = pipelineLayout;
		computePipelineCreateInfo.flags = 0;
		computePipelineCreateInfo.stage = shaderStageCI;
		RHICheck(vkCreateComputePipelines(getContext()->getDevice(), nullptr, 1, &computePipelineCreateInfo, nullptr, &pipeline));
	}

    GraphicPipeResources::GraphicPipeResources(
        const std::string& vertShaderPath,
        const std::string& fragShaderPath,
        const std::vector<VkDescriptorSetLayout>& inSetLayout,
        uint32_t pushConstSize,
        std::vector<VkFormat>&& inColorAttachmentFormats,
        std::vector<VkPipelineColorBlendAttachmentState>&& inBlendState,
        VkFormat depthFormat,
        VkCullModeFlags cullMode,
        VkCompareOp zTestComp,
        bool bEnableDepthClamp,
        bool bEnableDepthBias,
        const std::vector<VkVertexInputAttributeDescription>& inputAttributes,
        uint32_t vertexStrip,
        VkPolygonMode polygonMode,
        bool bZWrite)
    {
        auto vertShader = getContext()->getShaderCache().getShader(vertShaderPath, true);
        auto fragShader = getContext()->getShaderCache().getShader(fragShaderPath, true);

        std::vector<VkPipelineShaderStageCreateInfo> shaderStages =
        {
            RHIPipelineShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, vertShader),
            RHIPipelineShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, fragShader),
        };

        std::vector<VkDescriptorSetLayout> setLayouts = inSetLayout;
        std::vector<VkFormat> colorAttachmentFormats = inColorAttachmentFormats;
        std::vector<VkPipelineColorBlendAttachmentState> attachmentBlends = inBlendState;
        VkPipelineColorBlendStateCreateInfo colorBlending
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .logicOpEnable = VK_FALSE,
            .logicOp = VK_LOGIC_OP_COPY,
            .attachmentCount = uint32_t(attachmentBlends.size()),
            .pAttachments = attachmentBlends.size() > 0 ? attachmentBlends.data() : nullptr,
        };

        const VkPipelineRenderingCreateInfo pipelineRenderingCreateInfo
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
            .colorAttachmentCount = (uint32_t)colorAttachmentFormats.size(),
            .pColorAttachmentFormats = colorAttachmentFormats.size() > 0 ? colorAttachmentFormats.data() : nullptr,
            .depthAttachmentFormat = depthFormat,
        };

        auto defaultViewport = RHIDefaultViewportState();
        const auto& deafultDynamicState = RHIDefaultDynamicStateCreateInfo();
        auto vertexInputState = RHIVertexInputStateCreateInfo();

        VkVertexInputBindingDescription inputBindingDes { };
        if (!inputAttributes.empty() && vertexStrip > 0)
        {
            inputBindingDes =
            {
                .binding = 0,
                .stride = vertexStrip,
                .inputRate = VK_VERTEX_INPUT_RATE_VERTEX
            };

            vertexInputState.vertexAttributeDescriptionCount = (uint32_t)inputAttributes.size();
            vertexInputState.vertexBindingDescriptionCount = 1;
            vertexInputState.pVertexBindingDescriptions = &inputBindingDes;
            vertexInputState.pVertexAttributeDescriptions = inputAttributes.data();
        }

        auto assemblyCreateInfo = RHIInputAssemblyCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
        auto rasterState = RHIRasterizationStateCreateInfo(polygonMode);
        rasterState.cullMode = cullMode;
        rasterState.depthBiasEnable = bEnableDepthBias ? VK_TRUE : VK_FALSE;
        rasterState.depthClampEnable = bEnableDepthClamp ? VK_TRUE : VK_FALSE;
        auto multiSampleState = RHIMultisamplingStateCreateInfo();
        auto depthStencilState = RHIDepthStencilCreateInfo(true, bZWrite, zTestComp);

        VkPipelineLayoutCreateInfo plci = RHIPipelineLayoutCreateInfo();

        VkPushConstantRange pushRange{ .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, .offset = 0, .size = pushConstSize };
        if (pushConstSize > 0)
        {
            plci.pushConstantRangeCount = 1;
            plci.pPushConstantRanges = &pushRange;
        }

        plci.setLayoutCount = (uint32_t)setLayouts.size();
        plci.pSetLayouts = setLayouts.data();
        pipelineLayout = getContext()->createPipelineLayout(plci);
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
            .layout = pipelineLayout,
        };
        RHICheck(vkCreateGraphicsPipelines(getContext()->getDevice(), nullptr, 1, &pipelineCreateInfo, nullptr, &pipeline));
    }

    PipeResource::~PipeResource()
    {
        contextSafeRelease(pipeline);
        contextSafeRelease(pipelineLayout);
    }

    void PushSetBuilder::push(PipeResource* pipe)
    {
        std::vector<VkWriteDescriptorSet> writes(m_cacheBindingBuilder.size());
        std::vector<VkDescriptorImageInfo> images(m_cacheBindingBuilder.size());
        std::vector<VkWriteDescriptorSetAccelerationStructureKHR> ases(m_cacheBindingBuilder.size());

        for (uint32_t i = 0; i < m_cacheBindingBuilder.size(); i++)
        {
            auto& binding = m_cacheBindingBuilder[i];
            switch (m_cacheBindingBuilder[i].type)
            {
                case CacheBindingBuilder::EType::buffer:
                {
                    writes[i] = RHIPushWriteDescriptorSetBuffer(i, binding.descriptorType, &binding.bufferInfo);
                }
                break;
                case CacheBindingBuilder::EType::as:
                {
                    CHECK(binding.asBuilder->isInit());

                    VkWriteDescriptorSetAccelerationStructureKHR descriptorAccelerationStructureInfo{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR };
                    descriptorAccelerationStructureInfo.accelerationStructureCount = 1;
                    descriptorAccelerationStructureInfo.pAccelerationStructures = &binding.asBuilder->getAccelerationStructure();
                    ases[i] = descriptorAccelerationStructureInfo;

                    VkWriteDescriptorSet accelerationStructureWrite{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
                    accelerationStructureWrite.pNext = &ases[i];
                    accelerationStructureWrite.dstBinding = i;
                    accelerationStructureWrite.descriptorCount = 1;
                    accelerationStructureWrite.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
                    writes[i] = accelerationStructureWrite;
                }
                break;
                case CacheBindingBuilder::EType::srv:
                {
                    images[i] = RHIDescriptorImageInfoSample(binding.image->getOrCreateView(binding.imageRange, binding.viewType));
                    writes[i] = RHIPushWriteDescriptorSetImage(i, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &images[i]);
                }
                break;
                case CacheBindingBuilder::EType::uav:
                {
                    images[i] = RHIDescriptorImageInfoStorage(binding.image->getOrCreateView(binding.imageRange, binding.viewType));
                    writes[i] = RHIPushWriteDescriptorSetImage(i, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &images[i]);
                }
                break;
                default:
                {
                    CHECK_ENTRY();
                }
                break;
            }
        }

        getContext()->pushDescriptorSet(m_cmd, pipe->getBindPoint(), pipe->pipelineLayout, 0, uint32_t(writes.size()), writes.data());
    }
}