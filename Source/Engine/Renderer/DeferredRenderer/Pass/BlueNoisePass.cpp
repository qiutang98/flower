#include "Pch.h"
#include "../DeferredRenderer.h"
#include "../../Renderer.h"
#include "../../RenderSceneData.h"
#include "../../SceneTextures.h"
#include "../../RenderSettingContext.h"


namespace Flower
{
    constexpr uint32_t GBlueNoiseDim = 128u;

    struct BlueNoisePushConst
    {
        uint32_t sampleIndex;
    };

    class BlueNoisePass : public PassInterface
    {
    public:
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;

    public:
        virtual void init() override
        {
            CHECK(pipeline == VK_NULL_HANDLE);
            CHECK(pipelineLayout == VK_NULL_HANDLE);
            CHECK(setLayout == VK_NULL_HANDLE);

            // Config code.
            RHI::get()->descriptorFactoryBegin()
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0)
                .buildNoInfoPush(setLayout);

            std::vector<VkDescriptorSetLayout> setLayouts =
            {
                  setLayout // Owner setlayout.
                , StaticTexturesManager::get()->globalBlueNoise.spp_1_buffer.setLayouts // All blue noise set layout is same.
                , GetLayoutStatic(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) // viewData
                , GetLayoutStatic(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) // frameData
                , RHI::SamplerManager->getCommonDescriptorSetLayout()
            };
            auto shaderModule = RHI::ShaderManager->getShader("BlueNoiseGenerate.comp.spv", true);

            // Vulkan buid functions.
            VkPipelineLayoutCreateInfo plci = RHIPipelineLayoutCreateInfo();
            VkPushConstantRange pushConstant{};
            pushConstant.offset = 0;
            pushConstant.size = sizeof(BlueNoisePushConst);
            pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            plci.pPushConstantRanges = &pushConstant;
            plci.pushConstantRangeCount = 1;

            plci.setLayoutCount = (uint32_t)setLayouts.size();
            plci.pSetLayouts = setLayouts.data();
            pipelineLayout = RHI::get()->createPipelineLayout(plci);
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
            RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &pipeline));
        }

        virtual void release() override
        {
            RHISafeRelease(pipeline);
            RHISafeRelease(pipelineLayout);
            setLayout = VK_NULL_HANDLE;
        }
    };



    BlueNoiseMisc DeferredRenderer::renderBlueNoiseMisc(
        VkCommandBuffer cmd,
        Renderer* renderer,
        SceneTextures* inTextures,
        RenderSceneData* scene,
        BufferParamRefPointer& viewData,
        BufferParamRefPointer& frameData,
        const RuntimeModuleTickData& tickData)
    {
        BlueNoiseMisc result{};

        const auto blueNoiseFormat = VK_FORMAT_R8G8_UNORM;
        const auto blueNoiseUsage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;

        result.spp_1_image = m_rtPool->createPoolImage("BlueNoise_1spp",GBlueNoiseDim, GBlueNoiseDim, blueNoiseFormat, blueNoiseUsage);
        result.spp_1_image->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

        auto* pass = getPasses()->getPass<BlueNoisePass>();
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->pipeline);

        std::vector<VkDescriptorSet> passSets =
        {
            viewData->buffer.getSet(),
            frameData->buffer.getSet(),
            RHI::SamplerManager->getCommonDescriptorSet(), // samplers.
        };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->pipelineLayout, 2, (uint32_t)passSets.size(), passSets.data(), 0, nullptr);

        RHI::ScopePerframeMarker marker(cmd, "BlueNoisePrepare", { 1.0f, 1.0f, 0.0f, 1.0f });

        

        // need 1 spp blue noise rotate by golden radio for ssr jitter.
        {
            BlueNoisePushConst push{ .sampleIndex = 0};
            vkCmdPushConstants(cmd, pass->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

            auto set = StaticTexturesManager::get()->globalBlueNoise.spp_1_buffer.set;
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->pipelineLayout, 1, 1, &set, 0, nullptr);

            VkDescriptorImageInfo outImageInfo = RHIDescriptorImageInfoStorage(result.spp_1_image->getImage().getView(buildBasicImageSubresource()));
            auto pushSet = RHIPushWriteDescriptorSetImage(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outImageInfo);

            RHI::PushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->pipelineLayout, 0, 1, &pushSet);

            vkCmdDispatch(cmd, getGroupCount(GBlueNoiseDim, 8), getGroupCount(GBlueNoiseDim, 8), 1);

        }

        result.spp_1_image->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

        m_gpuTimer.getTimeStamp(cmd, "BlueNoisePrepare");
        return result;
    }
}