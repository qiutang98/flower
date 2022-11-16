#include "Pch.h"
#include "../DeferredRenderer.h"
#include "../../Renderer.h"
#include "../../RenderSceneData.h"
#include "../../SceneTextures.h"
#include "../../RenderSettingContext.h"

namespace Flower
{
    const uint32_t kHistogramBin = 128;
    const uint32_t kHistogramThreadDim = 16;

    struct AdaptiveExposurePush
    {
        float scale;
        float offset;
        float lowPercent;
        float highPercent;
        float minBrightness;
        float maxBrightness;
        float speedDown;
        float speedUp;
        float exposureCompensation;
        float deltaTime;
    };

    class AdaptiveExposurePass : public PassInterface
    {
    public:
        VkPipeline histogramPipeline = VK_NULL_HANDLE;
        VkPipeline averagePipeline = VK_NULL_HANDLE;

        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;

    public:
        virtual void init() override
        {
            CHECK(setLayout == VK_NULL_HANDLE);
            CHECK(pipelineLayout == VK_NULL_HANDLE);

            // Config code.
            RHI::get()->descriptorFactoryBegin()
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0) // in
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1) // out
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 2) // out
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 3) // in
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 4) // in
                .buildNoInfoPush(setLayout);

            std::vector<VkDescriptorSetLayout> setLayouts =
            {
                setLayout, // Owner setlayout.
                RHI::SamplerManager->getCommonDescriptorSetLayout()
                , GetLayoutStatic(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)  // viewData
                , GetLayoutStatic(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)  // frameData
            };

            VkPushConstantRange pushRange{ .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(AdaptiveExposurePush) };

            VkPipelineLayoutCreateInfo plci = RHIPipelineLayoutCreateInfo();
            plci.pushConstantRangeCount = 1;
            plci.pPushConstantRanges = &pushRange;
            plci.setLayoutCount = (uint32_t)setLayouts.size();
            plci.pSetLayouts = setLayouts.data();
            pipelineLayout = RHI::get()->createPipelineLayout(plci);

            VkPipelineShaderStageCreateInfo shaderStageCI{};
            shaderStageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            shaderStageCI.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            shaderStageCI.pName = "main";

            VkComputePipelineCreateInfo computePipelineCreateInfo{};
            computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            computePipelineCreateInfo.layout = pipelineLayout;
            computePipelineCreateInfo.flags = 0;


            {
                CHECK(histogramPipeline == VK_NULL_HANDLE);
                auto shaderModule = RHI::ShaderManager->getShader("AdaptiveExposureHistogramLumiance.comp.spv", true);

                shaderStageCI.module = shaderModule;
                computePipelineCreateInfo.stage = shaderStageCI;

                RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &histogramPipeline));
            }

            {
                CHECK(averagePipeline == VK_NULL_HANDLE);
                auto shaderModule = RHI::ShaderManager->getShader("AdaptiveExposureAverageLumiance.comp.spv", true);

                shaderStageCI.module = shaderModule;
                computePipelineCreateInfo.stage = shaderStageCI;

                RHICheck(vkCreateComputePipelines(RHI::Device, nullptr, 1, &computePipelineCreateInfo, nullptr, &averagePipeline));
            }
        }

        virtual void release() override
        {
            
            RHISafeRelease(pipelineLayout);
            setLayout = VK_NULL_HANDLE;


            RHISafeRelease(histogramPipeline);
            RHISafeRelease(averagePipeline);
        }
    };

    void DeferredRenderer::adaptiveExposure(VkCommandBuffer cmd, Renderer* renderer, SceneTextures* inTextures, RenderSceneData* scene, BufferParamRefPointer& viewData, BufferParamRefPointer& frameData, const RuntimeModuleTickData& tickData)
    {
        if (!m_averageLum)
        {
            m_averageLum = m_rtPool->createPoolImage(
                "AverageLum",
                1,
                1,
                VK_FORMAT_R16_SFLOAT,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);

            m_averageLum->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        }
        auto averageLumCurrent = m_rtPool->createPoolImage(
            "AverageLumCurrent",
            1,
            1,
            VK_FORMAT_R16_SFLOAT,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);


        auto histogram = m_rtPool->createPoolImage(
            "Histogram",
            kHistogramBin,
            1,
            VK_FORMAT_R32_UINT,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

        
        histogram->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());


        auto& hdrSceneColor = inTextures->getHdrSceneColorUpscale()->getImage();
        hdrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

        auto* pass = getPasses()->getPass<AdaptiveExposurePass>();

        auto inHdrColorInfo = RHIDescriptorImageInfoSample(hdrSceneColor.getView(buildBasicImageSubresource()));
        auto outImageInfo = RHIDescriptorImageInfoStorage(averageLumCurrent->getImage().getView(buildBasicImageSubresource()));
        auto histogramImageInfo = RHIDescriptorImageInfoStorage(histogram->getImage().getView(buildBasicImageSubresource()));
        auto histogramSampleInfo = RHIDescriptorImageInfoSample(histogram->getImage().getView(buildBasicImageSubresource()));
        auto averageLumHistoryInfo = RHIDescriptorImageInfoSample(m_averageLum->getImage().getView(buildBasicImageSubresource()));

        std::vector<VkWriteDescriptorSet> writes
        {
            RHIPushWriteDescriptorSetImage(0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &inHdrColorInfo),
            RHIPushWriteDescriptorSetImage(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outImageInfo),
            RHIPushWriteDescriptorSetImage(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &histogramImageInfo),
            RHIPushWriteDescriptorSetImage(3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &histogramSampleInfo),
            RHIPushWriteDescriptorSetImage(4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &averageLumHistoryInfo),
        }; 

        RHI::PushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->pipelineLayout, 0, uint32_t(writes.size()), writes.data());

        std::vector<VkDescriptorSet> passSets =
        {
            RHI::SamplerManager->getCommonDescriptorSet(), // samplers.
              viewData->buffer.getSet()
            , frameData->buffer.getSet()
        };

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->pipelineLayout,
            1, (uint32_t)passSets.size(), passSets.data(), 0, nullptr);

        const float maxEv = 9.0f;
        const float minEv = -9.0f;

        const float diff = maxEv - minEv;
        const float scale = 1.0f / diff;
        const float offset = -minEv * scale;

        RenderSettingManager::get()->AUTOEXPOSURE_minBrightness = glm::clamp(RenderSettingManager::get()->AUTOEXPOSURE_minBrightness, minEv, maxEv);
        RenderSettingManager::get()->AUTOEXPOSURE_maxBrightness = glm::clamp(RenderSettingManager::get()->AUTOEXPOSURE_maxBrightness, minEv, maxEv);
        

        RenderSettingManager::get()->AUTOEXPOSURE_lowPercent = glm::clamp(RenderSettingManager::get()->AUTOEXPOSURE_lowPercent, 0.01f, 0.99f);
        RenderSettingManager::get()->AUTOEXPOSURE_highPercent = glm::clamp(RenderSettingManager::get()->AUTOEXPOSURE_highPercent, 0.01f, 0.99f);

        AdaptiveExposurePush pushConst
        {
            .scale = scale,
            .offset = offset,

            .lowPercent = RenderSettingManager::get()->AUTOEXPOSURE_lowPercent,
            .highPercent = RenderSettingManager::get()->AUTOEXPOSURE_highPercent,

            .minBrightness = glm::exp2(RenderSettingManager::get()->AUTOEXPOSURE_minBrightness),
            .maxBrightness = glm::exp2(RenderSettingManager::get()->AUTOEXPOSURE_maxBrightness),

            .speedDown = RenderSettingManager::get()->AUTOEXPOSURE_speedDown,
            .speedUp = RenderSettingManager::get()->AUTOEXPOSURE_speedUp,
            .exposureCompensation = RenderSettingManager::get()->AUTOEXPOSURE_exposureCompensation,
            .deltaTime = tickData.smoothDeltaTime,
        };
        vkCmdPushConstants(cmd, pass->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConst), &pushConst);

        

        {
            RHI::ScopePerframeMarker marker(cmd, "AdaptiveExposure Histogram", { 1.0f, 1.0f, 0.0f, 1.0f });

            VkClearColorValue zeroClear = {
                .uint32 = {0,0,0,0}
            };

            auto rangeClear = buildBasicImageSubresource();

            vkCmdClearColorImage(cmd, histogram->getImage().getImage(), VK_IMAGE_LAYOUT_GENERAL, &zeroClear, 1, &rangeClear);
            VkImageMemoryBarrier2 clearBarrier =
            {
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT,
                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .image = histogram->getImage().getImage(),
                .subresourceRange = rangeClear
            };

            RHIPipelineBarrier(cmd, 0, 0, nullptr, 1, &clearBarrier);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->histogramPipeline);

            vkCmdDispatch(cmd, 
                getGroupCount(hdrSceneColor.getExtent().width / 3 + 1, kHistogramThreadDim), 
                getGroupCount(hdrSceneColor.getExtent().height / 3 + 1, kHistogramThreadDim), 1);

            histogram->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());


        }

        averageLumCurrent->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
        {
            RHI::ScopePerframeMarker marker(cmd, "AdaptiveExposure Average", { 1.0f, 1.0f, 0.0f, 1.0f });

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass->averagePipeline);

            vkCmdDispatch(cmd, 1, 1, 1);
        }
        averageLumCurrent->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

        m_averageLum = averageLumCurrent;

        m_gpuTimer.getTimeStamp(cmd, "AdaptiveExposure");
    }
}