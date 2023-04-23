#include "../renderer_interface.h"
#include "../render_scene.h"
#include "../renderer.h"
#include "../scene_textures.h"

namespace engine
{
    struct GpuGtaoPush
    {
        uint32_t sliceNum;
        uint32_t stepNum;
        float radius;
        float thickness;
        float power;
        float intensity;
    };

    class GtaoPass : public PassInterface
    {
    public:
        VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
        std::unique_ptr<ComputePipeResources> evaluate;
        std::unique_ptr<ComputePipeResources> prefilter;
        std::unique_ptr<ComputePipeResources> temporal;

    public:
        virtual void onInit() override
        {
            // Config code.
            getContext()->descriptorFactoryBegin()
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0) // inHiz
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1) // inDepth
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 2) // inGbufferA
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 3) // inGbufferB
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 4) // inGbufferS
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 5) // GTAOImage
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 6) // inGTAO
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 7) // GTAOFilterImageX
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 8) // inGTAOFilterImageX
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 9) // GTAO temp filter.
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 10) // in GTAO temp filter.
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 11) // GTAO history
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 12) // in GTAO history
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 13) // in Velocity
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 14) // inPrevDepth
                .bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 15) // frameBuffer
                .buildNoInfoPush(setLayout);

            std::vector<VkDescriptorSetLayout> setLayouts = { setLayout, getContext()->getSamplerCache().getCommonDescriptorSetLayout() };
            uint32_t pushSize = (uint32_t)sizeof(GpuGtaoPush);

            evaluate = std::make_unique<ComputePipeResources>("shader/gtao_evaluate.comp.spv", pushSize, setLayouts);
            prefilter = std::make_unique<ComputePipeResources>("shader/gtao_prefilter.comp.spv", pushSize, setLayouts);
            temporal = std::make_unique<ComputePipeResources>("shader/gtao_temporal.comp.spv", pushSize, setLayouts);
        }

        virtual void release() override
        {
            evaluate.reset();
            prefilter.reset();
            temporal.reset();
        }
    };

    PoolImageSharedRef RendererInterface::renderGTAO(
        VkCommandBuffer cmd,
        GBufferTextures* inGBuffers,
        RenderScene* scene,
        BufferParameterHandle perFrameGPU,
        PoolImageSharedRef inHiz)
    {
        auto* pass = getContext()->getPasses().get<GtaoPass>();
        auto* rtPool = &m_context->getRenderTargetPools();

        auto& gbufferA = inGBuffers->gbufferA->getImage();
        auto& gbufferB = inGBuffers->gbufferB->getImage();
        auto& gbufferS = inGBuffers->gbufferS->getImage();
        auto& gbufferV = inGBuffers->gbufferV->getImage();
        auto& sceneDepthZ = inGBuffers->depthTexture->getImage();

        gbufferA.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        gbufferB.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        gbufferS.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        gbufferV.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        sceneDepthZ.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));
        inHiz->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

        auto imageGTAOEvaluate = rtPool->createPoolImage(
            "GTAOEvaluate",
            sceneDepthZ.getExtent().width,
            sceneDepthZ.getExtent().height,
            VK_FORMAT_R8_UNORM,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

        auto imageGTAOFilter = rtPool->createPoolImage(
            "GTAOFilter",
            imageGTAOEvaluate->getImage().getExtent().width,
            imageGTAOEvaluate->getImage().getExtent().height,
            VK_FORMAT_R8_UNORM,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

        auto imageGTAOTempFilter = rtPool->createPoolImage(
            "GTAOTempFilter",
            imageGTAOEvaluate->getImage().getExtent().width,
            imageGTAOEvaluate->getImage().getExtent().height,
            VK_FORMAT_R8_UNORM,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

        bool bShouldCreateNewGTAO;
        if (m_gtaoHistory)
        {
            bShouldCreateNewGTAO =
                m_gtaoHistory->getImage().getExtent().width  != imageGTAOEvaluate->getImage().getExtent().width ||
                m_gtaoHistory->getImage().getExtent().height != imageGTAOEvaluate->getImage().getExtent().height;
        }
        else
        {
            bShouldCreateNewGTAO = true;
        }

        if (bShouldCreateNewGTAO)
        {
            m_gtaoHistory = rtPool->createPoolImage(
                "GTAOHistory",
                imageGTAOEvaluate->getImage().getExtent().width,
                imageGTAOEvaluate->getImage().getExtent().height,
                VK_FORMAT_R8_UNORM,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
            m_gtaoHistory->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        }
        CHECK(m_gtaoHistory);

        PushSetBuilder setBuilder(cmd);
        setBuilder
            .addSRV(inHiz)
            .addSRV(sceneDepthZ, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT))
            .addSRV(gbufferA)
            .addSRV(gbufferB)
            .addSRV(gbufferS)
            .addUAV(imageGTAOEvaluate)
            .addSRV(imageGTAOEvaluate)
            .addUAV(imageGTAOFilter)
            .addSRV(imageGTAOFilter)
            .addUAV(imageGTAOTempFilter)
            .addSRV(imageGTAOTempFilter)
            .addUAV(m_gtaoHistory)
            .addSRV(m_gtaoHistory)
            .addSRV(gbufferV)
            .addSRV(m_prevDepth == nullptr ? inGBuffers->depthTexture : m_prevDepth, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT))
            .addBuffer(perFrameGPU)
            .push(pass->evaluate.get()); // All gtao use same pipeline layout so just push once.

        std::vector<VkDescriptorSet> additionalSets =
        {
            m_context->getSamplerCache().getCommonDescriptorSet()
        };
        pass->evaluate->bindSet(cmd, additionalSets, 1);

        const auto& postProcessVolumeSetting = scene->getPostprocessVolumeSetting();

        GpuGtaoPush pushConst
        {
            .sliceNum = (uint32_t)postProcessVolumeSetting.gtaoSliceNum,
            .stepNum = (uint32_t)postProcessVolumeSetting.gtaoStepNum,
            .radius = postProcessVolumeSetting.gtaoRadius,
            .thickness = postProcessVolumeSetting.gtaoThickness,
            .power = postProcessVolumeSetting.gtaoPower,
            .intensity = postProcessVolumeSetting.gtaoIntensity,
        };
        pass->evaluate->pushConst(cmd, &pushConst);

        {
            ScopePerframeMarker marker(cmd, "Compute GTAO", { 1.0f, 1.0f, 0.0f, 1.0f });

            imageGTAOEvaluate->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            pass->evaluate->bind(cmd);

            vkCmdDispatch(cmd, getGroupCount(imageGTAOEvaluate->getImage().getExtent().width, 8), getGroupCount(imageGTAOEvaluate->getImage().getExtent().height, 8), 1);

            imageGTAOEvaluate->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        }

        {
            ScopePerframeMarker marker(cmd, "Filter GTAO", { 1.0f, 1.0f, 0.0f, 1.0f });
            imageGTAOFilter->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

            pass->prefilter->bind(cmd);

            vkCmdDispatch(cmd, getGroupCount(imageGTAOFilter->getImage().getExtent().width, 16), getGroupCount(imageGTAOFilter->getImage().getExtent().height, 16), 1);

            imageGTAOFilter->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        }

        {
            ScopePerframeMarker marker(cmd, "TempFilter GTAO", { 1.0f, 1.0f, 0.0f, 1.0f });
            imageGTAOTempFilter->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

            pass->temporal->bind(cmd);

            vkCmdDispatch(cmd, getGroupCount(imageGTAOTempFilter->getImage().getExtent().width, 8), getGroupCount(imageGTAOTempFilter->getImage().getExtent().height, 8), 1);

            imageGTAOTempFilter->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        }
        m_gpuTimer.getTimeStamp(cmd, "GTAO");

        // Update GTAO history.
        m_gtaoHistory = imageGTAOTempFilter;
        return imageGTAOTempFilter;
    }
}