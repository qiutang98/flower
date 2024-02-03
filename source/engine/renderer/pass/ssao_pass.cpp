#include "../deferred_renderer.h"
#include "../render_scene.h"
#include "../renderer.h"
#include "../scene_textures.h"
#include "../../scene/component/postprocess_component.h"


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
        float falloffEnd;
    };

    class SSAOPass : public PassInterface
    {
    public:
        std::unique_ptr<ComputePipeResources> intersect;

        std::unique_ptr<ComputePipeResources> gtao_evaluate;
        std::unique_ptr<ComputePipeResources> gtao_prefilter;
        std::unique_ptr<ComputePipeResources> gtao_temporal;

    public:
        virtual void onInit() override
        {
            {
                VkDescriptorSetLayout intersectLayout = VK_NULL_HANDLE;

                getContext()->descriptorFactoryBegin()
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 0) // 
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1) // 
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 2) // 
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 3) // 
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4) // 
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 5) // 
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 6) // 
                    .buildNoInfoPush(intersectLayout);

                std::vector<VkDescriptorSetLayout> intersectLayouts =
                {
                    intersectLayout,
                    getContext()->getSamplerCache().getCommonDescriptorSetLayout(),
                    getRenderer()->getBlueNoise().spp_1_buffer.setLayouts
                };

                intersect = std::make_unique<ComputePipeResources>("shader/ssao.glsl", 0, intersectLayouts);
            }

            {
                VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;

                // Config code.
                getContext()->descriptorFactoryBegin()
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 0) // inHiz
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1) // inDepth
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 2) // inGbufferA
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 3) // inGbufferB
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 4) // inGbufferS
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 5) // GTAOImage
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 6) // inGTAO
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 7) // GTAOFilterImageX
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 8) // inGTAOFilterImageX
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 9) // GTAO temp filter.
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 10) // in GTAO temp filter.
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 11) // GTAO history
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 12) // in GTAO history
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 13) // in Velocity
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 14) // inPrevDepth
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 15) // frameBuffer
                    .buildNoInfoPush(setLayout);

                std::vector<VkDescriptorSetLayout> setLayouts = { setLayout, getContext()->getSamplerCache().getCommonDescriptorSetLayout() };
                uint32_t pushSize = (uint32_t)sizeof(GpuGtaoPush);

                gtao_evaluate = std::make_unique<ComputePipeResources>("shader/gtao_evaluate.glsl", pushSize, setLayouts);
                gtao_prefilter = std::make_unique<ComputePipeResources>("shader/gtao_prefilter.glsl", pushSize, setLayouts);
                gtao_temporal = std::make_unique<ComputePipeResources>("shader/gtao_temporal.glsl", pushSize, setLayouts);
            }
        }

        virtual void release() override
        {
            intersect.reset();
            gtao_evaluate.reset();
            gtao_prefilter.reset();
            gtao_temporal.reset();
        }
    };


    PoolImageSharedRef DeferredRenderer::renderSSAO(
        VkCommandBuffer cmd,
        GBufferTextures* inGBuffers,
        RenderScene* scene,
        BufferParameterHandle perFrameGPU,
        PoolImageSharedRef inHiz)
    {
        auto* pass = getContext()->getPasses().get<SSAOPass>();
        auto* rtPool = &getContext()->getRenderTargetPools();

        bool bSuperResolution = (m_dimensionConfig.getRenderWidth() < m_dimensionConfig.getPostWidth());



        auto gbufferB = inGBuffers->gbufferB;
        auto historyColor = m_history.prevHdrBeforeAA ? m_history.prevHdrBeforeAA : inGBuffers->hdrSceneColor;

        auto& gbufferA = inGBuffers->gbufferA->getImage();
        auto& gbufferS = inGBuffers->gbufferS->getImage();
        auto& gbufferV = inGBuffers->gbufferV->getImage();
        auto& sceneDepthZ = inGBuffers->depthTexture->getImage();

        gbufferA.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        gbufferB->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        gbufferS.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        gbufferV.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        sceneDepthZ.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));
        inHiz->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

        
        GpuGtaoPush pushConst;
        bool bForceGTAO = false;
        if(scene->getPostprocessComponent() != nullptr)
        {
            pushConst =
            {
                .sliceNum = (uint32_t)scene->getPostprocessComponent()->getSetting().ssao_sliceCount,
                .stepNum = (uint32_t)scene->getPostprocessComponent()->getSetting().ssao_stepCount,
                .radius = scene->getPostprocessComponent()->getSetting().gtao_radius,
                .thickness = scene->getPostprocessComponent()->getSetting().gtao_thickness,
                .power = scene->getPostprocessComponent()->getSetting().ssao_power,
                .intensity = scene->getPostprocessComponent()->getSetting().ssao_intensity,
                .falloffEnd = scene->getPostprocessComponent()->getSetting().gtao_falloffEnd,
            };

            bForceGTAO = scene->getPostprocessComponent()->getSetting().ssao_bGTAO != 0;
        }
        else
        {
            pushConst.sliceNum = 2;
            pushConst.stepNum = 4;
            pushConst.radius = 0.5f;
            pushConst.thickness = 0.1f;
            pushConst.power = 1.0f;
            pushConst.intensity = 1.0f;
            pushConst.falloffEnd = 2.0f;
        }


        if (bSuperResolution || bForceGTAO)
        {
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
            if (m_history.gtaoHistory)
            {
                bShouldCreateNewGTAO =
                    m_history.gtaoHistory->getImage().getExtent().width != imageGTAOEvaluate->getImage().getExtent().width ||
                    m_history.gtaoHistory->getImage().getExtent().height != imageGTAOEvaluate->getImage().getExtent().height;
            }
            else
            {
                bShouldCreateNewGTAO = true;
            }

            if (bShouldCreateNewGTAO)
            {
                m_history.gtaoHistory = rtPool->createPoolImage(
                    "GTAOHistory",
                    imageGTAOEvaluate->getImage().getExtent().width,
                    imageGTAOEvaluate->getImage().getExtent().height,
                    VK_FORMAT_R8_UNORM,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
                m_history.gtaoHistory->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            }
            CHECK(m_history.gtaoHistory);

            PushSetBuilder setBuilder(cmd);
            setBuilder
                .addSRV(inHiz)
                .addSRV(sceneDepthZ, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT))
                .addSRV(gbufferA)
                .addSRV(inGBuffers->vertexNormal)
                .addSRV(gbufferS)
                .addUAV(imageGTAOEvaluate)
                .addSRV(imageGTAOEvaluate)
                .addUAV(imageGTAOFilter)
                .addSRV(imageGTAOFilter)
                .addUAV(imageGTAOTempFilter)
                .addSRV(imageGTAOTempFilter)
                .addUAV(m_history.gtaoHistory)
                .addSRV(m_history.gtaoHistory)
                .addSRV(gbufferV)
                .addSRV(m_history.prevDepth == nullptr ? inGBuffers->depthTexture : m_history.prevDepth, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT))
                .addBuffer(perFrameGPU)
                .push(pass->gtao_evaluate.get()); // All gtao use same pipeline layout so just push once.

            std::vector<VkDescriptorSet> additionalSets =
            {
                getContext()->getSamplerCache().getCommonDescriptorSet()
            };
            pass->gtao_evaluate->bindSet(cmd, additionalSets, 1);

            pass->gtao_evaluate->pushConst(cmd, &pushConst);

            {
                imageGTAOEvaluate->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                pass->gtao_evaluate->bind(cmd);

                vkCmdDispatch(cmd, getGroupCount(imageGTAOEvaluate->getImage().getExtent().width, 8), getGroupCount(imageGTAOEvaluate->getImage().getExtent().height, 8), 1);

                imageGTAOEvaluate->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            }

            {
                imageGTAOFilter->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

                pass->gtao_prefilter->bind(cmd);

                vkCmdDispatch(cmd, getGroupCount(imageGTAOFilter->getImage().getExtent().width, 16), getGroupCount(imageGTAOFilter->getImage().getExtent().height, 16), 1);

                imageGTAOFilter->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            }

            {
                imageGTAOTempFilter->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

                pass->gtao_temporal->bind(cmd);

                vkCmdDispatch(cmd, getGroupCount(imageGTAOTempFilter->getImage().getExtent().width, 8), getGroupCount(imageGTAOTempFilter->getImage().getExtent().height, 8), 1);

                imageGTAOTempFilter->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            }
            m_gpuTimer.getTimeStamp(cmd, "GTAO");

            // Update GTAO history.
            m_history.gtaoHistory = imageGTAOTempFilter;

            return imageGTAOTempFilter;
        }
        else
        {
            PoolImageSharedRef ssaoIntersectResultBentNormal = rtPool->createPoolImage(
                "SSAO Intersect bent normal",
                gbufferB->getImage().getExtent().width,
                gbufferB->getImage().getExtent().height,
                VK_FORMAT_R8_UNORM,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

            ScopePerframeMarker tonemapperMarker(cmd, "SSAO", { 1.0f, 1.0f, 0.0f, 1.0f }, &m_gpuTimer);

            gbufferB->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            historyColor->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            ssaoIntersectResultBentNormal->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

            pass->intersect->bind(cmd);
            PushSetBuilder(cmd)
                .addSRV(inHiz)
                .addSRV(inGBuffers->depthTexture, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT))
                .addSRV(gbufferS)
                .addSRV(gbufferB)
                .addUAV(ssaoIntersectResultBentNormal)
                .addSRV(historyColor)
                .addBuffer(perFrameGPU)
                .push(pass->intersect.get());

            pass->intersect->bindSet(cmd, std::vector<VkDescriptorSet>
            {
                getContext()->getSamplerCache().getCommonDescriptorSet(), 
                getRenderer()->getBlueNoise().spp_1_buffer.set
            }, 1);

            vkCmdDispatch(cmd,
                getGroupCount(ssaoIntersectResultBentNormal->getImage().getExtent().width, 8),
                getGroupCount(ssaoIntersectResultBentNormal->getImage().getExtent().height, 8), 1);
            ssaoIntersectResultBentNormal->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

            return ssaoIntersectResultBentNormal;
        }


        return nullptr;
    }
}