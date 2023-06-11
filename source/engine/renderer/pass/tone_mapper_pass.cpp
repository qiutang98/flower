#include "../renderer_interface.h"
#include "../render_scene.h"
#include "../renderer.h"
#include "../scene_textures.h"

namespace engine
{
    struct TonemapperPushComposite
    {
        math::vec4 prefilterFactor;
        float bloomIntensity;
        float bloomBlur;
    };

    struct ExposureApplyPushConsts
    {
        float black     = 0.25f;
        float shadow    = 0.5f;
        float highLight = 2.0f;
    };

    struct ExposureWeightPushConsts
    {
        float kSigma = 0.2f;
        float kWellExposureValue = 0.5f;
        float kContrastPow = 1.0f;
        float kSaturationPow = 1.0f;
        float kExposurePow = 1.0f;
    };

    struct FusionGaussianPushConst
    {
        math::vec2 kDirection;
    };

    struct TonemapperPostPushComposite
    {
        uint32_t bExposureFusion;
    };

    class TonemapperPass : public PassInterface
    {
    public:
        VkDescriptorSetLayout setLayoutCombine = VK_NULL_HANDLE;
        std::unique_ptr<ComputePipeResources> pipeCombine;

        VkDescriptorSetLayout setLayoutTone = VK_NULL_HANDLE;
        std::unique_ptr<ComputePipeResources> pipeTone;


        VkDescriptorSetLayout setLayoutExposureApply = VK_NULL_HANDLE;
        std::unique_ptr<ComputePipeResources> pipeExposureApply;

        VkDescriptorSetLayout setLayoutExposureWieght = VK_NULL_HANDLE;
        std::unique_ptr<ComputePipeResources> pipeExposureWeight;


        VkDescriptorSetLayout setLayoutFusionGaussian = VK_NULL_HANDLE;
        std::unique_ptr<ComputePipeResources> pipeFusionGaussian;
        std::unique_ptr<ComputePipeResources> pipeFusionDownsample;

        VkDescriptorSetLayout setLayoutLaplace4 = VK_NULL_HANDLE;
        std::unique_ptr<ComputePipeResources> pipeFusionLaplace4;

        VkDescriptorSetLayout setLayoutFusionBlend = VK_NULL_HANDLE;
        std::unique_ptr<ComputePipeResources> pipeFusionBlend;

        VkDescriptorSetLayout setLayoutFusion = VK_NULL_HANDLE;
        std::unique_ptr<ComputePipeResources> pipeFusion;

        VkDescriptorSetLayout setLayoutG4 = VK_NULL_HANDLE;
        std::unique_ptr<ComputePipeResources> pipeG4;

        std::unique_ptr<ComputePipeResources> pipeD4;

    public:
        virtual void onInit() override
        {
            // Config code.
            getContext()->descriptorFactoryBegin()
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0) // outLdr
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1) // outLdr
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 2) // outLdr
                .bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 3) // uniform
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 4) // inHdr
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 5)
                .buildNoInfoPush(setLayoutCombine);

            getContext()->descriptorFactoryBegin()
                .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0) // outLdr
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1) // uniform
                .bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 2) // uniform
                .buildNoInfoPush(setLayoutTone);

            std::vector<VkDescriptorSetLayout> setLayoutsCombine = {
                setLayoutCombine,
                getContext()->getSamplerCache().getCommonDescriptorSetLayout(),
                getRenderer()->getBlueNoise().spp_1_buffer.setLayouts
            };

            pipeCombine = std::make_unique<ComputePipeResources>("shader/pp_combine.comp.spv", (uint32_t)sizeof(TonemapperPushComposite), setLayoutsCombine);

            std::vector<VkDescriptorSetLayout> setLayoutsTone = {
                setLayoutTone,
                getContext()->getSamplerCache().getCommonDescriptorSetLayout(),
                getRenderer()->getBlueNoise().spp_1_buffer.setLayouts
            };

            pipeTone = std::make_unique<ComputePipeResources>("shader/pp_tonemapper.comp.spv", (uint32_t)sizeof(TonemapperPostPushComposite), setLayoutsTone);


            {
                getContext()->descriptorFactoryBegin()
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0)
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1) 
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 2)
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 3)
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 4)
                    .buildNoInfoPush(setLayoutExposureApply);

                std::vector<VkDescriptorSetLayout> setLayouts = 
                {
                    setLayoutExposureApply,
                    getContext()->getSamplerCache().getCommonDescriptorSetLayout()
                };
                pipeExposureApply = std::make_unique<ComputePipeResources>("shader/pp_exposure_apply.comp.spv", (uint32_t)sizeof(ExposureApplyPushConsts), setLayouts);
            }

            {
                getContext()->descriptorFactoryBegin()
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  VK_SHADER_STAGE_COMPUTE_BIT, 0)
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  VK_SHADER_STAGE_COMPUTE_BIT, 1)
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  VK_SHADER_STAGE_COMPUTE_BIT, 2)
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  VK_SHADER_STAGE_COMPUTE_BIT, 3)
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 4)
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  VK_SHADER_STAGE_COMPUTE_BIT, 5)
                    .buildNoInfoPush(setLayoutExposureWieght);

                std::vector<VkDescriptorSetLayout> setLayouts =
                {
                    setLayoutExposureWieght,
                    getContext()->getSamplerCache().getCommonDescriptorSetLayout()
                };
                pipeExposureWeight = std::make_unique<ComputePipeResources>("shader/pp_exposure_weight.comp.spv", (uint32_t)sizeof(ExposureWeightPushConsts), setLayouts);
            }

            {
                getContext()->descriptorFactoryBegin()
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0)
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1)
                    .buildNoInfoPush(setLayoutFusionGaussian);

                std::vector<VkDescriptorSetLayout> setLayouts =
                {
                    setLayoutFusionGaussian,
                    getContext()->getSamplerCache().getCommonDescriptorSetLayout()
                };
                pipeFusionGaussian = std::make_unique<ComputePipeResources>("shader/pp_fusion_gaussian.comp.spv", (uint32_t)sizeof(FusionGaussianPushConst), setLayouts);
            }

            {
                std::vector<VkDescriptorSetLayout> setLayouts =
                {
                    setLayoutFusionGaussian,
                    getContext()->getSamplerCache().getCommonDescriptorSetLayout()
                };
                pipeFusionDownsample = std::make_unique<ComputePipeResources>("shader/down_point.comp.spv", 0, setLayouts);
            }

            {
                getContext()->descriptorFactoryBegin()
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0)
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1)
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 2)
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 3)
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 4)
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 5)
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 6)
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 7)
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 8)
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 9)
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 10)
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 11)
                    .buildNoInfoPush(setLayoutLaplace4);

                std::vector<VkDescriptorSetLayout> setLayouts =
                {
                    setLayoutLaplace4,
                    getContext()->getSamplerCache().getCommonDescriptorSetLayout()
                };
                pipeFusionLaplace4 = std::make_unique<ComputePipeResources>("shader/pp_fusion_laplace4.comp.spv", 0, setLayouts);
            }

            {
                getContext()->descriptorFactoryBegin()
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0)
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1)
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 2)
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 3)
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 4)
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 5)
                    .buildNoInfoPush(setLayoutFusionBlend);

                std::vector<VkDescriptorSetLayout> setLayouts =
                {
                    setLayoutFusionBlend,
                    getContext()->getSamplerCache().getCommonDescriptorSetLayout()
                };
                pipeFusionBlend = std::make_unique<ComputePipeResources>("shader/pp_fusion_blend.comp.spv", 0, setLayouts);
            }

            {
                getContext()->descriptorFactoryBegin()
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0)
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1)
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 2)
                    .buildNoInfoPush(setLayoutFusion);

                std::vector<VkDescriptorSetLayout> setLayouts =
                {
                    setLayoutFusion,
                    getContext()->getSamplerCache().getCommonDescriptorSetLayout()
                };
                pipeFusion = std::make_unique<ComputePipeResources>("shader/pp_fusion.comp.spv", 0, setLayouts);
            }

            {
                getContext()->descriptorFactoryBegin()
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0)
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1)
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 2)
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 3)
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 4)
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 5)
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 6)
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 7)
                    .buildNoInfoPush(setLayoutG4);

                std::vector<VkDescriptorSetLayout> setLayouts =
                {
                    setLayoutG4,
                    getContext()->getSamplerCache().getCommonDescriptorSetLayout()
                };
                pipeG4 = std::make_unique<ComputePipeResources>("shader/gaussian4.comp.spv", (uint32_t)sizeof(FusionGaussianPushConst), setLayouts);
            }

            {
                std::vector<VkDescriptorSetLayout> setLayouts =
                {
                    setLayoutG4,
                    getContext()->getSamplerCache().getCommonDescriptorSetLayout()
                };
                pipeD4 = std::make_unique<ComputePipeResources>("shader/down_point4.comp.spv", 0, setLayouts);
            }
        }

        virtual void release() override
        {
            pipeCombine.reset();
            pipeTone.reset();

            pipeExposureApply.reset();
            pipeExposureWeight.reset();

            pipeFusionGaussian.reset();
            pipeFusionDownsample.reset();

            pipeFusionLaplace4.reset();
            pipeFusionBlend.reset();

            pipeFusion.reset();
            pipeG4.reset();
            pipeD4.reset();
        }
    };


    void RendererInterface::renderTonemapper(VkCommandBuffer cmd, GBufferTextures* inGBuffers, BufferParameterHandle perFrameGPU, RenderScene* scene, PoolImageSharedRef bloomTex,
        BufferParameterHandle lensBuffer)
    {
        auto& hdrSceneColor = inGBuffers->hdrSceneColorUpscale->getImage();
        auto& ldrSceneColor = getDisplayOutput();

        auto* pass = getContext()->getPasses().get<TonemapperPass>();
        const auto& postProcessVolumeSetting = scene->getPostprocessVolumeSetting();

        auto* rtPool = &m_context->getRenderTargetPools();
        auto combineResult = rtPool->createPoolImage(
            "combineResult",
            hdrSceneColor.getExtent().width,
            hdrSceneColor.getExtent().height,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

        {
            combineResult->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            hdrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

        }

        {
            ScopePerframeMarker tonemapperMarker(cmd, "Tonemapper", { 1.0f, 1.0f, 0.0f, 1.0f });


            TonemapperPushComposite compositePush
            {
                .prefilterFactor = getBloomPrefilter(postProcessVolumeSetting.bloomThreshold, postProcessVolumeSetting.bloomThresholdSoft),
                .bloomIntensity = postProcessVolumeSetting.bloomIntensity,
                .bloomBlur = postProcessVolumeSetting.bloomRadius,
            };

            pass->pipeCombine->bindAndPushConst(cmd, &compositePush);
            PushSetBuilder pusher(cmd);
            pusher
                .addSRV(hdrSceneColor)
                .addUAV(combineResult)
                .addSRV(m_averageLum ? m_averageLum : inGBuffers->hdrSceneColorUpscale)
                .addBuffer(perFrameGPU)
                .addSRV(bloomTex);
            if (lensBuffer)
            {
                pusher.addBuffer(lensBuffer);
            }
            else
            {
                auto lensSSBO = m_context->getBufferParameters().getStaticStorage(
                    "lensSSBO", sizeof(float));
                pusher.addBuffer(lensSSBO);
            }

            pusher.push(pass->pipeCombine.get());

            pass->pipeCombine->bindSet(cmd, std::vector<VkDescriptorSet>{
                m_context->getSamplerCache().getCommonDescriptorSet()
                    , m_renderer->getBlueNoise().spp_1_buffer.set
            }, 1);

            vkCmdDispatch(cmd, getGroupCount(hdrSceneColor.getExtent().width, 8), getGroupCount(hdrSceneColor.getExtent().height, 8), 1);

            ///////////////////

            combineResult->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());


            m_gpuTimer.getTimeStamp(cmd, "Combine");
        }


        PoolImageSharedRef blendFusion = nullptr;
        if (postProcessVolumeSetting.bEnableExposureFusion)
        {
            ScopePerframeMarker marker(cmd, "Exposure Fusion", { 1.0f, 1.0f, 0.0f, 1.0f });

            PoolImageSharedRef color0;
            PoolImageSharedRef color1;
            PoolImageSharedRef color2;
            PoolImageSharedRef color3;

            PoolImageSharedRef weight;

            {
                uint32_t w = hdrSceneColor.getExtent().width;
                uint32_t h = hdrSceneColor.getExtent().height;

                color0 = rtPool->createPoolImage("c0", w, h, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
                color1 = rtPool->createPoolImage("c1", w, h, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
                color2 = rtPool->createPoolImage("c2", w, h, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
                color3 = rtPool->createPoolImage("c3", w, h, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

                color0->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                color1->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                color2->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                color3->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

                {
                    ExposureApplyPushConsts push{ };

                    push.black = postProcessVolumeSetting.exposureFusionBlack;
                    push.shadow = postProcessVolumeSetting.exposureFusionShadows;
                    push.highLight = postProcessVolumeSetting.exposureFusionHighlights;

                    pass->pipeExposureApply->bindAndPushConst(cmd, &push);
                    PushSetBuilder(cmd)
                        .addSRV(combineResult)
                        .addUAV(color0)
                        .addUAV(color1)
                        .addUAV(color2)
                        .addUAV(color3)
                        .push(pass->pipeExposureApply.get());

                    pass->pipeExposureApply->bindSet(cmd, std::vector<VkDescriptorSet>
                    {
                        m_context->getSamplerCache().getCommonDescriptorSet()
                    }, 1);

                    vkCmdDispatch(cmd, getGroupCount(w, 8), getGroupCount(h, 8), 1);
                }

                color0->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
                color1->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
                color2->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
                color3->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

                weight = rtPool->createPoolImage("weight", w, h, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
                weight->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                {
                    ExposureWeightPushConsts push{ };
                    push.kContrastPow = postProcessVolumeSetting.exposureFusionContrastPow;
                    push.kExposurePow = postProcessVolumeSetting.exposureFusionExposurePow;
                    push.kSaturationPow = postProcessVolumeSetting.exposureFusionSaturationPow;
                    push.kSigma = postProcessVolumeSetting.exposureFusionSigma;
                    push.kWellExposureValue = postProcessVolumeSetting.exposureFusionWellExposureValue;

                    pass->pipeExposureWeight->bindAndPushConst(cmd, &push);
                    PushSetBuilder(cmd)
                        .addSRV(color0)
                        .addSRV(color1)
                        .addSRV(color2)
                        .addSRV(color3)
                        .addBuffer(perFrameGPU)
                        .addUAV(weight)
                        .push(pass->pipeExposureWeight.get());

                    pass->pipeExposureWeight->bindSet(cmd, std::vector<VkDescriptorSet>
                    {
                        m_context->getSamplerCache().getCommonDescriptorSet()
                    }, 1);

                    vkCmdDispatch(cmd, getGroupCount(w, 8), getGroupCount(h, 8), 1);
                }
                weight->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            }

            std::vector<VkDescriptorSet> setSample = { m_context->getSamplerCache().getCommonDescriptorSet() };

            m_gpuTimer.getTimeStamp(cmd, "weight");


            // Gaussian pyramid for weight.
            {


                auto buidlGaussianPyramid = [&](PoolImageSharedRef inSrc, int depth)
                {
                    ScopePerframeMarker marker(cmd, "buidlGaussianPyramid", { 1.0f, 1.0f, 0.0f, 1.0f });

                    std::vector<PoolImageSharedRef> gaussianBlurs{ };
                    gaussianBlurs.push_back(inSrc);

                    // Create rts.
                    {
                        uint32_t w = inSrc->getImage().getExtent().width / 2;
                        uint32_t h = inSrc->getImage().getExtent().height / 2;
                        while (depth > 0 && w > 0 && h > 0)
                        {
                            gaussianBlurs.push_back(rtPool->createPoolImage("d", w, h, inSrc->getImage().getFormat(), VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT));

                            w /= 2;
                            h /= 2;
                            depth--;
                        }
                    }

                    std::vector<VkDescriptorSet> setSample = { m_context->getSamplerCache().getCommonDescriptorSet() };

                    for (size_t i = 1; i < gaussianBlurs.size(); i++)
                    {
                        const auto& srcIn = gaussianBlurs[i - 1];

                        uint32_t w = srcIn->getImage().getExtent().width;
                        uint32_t h = srcIn->getImage().getExtent().height;



                        // blur x.
                        auto tempX = rtPool->createPoolImage("d", w, h, inSrc->getImage().getFormat(), VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
                        tempX->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                        {
                            ScopePerframeMarker marker(cmd, "blur x", { 1.0f, 1.0f, 0.0f, 1.0f });

                            FusionGaussianPushConst push{ .kDirection = {1.0f, 0.0f} };

                            pass->pipeFusionGaussian->bindAndPushConst(cmd, &push);
                            PushSetBuilder(cmd)
                                .addSRV(srcIn)
                                .addUAV(tempX)
                                .push(pass->pipeFusionGaussian.get());
                            pass->pipeFusionGaussian->bindSet(cmd, setSample, 1);

                            vkCmdDispatch(cmd, getGroupCount(w, 8), getGroupCount(h, 8), 1);
                        }
                        tempX->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

                        // blur y.
                        auto tempY = rtPool->createPoolImage("d", w, h, inSrc->getImage().getFormat(), VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
                        tempY->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                        {
                            ScopePerframeMarker marker(cmd, "blur y", { 1.0f, 1.0f, 0.0f, 1.0f });

                            FusionGaussianPushConst push{ .kDirection = {0.0f, 1.0f} };

                            pass->pipeFusionGaussian->bindAndPushConst(cmd, &push);
                            PushSetBuilder(cmd)
                                .addSRV(tempX)
                                .addUAV(tempY)
                                .push(pass->pipeFusionGaussian.get());
                            pass->pipeFusionGaussian->bindSet(cmd, setSample, 1);

                            vkCmdDispatch(cmd, getGroupCount(w, 8), getGroupCount(h, 8), 1);
                        }
                        tempY->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

                        // Down sample to out.
                        gaussianBlurs[i]->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                        {
                            ScopePerframeMarker marker(cmd, "down", { 1.0f, 1.0f, 0.0f, 1.0f });

                            pass->pipeFusionDownsample->bind(cmd);
                            PushSetBuilder(cmd)
                                .addSRV(tempY)
                                .addUAV(gaussianBlurs[i])
                                .push(pass->pipeFusionDownsample.get());
                            pass->pipeFusionDownsample->bindSet(cmd, setSample, 1);

                            vkCmdDispatch(cmd, getGroupCount(gaussianBlurs[i]->getImage().getExtent().width, 8), getGroupCount(gaussianBlurs[i]->getImage().getExtent().height, 8), 1);
                        }
                        gaussianBlurs[i]->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
                    }

                    return gaussianBlurs;
                };

                auto buidlGaussianPyramid4 = [&](
                    PoolImageSharedRef inSrc0, PoolImageSharedRef inSrc1, PoolImageSharedRef inSrc2, PoolImageSharedRef inSrc3,
                    std::vector<PoolImageSharedRef>& out0, std::vector<PoolImageSharedRef>& out1, std::vector<PoolImageSharedRef>& out2, std::vector<PoolImageSharedRef>& out3)
                {
                    ScopePerframeMarker marker(cmd, "buidlGaussianPyramid4", { 1.0f, 1.0f, 0.0f, 1.0f });

                    out0.clear();
                    out1.clear();
                    out2.clear();
                    out3.clear();

                    out0.push_back(inSrc0);
                    out1.push_back(inSrc1);
                    out2.push_back(inSrc2);
                    out3.push_back(inSrc3);

                    // Create rts.
                    {
                        uint32_t w = inSrc0->getImage().getExtent().width / 2;
                        uint32_t h = inSrc0->getImage().getExtent().height / 2;
                        while (w > 0 && h > 0)
                        {
                            out0.push_back(rtPool->createPoolImage("d0", w, h, inSrc0->getImage().getFormat(), VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT));
                            out1.push_back(rtPool->createPoolImage("d1", w, h, inSrc1->getImage().getFormat(), VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT));
                            out2.push_back(rtPool->createPoolImage("d2", w, h, inSrc2->getImage().getFormat(), VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT));
                            out3.push_back(rtPool->createPoolImage("d3", w, h, inSrc3->getImage().getFormat(), VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT));

                            w /= 2;
                            h /= 2;
                        }
                    }

                    std::vector<VkDescriptorSet> setSample = { m_context->getSamplerCache().getCommonDescriptorSet() };

                    for (size_t i = 1; i < out0.size(); i++)
                    {
                        const auto& srcIn0 = out0[i - 1];
                        const auto& srcIn1 = out1[i - 1];
                        const auto& srcIn2 = out2[i - 1];
                        const auto& srcIn3 = out3[i - 1];

                        uint32_t w = srcIn0->getImage().getExtent().width;
                        uint32_t h = srcIn0->getImage().getExtent().height;

                        // blur x.
                        auto tempX0 = rtPool->createPoolImage("d0", w, h, inSrc0->getImage().getFormat(), VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
                        auto tempX1 = rtPool->createPoolImage("d1", w, h, inSrc1->getImage().getFormat(), VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
                        auto tempX2 = rtPool->createPoolImage("d2", w, h, inSrc2->getImage().getFormat(), VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
                        auto tempX3 = rtPool->createPoolImage("d3", w, h, inSrc3->getImage().getFormat(), VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
                        tempX0->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                        tempX1->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                        tempX2->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                        tempX3->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                        {
                            ScopePerframeMarker marker(cmd, "blur x 4", { 1.0f, 1.0f, 0.0f, 1.0f });

                            FusionGaussianPushConst push{ .kDirection = {1.0f, 0.0f} };

                            pass->pipeG4->bindAndPushConst(cmd, &push);
                            PushSetBuilder(cmd)
                                .addSRV(srcIn0)
                                .addSRV(srcIn1)
                                .addSRV(srcIn2)
                                .addSRV(srcIn3)
                                .addUAV(tempX0)
                                .addUAV(tempX1)
                                .addUAV(tempX2)
                                .addUAV(tempX3)
                                .push(pass->pipeG4.get());
                            pass->pipeG4->bindSet(cmd, setSample, 1);

                            vkCmdDispatch(cmd, getGroupCount(w, 8), getGroupCount(h, 8), 1);
                        }
                        tempX0->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
                        tempX1->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
                        tempX2->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
                        tempX3->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

                        // blur y.
                        auto tempY0 = rtPool->createPoolImage("d", w, h, inSrc0->getImage().getFormat(), VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
                        auto tempY1 = rtPool->createPoolImage("d", w, h, inSrc1->getImage().getFormat(), VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
                        auto tempY2 = rtPool->createPoolImage("d", w, h, inSrc2->getImage().getFormat(), VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
                        auto tempY3 = rtPool->createPoolImage("d", w, h, inSrc3->getImage().getFormat(), VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
                        tempY0->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                        tempY1->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                        tempY2->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                        tempY3->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                        {
                            ScopePerframeMarker marker(cmd, "blur y 4", { 1.0f, 1.0f, 0.0f, 1.0f });

                            FusionGaussianPushConst push{ .kDirection = {0.0f, 1.0f} };

                            pass->pipeG4->bindAndPushConst(cmd, &push);
                            PushSetBuilder(cmd)
                                .addSRV(tempX0)
                                .addSRV(tempX1)
                                .addSRV(tempX2)
                                .addSRV(tempX3)
                                .addUAV(tempY0)
                                .addUAV(tempY1)
                                .addUAV(tempY2)
                                .addUAV(tempY3)
                                .push(pass->pipeG4.get());
                            pass->pipeG4->bindSet(cmd, setSample, 1);

                            vkCmdDispatch(cmd, getGroupCount(w, 8), getGroupCount(h, 8), 1);
                        }
                        tempY0->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
                        tempY1->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
                        tempY2->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
                        tempY3->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

                        // Down sample to out.
                        out0[i]->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                        out1[i]->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                        out2[i]->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                        out3[i]->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                        {
                            ScopePerframeMarker marker(cmd, "down 4", { 1.0f, 1.0f, 0.0f, 1.0f });

                            pass->pipeD4->bind(cmd);
                            PushSetBuilder(cmd)
                                .addSRV(tempY0)
                                .addSRV(tempY1)
                                .addSRV(tempY2)
                                .addSRV(tempY3)
                                .addUAV(out0[i])
                                .addUAV(out1[i])
                                .addUAV(out2[i])
                                .addUAV(out3[i])
                                .push(pass->pipeD4.get());
                            pass->pipeD4->bindSet(cmd, setSample, 1);

                            vkCmdDispatch(cmd, getGroupCount(out0[i]->getImage().getExtent().width, 8), getGroupCount(out0[i]->getImage().getExtent().height, 8), 1);
                        }
                        out0[i]->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
                        out1[i]->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
                        out2[i]->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
                        out3[i]->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
                    }
                };


                auto buildLaplacePyramid4 = [&](
                    std::vector<PoolImageSharedRef>& gaussianPyramid0, std::vector<PoolImageSharedRef>& gaussianPyramid1, std::vector<PoolImageSharedRef>& gaussianPyramid2, std::vector<PoolImageSharedRef>& gaussianPyramid3,
                    std::vector<PoolImageSharedRef>& laplacePyramid0, std::vector<PoolImageSharedRef>& laplacePyramid1, std::vector<PoolImageSharedRef>& laplacePyramid2, std::vector<PoolImageSharedRef>& laplacePyramid3)
                {
                    ScopePerframeMarker marker(cmd, "buildLaplacePyramid 4", { 1.0f, 1.0f, 0.0f, 1.0f });

                    laplacePyramid0.clear();
                    laplacePyramid1.clear();
                    laplacePyramid2.clear();
                    laplacePyramid3.clear();

                    for (size_t i = 0; i < gaussianPyramid0.size() - 1; i++)
                    {
                        auto& src0 = gaussianPyramid0[i];
                        auto& src1 = gaussianPyramid1[i];
                        auto& src2 = gaussianPyramid2[i];
                        auto& src3 = gaussianPyramid3[i];
                        auto& low0 = gaussianPyramid0[i + 1];
                        auto& low1 = gaussianPyramid1[i + 1];
                        auto& low2 = gaussianPyramid2[i + 1];
                        auto& low3 = gaussianPyramid3[i + 1];

                        uint32_t w = src0->getImage().getExtent().width;
                        uint32_t h = src0->getImage().getExtent().height;

                        auto upscale0 = rtPool->createPoolImage("x0", w, h, src0->getImage().getFormat(), VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
                        auto upscale1 = rtPool->createPoolImage("x1", w, h, src1->getImage().getFormat(), VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
                        auto upscale2 = rtPool->createPoolImage("x2", w, h, src2->getImage().getFormat(), VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
                        auto upscale3 = rtPool->createPoolImage("x3", w, h, src3->getImage().getFormat(), VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
                        upscale0->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                        upscale1->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                        upscale2->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                        upscale3->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                        {
                            ScopePerframeMarker marker(cmd, "upscale 4", { 1.0f, 1.0f, 0.0f, 1.0f });

                            pass->pipeD4->bind(cmd);
                            PushSetBuilder(cmd)
                                .addSRV(low0)
                                .addSRV(low1)
                                .addSRV(low2)
                                .addSRV(low3)
                                .addUAV(upscale0)
                                .addUAV(upscale1)
                                .addUAV(upscale2)
                                .addUAV(upscale3)
                                .push(pass->pipeD4.get());
                            pass->pipeD4->bindSet(cmd, setSample, 1);

                            vkCmdDispatch(cmd, getGroupCount(upscale0->getImage().getExtent().width, 8), getGroupCount(upscale0->getImage().getExtent().height, 8), 1);
                        }
                        upscale0->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
                        upscale1->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
                        upscale2->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
                        upscale3->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

                        // blur x.
                        auto tempX0 = rtPool->createPoolImage("d 0", w, h, src0->getImage().getFormat(), VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
                        auto tempX1 = rtPool->createPoolImage("d 1", w, h, src1->getImage().getFormat(), VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
                        auto tempX2 = rtPool->createPoolImage("d 2", w, h, src2->getImage().getFormat(), VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
                        auto tempX3 = rtPool->createPoolImage("d 3", w, h, src3->getImage().getFormat(), VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
                        tempX0->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                        tempX1->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                        tempX2->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                        tempX3->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                        {
                            ScopePerframeMarker marker(cmd, "blur x 4", { 1.0f, 1.0f, 0.0f, 1.0f });

                            FusionGaussianPushConst push{ .kDirection = {1.0f, 0.0f} };

                            pass->pipeG4->bindAndPushConst(cmd, &push);
                            PushSetBuilder(cmd)
                                .addSRV(upscale0)
                                .addSRV(upscale1)
                                .addSRV(upscale2)
                                .addSRV(upscale3)
                                .addUAV(tempX0)
                                .addUAV(tempX1)
                                .addUAV(tempX2)
                                .addUAV(tempX3)
                                .push(pass->pipeG4.get());
                            pass->pipeG4->bindSet(cmd, setSample, 1);

                            vkCmdDispatch(cmd, getGroupCount(w, 8), getGroupCount(h, 8), 1);
                        }
                        tempX0->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
                        tempX1->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
                        tempX2->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
                        tempX3->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

                        // blur y.
                        auto tempY0 = rtPool->createPoolImage("d 0", w, h, src0->getImage().getFormat(), VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
                        auto tempY1 = rtPool->createPoolImage("d 1", w, h, src1->getImage().getFormat(), VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
                        auto tempY2 = rtPool->createPoolImage("d 2", w, h, src2->getImage().getFormat(), VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
                        auto tempY3 = rtPool->createPoolImage("d 3", w, h, src3->getImage().getFormat(), VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
                        tempY0->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                        tempY1->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                        tempY2->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                        tempY3->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                        {
                            ScopePerframeMarker marker(cmd, "blur y", { 1.0f, 1.0f, 0.0f, 1.0f });

                            FusionGaussianPushConst push{ .kDirection = {0.0f, 1.0f} };

                            pass->pipeG4->bindAndPushConst(cmd, &push);
                            PushSetBuilder(cmd)
                                .addSRV(tempX0)
                                .addSRV(tempX1)
                                .addSRV(tempX2)
                                .addSRV(tempX3)
                                .addUAV(tempY0)
                                .addUAV(tempY1)
                                .addUAV(tempY2)
                                .addUAV(tempY3)
                                .push(pass->pipeG4.get());
                            pass->pipeG4->bindSet(cmd, setSample, 1);

                            vkCmdDispatch(cmd, getGroupCount(w, 8), getGroupCount(h, 8), 1);
                        }
                        tempY0->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
                        tempY1->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
                        tempY2->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
                        tempY3->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

                        // Blur ready, then compute laplace.
                        auto laplace0 = rtPool->createPoolImage("l 0", w, h, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
                        auto laplace1 = rtPool->createPoolImage("l 1", w, h, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
                        auto laplace2 = rtPool->createPoolImage("l 2", w, h, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
                        auto laplace3 = rtPool->createPoolImage("l 3", w, h, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
                        laplace0->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                        laplace1->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                        laplace2->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                        laplace3->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                        {
                            ScopePerframeMarker marker(cmd, "laplace", { 1.0f, 1.0f, 0.0f, 1.0f });

                            pass->pipeFusionLaplace4->bind(cmd);
                            PushSetBuilder(cmd)
                                .addSRV(tempY0)
                                .addSRV(tempY1)
                                .addSRV(tempY2)
                                .addSRV(tempY3)
                                .addSRV(src0)
                                .addSRV(src1)
                                .addSRV(src2)
                                .addSRV(src3)
                                .addUAV(laplace0)
                                .addUAV(laplace1)
                                .addUAV(laplace2)
                                .addUAV(laplace3)
                                .push(pass->pipeFusionLaplace4.get());
                            pass->pipeFusionLaplace4->bindSet(cmd, setSample, 1);

                            vkCmdDispatch(cmd, getGroupCount(w, 8), getGroupCount(h, 8), 1);
                        }
                        laplace0->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
                        laplace1->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
                        laplace2->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
                        laplace3->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

                        laplacePyramid0.push_back(laplace0);
                        laplacePyramid1.push_back(laplace1);
                        laplacePyramid2.push_back(laplace2);
                        laplacePyramid3.push_back(laplace3);
                    }

                    // Last of laplace pyramid just copy.
                    laplacePyramid0.push_back(gaussianPyramid0.back());
                    laplacePyramid1.push_back(gaussianPyramid1.back());
                    laplacePyramid2.push_back(gaussianPyramid2.back());
                    laplacePyramid3.push_back(gaussianPyramid3.back());
                };

                const int kDepth = 999;
                auto weightPyraimd = buidlGaussianPyramid(weight, kDepth);

                m_gpuTimer.getTimeStamp(cmd, "Weight Gaussian");

                std::vector<PoolImageSharedRef> laplace0;
                std::vector<PoolImageSharedRef> laplace1;
                std::vector<PoolImageSharedRef> laplace2;
                std::vector<PoolImageSharedRef> laplace3;
                {
                    std::vector<PoolImageSharedRef> g0;
                    std::vector<PoolImageSharedRef> g1;
                    std::vector<PoolImageSharedRef> g2;
                    std::vector<PoolImageSharedRef> g3;

                    buidlGaussianPyramid4(color0, color1, color2, color3, g0, g1, g2, g3);

                    buildLaplacePyramid4(g0, g1, g2, g3, laplace0, laplace1, laplace2, laplace3);
                }

                m_gpuTimer.getTimeStamp(cmd, "laplace4");

                std::vector<PoolImageSharedRef> fusionBlend;
                for (size_t i = 0; i < laplace0.size(); i++)
                {
                    uint32_t w = laplace0[i]->getImage().getExtent().width;
                    uint32_t h = laplace0[i]->getImage().getExtent().height;

                    PoolImageSharedRef blend = rtPool->createPoolImage("blend", w, h, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
                    blend->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                    {
                        ScopePerframeMarker marker(cmd, "blend", { 1.0f, 1.0f, 0.0f, 1.0f });

                        pass->pipeFusionBlend->bind(cmd);
                        PushSetBuilder(cmd)
                            .addSRV(laplace0[i])
                            .addSRV(laplace1[i])
                            .addSRV(laplace2[i])
                            .addSRV(laplace3[i])
                            .addSRV(weightPyraimd[i])
                            .addUAV(blend)
                            .push(pass->pipeFusionBlend.get());
                        pass->pipeFusionBlend->bindSet(cmd, setSample, 1);

                        vkCmdDispatch(cmd, getGroupCount(w, 8), getGroupCount(h, 8), 1);
                    }
                    blend->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

                    fusionBlend.push_back(blend);
                }

                m_gpuTimer.getTimeStamp(cmd, "blend");

                // Fusion

                for (int i = fusionBlend.size() - 2; i >= 0; i--)
                {
                    auto& lowImage = blendFusion ? blendFusion : fusionBlend[i + 1];
                    auto& highImage = fusionBlend[i];

                    // Upscale low image with gaussian blur.
                    uint32_t w = highImage->getImage().getExtent().width;
                    uint32_t h = highImage->getImage().getExtent().height;

                    auto upscale = rtPool->createPoolImage("x", w, h, highImage->getImage().getFormat(), VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
                    upscale->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                    {
                        ScopePerframeMarker marker(cmd, "upscale", { 1.0f, 1.0f, 0.0f, 1.0f });

                        pass->pipeFusionDownsample->bind(cmd);
                        PushSetBuilder(cmd)
                            .addSRV(lowImage)
                            .addUAV(upscale)
                            .push(pass->pipeFusionDownsample.get());
                        pass->pipeFusionDownsample->bindSet(cmd, setSample, 1);

                        vkCmdDispatch(cmd, getGroupCount(upscale->getImage().getExtent().width, 8), getGroupCount(upscale->getImage().getExtent().height, 8), 1);
                    }
                    upscale->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

                    // blur x.
                    auto tempX = rtPool->createPoolImage("d", w, h, highImage->getImage().getFormat(), VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
                    tempX->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                    {
                        ScopePerframeMarker marker(cmd, "blur x", { 1.0f, 1.0f, 0.0f, 1.0f });

                        FusionGaussianPushConst push{ .kDirection = {1.0f, 0.0f} };

                        pass->pipeFusionGaussian->bindAndPushConst(cmd, &push);
                        PushSetBuilder(cmd)
                            .addSRV(upscale)
                            .addUAV(tempX)
                            .push(pass->pipeFusionGaussian.get());
                        pass->pipeFusionGaussian->bindSet(cmd, setSample, 1);

                        vkCmdDispatch(cmd, getGroupCount(w, 8), getGroupCount(h, 8), 1);
                    }
                    tempX->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

                    // blur y.
                    auto tempY = rtPool->createPoolImage("d", w, h, highImage->getImage().getFormat(), VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
                    tempY->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                    {
                        ScopePerframeMarker marker(cmd, "blur y", { 1.0f, 1.0f, 0.0f, 1.0f });

                        FusionGaussianPushConst push{ .kDirection = {0.0f, 1.0f} };

                        pass->pipeFusionGaussian->bindAndPushConst(cmd, &push);
                        PushSetBuilder(cmd)
                            .addSRV(tempX)
                            .addUAV(tempY)
                            .push(pass->pipeFusionGaussian.get());
                        pass->pipeFusionGaussian->bindSet(cmd, setSample, 1);

                        vkCmdDispatch(cmd, getGroupCount(w, 8), getGroupCount(h, 8), 1);
                    }
                    tempY->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());


                    auto blendImage = rtPool->createPoolImage("fusion", w, h, highImage->getImage().getFormat(), VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
                    blendImage->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                    {
                        ScopePerframeMarker marker(cmd, "fusion", { 1.0f, 1.0f, 0.0f, 1.0f });

                        pass->pipeFusion->bind(cmd);
                        PushSetBuilder(cmd)
                            .addSRV(highImage)
                            .addSRV(tempY)
                            .addUAV(blendImage)
                            .push(pass->pipeFusion.get());
                        pass->pipeFusion->bindSet(cmd, setSample, 1);

                        vkCmdDispatch(cmd, getGroupCount(w, 8), getGroupCount(h, 8), 1);
                    }
                    blendImage->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

                    blendFusion = blendImage;
                }


                m_gpuTimer.getTimeStamp(cmd, "Fusion");
            }
        }
        
        {
            ldrSceneColor.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

            {
                TonemapperPostPushComposite push{};
                push.bExposureFusion = postProcessVolumeSetting.bEnableExposureFusion ? 1U : 0U;
                pass->pipeTone->bindAndPushConst(cmd, &push);
                PushSetBuilder(cmd)
                    .addSRV(blendFusion ? blendFusion : combineResult)
                    .addUAV(ldrSceneColor)
                    .addBuffer(perFrameGPU)
                    .push(pass->pipeTone.get());

                pass->pipeTone->bindSet(cmd, std::vector<VkDescriptorSet>{
                    m_context->getSamplerCache().getCommonDescriptorSet()
                  , m_renderer->getBlueNoise().spp_1_buffer.set
                }, 1);

                vkCmdDispatch(cmd, getGroupCount(ldrSceneColor.getExtent().width, 8), getGroupCount(ldrSceneColor.getExtent().height, 8), 1);
            }


            m_gpuTimer.getTimeStamp(cmd, "Tonemappering");
        }
    }
}