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

        VkDescriptorSetLayout setLayoutLaplace = VK_NULL_HANDLE;
        std::unique_ptr<ComputePipeResources> pipeFusionLaplace;

        VkDescriptorSetLayout setLayoutFusionBlend = VK_NULL_HANDLE;
        std::unique_ptr<ComputePipeResources> pipeFusionBlend;

        VkDescriptorSetLayout setLayoutFusion = VK_NULL_HANDLE;
        std::unique_ptr<ComputePipeResources> pipeFusion;

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
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 2)
                    .buildNoInfoPush(setLayoutLaplace);

                std::vector<VkDescriptorSetLayout> setLayouts =
                {
                    setLayoutLaplace,
                    getContext()->getSamplerCache().getCommonDescriptorSetLayout()
                };
                pipeFusionLaplace = std::make_unique<ComputePipeResources>("shader/pp_fusion_laplace.comp.spv", 0, setLayouts);
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
        }

        virtual void release() override
        {
            pipeCombine.reset();
            pipeTone.reset();

            pipeExposureApply.reset();
            pipeExposureWeight.reset();

            pipeFusionGaussian.reset();
            pipeFusionDownsample.reset();

            pipeFusionLaplace.reset();
            pipeFusionBlend.reset();

            pipeFusion.reset();
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
        }


        PoolImageSharedRef blendFusion = nullptr;
        if (postProcessVolumeSetting.bEnableExposureFusion)
        {
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

                auto buildLaplacePyramid = [&](std::vector<PoolImageSharedRef>& gaussianPyramid)
                {
                    ScopePerframeMarker marker(cmd, "buildLaplacePyramid", { 1.0f, 1.0f, 0.0f, 1.0f });


                    std::vector<PoolImageSharedRef> laplacePyramid{ };

                    for (size_t i = 0; i < gaussianPyramid.size() - 1; i++)
                    {
                        auto& src = gaussianPyramid[i];
                        auto& low = gaussianPyramid[i + 1];

                        uint32_t w = src->getImage().getExtent().width;
                        uint32_t h = src->getImage().getExtent().height;

                        auto upscale = rtPool->createPoolImage("x", w, h, src->getImage().getFormat(), VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
                        upscale->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                        {
                            ScopePerframeMarker marker(cmd, "upscale", { 1.0f, 1.0f, 0.0f, 1.0f });

                            pass->pipeFusionDownsample->bind(cmd);
                            PushSetBuilder(cmd)
                                .addSRV(low)
                                .addUAV(upscale)
                                .push(pass->pipeFusionDownsample.get());
                            pass->pipeFusionDownsample->bindSet(cmd, setSample, 1);

                            vkCmdDispatch(cmd, getGroupCount(upscale->getImage().getExtent().width, 8), getGroupCount(upscale->getImage().getExtent().height, 8), 1);
                        }
                        upscale->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

                        // blur x.
                        auto tempX = rtPool->createPoolImage("d", w, h, src->getImage().getFormat(), VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
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
                        auto tempY = rtPool->createPoolImage("d", w, h, src->getImage().getFormat(), VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
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

                        // Blur ready, then compute laplace.
                        auto laplace = rtPool->createPoolImage("l", w, h, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
                        laplace->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                        {
                            ScopePerframeMarker marker(cmd, "laplace", { 1.0f, 1.0f, 0.0f, 1.0f });

                            pass->pipeFusionLaplace->bind(cmd);
                            PushSetBuilder(cmd)
                                .addSRV(tempY)
                                .addSRV(src)
                                .addUAV(laplace)
                                .push(pass->pipeFusionLaplace.get());
                            pass->pipeFusionLaplace->bindSet(cmd, setSample, 1);

                            vkCmdDispatch(cmd, getGroupCount(w, 8), getGroupCount(h, 8), 1);
                        }
                        laplace->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

                        laplacePyramid.push_back(laplace);
                    }

                    // Last of laplace pyramid just copy.
                    laplacePyramid.push_back(gaussianPyramid.back());

                    return laplacePyramid;
                };

                const int kDepth = 999;
                auto weightPyraimd = buidlGaussianPyramid(weight, kDepth);

                std::vector<PoolImageSharedRef> laplace0;
                std::vector<PoolImageSharedRef> laplace1;
                std::vector<PoolImageSharedRef> laplace2;
                std::vector<PoolImageSharedRef> laplace3;
                {
                    {
                        auto gaussian0 = buidlGaussianPyramid(color0, kDepth);
                        laplace0 = buildLaplacePyramid(gaussian0);
                    }
                    {
                        auto gaussian1 = buidlGaussianPyramid(color1, kDepth);
                        laplace1 = buildLaplacePyramid(gaussian1);
                    }
                    {
                        auto gaussian2 = buidlGaussianPyramid(color2, kDepth);
                        laplace2 = buildLaplacePyramid(gaussian2);
                    }
                    {
                        auto gaussian3 = buidlGaussianPyramid(color3, kDepth);
                        laplace3 = buildLaplacePyramid(gaussian3);
                    }
                }

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