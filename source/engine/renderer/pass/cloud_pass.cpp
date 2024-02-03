#include "../deferred_renderer.h"
#include "../render_scene.h"
#include "../renderer.h"
#include "../scene_textures.h"

namespace engine
{
    struct CloudPush
    {
        uint sdsmShadowDepthIndices[kMaxCascadeNum];
        uint cascadeCount;

        float kStepNum;
        float mixWeight;
    };

    static AutoCVarInt32 cVarCloudDownSampleRender("r.cloud.downsample", "Downsample cloud to save performance.", "Render", 1, CVarFlags::ReadAndWrite);

    struct CloudBlurShadowDepthPush
    {
        vec2  kBlurDirection;
        float kRadius;
        float kSigma;
        float kRadiusLow;
    };

    struct LensPush
    {
        uint32_t bCloud;
        uint32_t bFog;
    };

    static AutoCVarInt32 cVarFogRayStep("r.fog.rayStep", "Ray marching step.", "Render", 16, CVarFlags::ReadAndWrite);
    struct FogPush
    {
        uint sdsmShadowDepthIndices[kMaxCascadeNum];
        uint cascadeCount;
        uint kGodRaySteps;
        uint kSkyPass;
    };

    static AutoCVarInt32 cVarCloudBlurRadius("r.cloud.shadowdepth.blurRadius", "cloud shadow depth blur radius.", "Render", 
        8, CVarFlags::ReadAndWrite);

    static AutoCVarInt32 cVarCloudBlurRadiusLow("r.cloud.shadowdepth.blurLow", "cloud shadow depth blur radius low radius.", "Render",
        2, CVarFlags::ReadAndWrite);

    static AutoCVarInt32 cVarCloudBlurRadiusLow2("r.cloud.shadowdepth.blurLow2", "cloud shadow depth blur radius low radius.", "Render",
        6, CVarFlags::ReadAndWrite);

    static AutoCVarFloat cVarCloudBlurMix("r.cloud.shadowdepth.mix", "cloud shadow depth blur mix.", "Render",
        0.125f, CVarFlags::ReadAndWrite);

    static AutoCVarFloat cVarCloudBlurMixLow("r.cloud.shadowdepth.mixLow", "cloud shadow depth blur mix low.", "Render",
        0.25f, CVarFlags::ReadAndWrite);

    struct CloudDepthMixPush
    {
        vec2 mixWeight;
    };

    class CloudPass : public PassInterface
    {
    public:
        ComputePipeResourcesRef fogPipe;
        ComputePipeResourcesRef lensPipe;
        ComputePipeResourcesRef computeCloudPipeline;
        ComputePipeResourcesRef reconstructionPipeline;
        ComputePipeResourcesRef compositeCloudPipeline;
        ComputePipeResourcesRef cloudDepthMixPipeline;
        ComputePipeResourcesRef cloudDepthPipeline;
        ComputePipeResourcesRef cloudBlurDepthPipeline;

    public:
        virtual void onInit() override
        {
            {
                VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;

                getContext()->descriptorFactoryBegin()
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 0) // inHdrSceneColor
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1) // inHdrSceneColor
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2) // imageHdrSceneColor
                    .buildNoInfoPush(setLayout);

                std::vector<VkDescriptorSetLayout> setLayouts =
                {
                      setLayout
                    , m_context->getSamplerCache().getCommonDescriptorSetLayout()
                };

                cloudDepthMixPipeline = createComputePipe("shader/cloud_shadow_depth_mix.glsl",
                    sizeof(CloudDepthMixPush), setLayouts);
            }

            {
                VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;

                getContext()->descriptorFactoryBegin()
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 0) // imageHdrSceneColor
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1) // inHdrSceneColor
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 2) // inDepth
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 3) // inGBufferA
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 4) // inBasicNoise
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,5)
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 6) // inCloudReconstructionTexture
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,7) // Framedata.
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 8) // inCloudReconstructionTexture
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 9) // inDepth
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 10) // imageCloudReconstructionTexture
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 11) // inDepth
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 12) // inDepth
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 13) // inDepth
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 14) // inDepth
                    .buildNoInfoPush(setLayout);

                std::vector<VkDescriptorSetLayout> setLayouts =
                {
                      setLayout
                    , m_context->getSamplerCache().getCommonDescriptorSetLayout()
                    , getRenderer()->getBlueNoise().spp_1_buffer.setLayouts
                    , m_context->getBindlessTextureSetLayout()
                };

                ShaderVariant shaderVariant("shader/volumetric_fog_raymarching.glsl");
                shaderVariant.setStage(EShaderStage::eComputeShader);

                {
                    auto copyVariant = shaderVariant;
                    copyVariant.setMacro(L"COMPUTE_PASS");
                    fogPipe = createComputePipe(copyVariant, sizeof(FogPush), setLayouts);
                }

            }

            {
                VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;

                getContext()->descriptorFactoryBegin()
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0)
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1)
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 2)
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 3)
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 4)
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 5)
                    .buildNoInfoPush(setLayout);

                std::vector<VkDescriptorSetLayout> setLayoutsLens =
                {
                      setLayout
                    , m_context->getSamplerCache().getCommonDescriptorSetLayout()
                };

                lensPipe = createComputePipe("shader/lens_visible.glsl", sizeof(LensPush), setLayoutsLens);
            }

            {
                VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
                // Config code.
                getContext()->descriptorFactoryBegin()
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 0) // imageHdrSceneColor
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1) // inHdrSceneColor
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2) // imageCloudRenderTexture
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 3) // inCloudRenderTexture
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 4) // inDepth
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 5) // inGBufferA
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 6) // inBasicNoise
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 7) // inDetailNoise
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 8) // inCloudWeather
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 9) // inCloudCurl
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 10) // inTransmittanceLut
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 11) // inFroxelScatter
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 12) // imageCloudReconstructionTexture
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 13) // inCloudReconstructionTexture
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 14) // imageCloudReconstructionTexture
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 15) // inCloudReconstructionTexture
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 16) // imageCloudReconstructionTexture
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 17) // inCloudReconstructionTexture
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 18) // inCloudReconstructionTexture
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 19) // inCloudReconstructionTexture
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 20) // inDepth
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 21) // Framedata.
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 22) // inSkylight
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 23)
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 24) // imageCloudReconstructionTexture
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 25)
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 26) 
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 27) // imageCloudReconstructionTexture
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 28) // imageCloudReconstructionTexture
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 29) // imageCloudReconstructionTexture
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 30) // imageCloudReconstructionTexture
                    .buildNoInfoPush(setLayout);


                std::vector<VkDescriptorSetLayout> setLayouts =
                {
                      setLayout
                    , m_context->getSamplerCache().getCommonDescriptorSetLayout()
                    , getRenderer()->getBlueNoise().spp_1_buffer.setLayouts
                    , m_context->getBindlessTextureSetLayout()
                };

                computeCloudPipeline = createComputePipe("shader/cloud_render_raymarching.glsl", sizeof(CloudPush), setLayouts);
                reconstructionPipeline = createComputePipe("shader/cloud_render_reconstruct.glsl", sizeof(CloudPush), setLayouts);
                compositeCloudPipeline = createComputePipe("shader/cloud_render_composite.glsl", sizeof(CloudPush), setLayouts);
                cloudDepthPipeline = createComputePipe("shader/cloud_shadow_depth.glsl", sizeof(CloudPush), setLayouts);
            }


            {
                VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
                // Config code.
                getContext()->descriptorFactoryBegin()
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 0) // imageHdrSceneColor
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1) // inHdrSceneColor
                    .buildNoInfoPush(setLayout);

                std::vector<VkDescriptorSetLayout> setLayouts =
                {
                      setLayout
                    , m_context->getSamplerCache().getCommonDescriptorSetLayout()
                };

                cloudBlurDepthPipeline = createComputePipe("shader/cloud_shadow_depth_blur.glsl",
                    sizeof(CloudBlurShadowDepthPush), setLayouts);
            }
        }
    };


    void engine::updateCloudPass()
    {
        getContext()->getPasses().updatePass<class CloudPass>();
        CVarSystem::get()->setCVar("cmd.clearAllReflectionCapture", true);
    }

    bool shouldRenderCloud(const PerFrameData& perframe)
    {
        return (perframe.bSkyComponentValid != 0);
    }

    void DeferredRenderer::renderVolumetricCloudShadowDepth(
        VkCommandBuffer cmd,
        GBufferTextures* inGBuffers,
        RenderScene* scene,
        BufferParameterHandle perFrameGPU,
        AtmosphereTextures& inAtmosphere,
        const PerFrameData& perframe,
        const SkyLightRenderContext& skyContext)
    {
        if (!shouldRenderCloud(perframe))
        {
            m_history.cloudShadowDepthHistory = nullptr;
            return;
        }

        if (m_history.cloudShadowDepthHistory == nullptr ||
            perframe.frameIndex.x % 4 == 0)
        {
            // Update.
        }
        else
        {
            return;
        }

        auto* pass = getContext()->getPasses().get<CloudPass>();
        auto* rtPool = &getContext()->getRenderTargetPools();

        auto sunCloudShadowDepth = rtPool->createPoolImage(
            "sun cloud shadow depth",
            getShadowDepthDimCloud(),
            getShadowDepthDimCloud(),
            VK_FORMAT_R32G32_SFLOAT,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

        auto& sceneDepthZ = inGBuffers->depthTexture->getImage();
        auto& sceneColorHdr = inGBuffers->hdrSceneColor->getImage();
        auto& gbufferA = inGBuffers->gbufferA->getImage();

        auto weatherTexture = std::dynamic_pointer_cast<GPUImageAsset>(getContext()->getBuiltinTexture(EBuiltinTextures::cloudWeather));
        auto curlNoiseTexture = std::dynamic_pointer_cast<GPUImageAsset>(getContext()->getBuiltinTexture(EBuiltinTextures::cloudNoise));

        auto cloudDistanceLit = inAtmosphere.distant;

        auto pushCommonSet = [&]()
        {
            auto& placeHolder = sceneColorHdr;

            PushSetBuilder setBuilder(cmd);
            setBuilder
                .addUAV(placeHolder)
                .addSRV(placeHolder)
                .addUAV(placeHolder)
                .addSRV(placeHolder)
                .addSRV(sceneDepthZ, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT))
                .addSRV(gbufferA)
                .addSRV(*getRenderer()->getSharedTextures().cloudBasicNoise, buildBasicImageSubresource(), VK_IMAGE_VIEW_TYPE_3D)
                .addSRV(*getRenderer()->getSharedTextures().cloudDetailNoise, buildBasicImageSubresource(), VK_IMAGE_VIEW_TYPE_3D)
                .addSRV(weatherTexture->getSelfImage()) // 8
                .addSRV(curlNoiseTexture->getSelfImage()) // 8
                .addSRV(inAtmosphere.transmittance) // 10
                .addSRV(inAtmosphere.distantGrid, buildBasicImageSubresource(), VK_IMAGE_VIEW_TYPE_3D) // 11
                .addUAV(placeHolder) // 12
                .addSRV(placeHolder) // 13
                .addUAV(placeHolder) // 14
                .addSRV(placeHolder) // 15
                .addUAV(placeHolder) // 16
                .addSRV(placeHolder) // 17
                .addSRV(placeHolder) // 18
                .addSRV(placeHolder) // 19
                .addSRV(inAtmosphere.skyView) // 20
                .addBuffer(perFrameGPU) // 21
                .addSRV(skyContext.skylightRadiance, buildBasicImageSubresourceCube(), VK_IMAGE_VIEW_TYPE_CUBE)
                .addSRV(cloudDistanceLit)
                .addUAV(sunCloudShadowDepth)
                .addSRV(m_history.cloudShadowDepthHistory ? m_history.cloudShadowDepthHistory->getImage() : getContext()->getBuiltinTextureTranslucent()->getSelfImage())
                .push(pass->cloudDepthPipeline);

            std::vector<VkDescriptorSet> additionalSets =
            {
                getContext()->getSamplerCache().getCommonDescriptorSet(),
                getRenderer()->getBlueNoise().spp_1_buffer.set,
                getContext()->getBindlessTexture().getSet()
            };
            pass->cloudDepthPipeline->bindSet(cmd, additionalSets, 1);
        };

        cloudDistanceLit->getImage().transitionShaderReadOnly(cmd);
        pushCommonSet();
        {
            sunCloudShadowDepth->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

            CloudPush push;
            push.kStepNum  = 25;

            pass->cloudDepthPipeline->bindAndPushConst(cmd, &push);
            vkCmdDispatch(cmd,
                getGroupCount(sunCloudShadowDepth->getImage().getExtent().width, 8),
                getGroupCount(sunCloudShadowDepth->getImage().getExtent().height, 8), 1);

            sunCloudShadowDepth->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        }

        // Blur x.
        auto sunCloudShadowDepthBlurX = rtPool->createPoolImage(
            "sun cloud shadow depth blurX",
            sunCloudShadowDepth->getImage().getExtent().width,
            sunCloudShadowDepth->getImage().getExtent().height,
            sunCloudShadowDepth->getImage().getFormat(),
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

        {


            {
                PushSetBuilder setBuilder(cmd);
                setBuilder
                    .addSRV(sunCloudShadowDepth)
                    .addUAV(sunCloudShadowDepthBlurX)
                    .push(pass->cloudBlurDepthPipeline);

                std::vector<VkDescriptorSet> additionalSets =
                {
                    getContext()->getSamplerCache().getCommonDescriptorSet(),
                };
                pass->cloudBlurDepthPipeline->bindSet(cmd, additionalSets, 1);
            }

            sunCloudShadowDepthBlurX->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

            CloudBlurShadowDepthPush push{};
            push.kBlurDirection = vec2(1.0f, 0.0f);
            push.kRadius = float(cVarCloudBlurRadius.get());
            push.kSigma  = 0.2f;
            push.kRadiusLow = math::mix(float(cVarCloudBlurRadiusLow.get()), float(cVarCloudBlurRadiusLow2.get()), 
                1.0f - math::clamp(perframe.sunLightInfo.direction.y * 2.0f, 0.0f, 1.0f));

            pass->cloudBlurDepthPipeline->bindAndPushConst(cmd, &push);
            vkCmdDispatch(cmd,
                getGroupCount(sunCloudShadowDepthBlurX->getImage().getExtent().width, 8),
                getGroupCount(sunCloudShadowDepthBlurX->getImage().getExtent().height, 8), 1);

            sunCloudShadowDepthBlurX->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

            // Blur Y.
            sunCloudShadowDepth->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

            {

                PushSetBuilder setBuilder2(cmd);
                setBuilder2
                    .addSRV(sunCloudShadowDepthBlurX)
                    .addUAV(sunCloudShadowDepth)
                    .push(pass->cloudBlurDepthPipeline);
            }


            push.kBlurDirection = vec2(0.0f, 1.0f);
            pass->cloudBlurDepthPipeline->bindAndPushConst(cmd, &push);
            vkCmdDispatch(cmd,
                getGroupCount(sunCloudShadowDepth->getImage().getExtent().width, 8),
                getGroupCount(sunCloudShadowDepth->getImage().getExtent().height, 8), 1);

            sunCloudShadowDepth->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        }

        if (m_history.cloudShadowDepthHistory != nullptr)
        {
            sunCloudShadowDepthBlurX->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
            m_history.cloudShadowDepthHistory->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

            CloudDepthMixPush push{};
            push.mixWeight = vec2(cVarCloudBlurMix.get(), cVarCloudBlurMixLow.get());
            pass->cloudDepthMixPipeline->bindAndPushConst(cmd, &push);

            PushSetBuilder setBuilder(cmd);
            setBuilder
                .addSRV(sunCloudShadowDepth)
                .addSRV(m_history.cloudShadowDepthHistory)
                .addUAV(sunCloudShadowDepthBlurX)
                .push(pass->cloudDepthMixPipeline);

            std::vector<VkDescriptorSet> additionalSets =
            {
                getContext()->getSamplerCache().getCommonDescriptorSet(),
            };
            pass->cloudDepthMixPipeline->bindSet(cmd, additionalSets, 1);

            vkCmdDispatch(cmd,
                getGroupCount(sunCloudShadowDepth->getImage().getExtent().width, 8),
                getGroupCount(sunCloudShadowDepth->getImage().getExtent().height, 8), 1);

            sunCloudShadowDepthBlurX->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

            m_history.cloudShadowDepthHistory = sunCloudShadowDepthBlurX;
        }
        else
        {
            m_history.cloudShadowDepthHistory = sunCloudShadowDepth;
        }
    }

    void DeferredRenderer::renderVolumetricCloud(
        VkCommandBuffer cmd, 
        GBufferTextures* inGBuffers, 
        RenderScene* scene, 
        BufferParameterHandle perFrameGPU, 
        AtmosphereTextures& inAtmosphere, 
        const PerFrameData& perframe,
        const SkyLightRenderContext& skyContext,
        const SDSMInfos& sunSDSMInfos)
    {

        auto& sceneDepthZ = inGBuffers->depthTexture->getImage();
        auto& sceneColorHdr = inGBuffers->hdrSceneColor->getImage();
        auto& gbufferA = inGBuffers->gbufferA->getImage();

        bool bExistCloud = false;
        auto* pass = getContext()->getPasses().get<CloudPass>();
        auto* rtPool = &getContext()->getRenderTargetPools();

        if (scene->getSkyComponent() == nullptr)
        {
            return;
        }

        const bool bDownSample = cVarCloudDownSampleRender.get() != 0;
        int kDownSampleSize = bDownSample ? 4 : 1;
        auto cloudDistanceLit = inAtmosphere.distant;

        cloudDistanceLit->getImage().transitionShaderReadOnly(cmd);

        PoolImageSharedRef fogFullImage;
        {
            fogFullImage = rtPool->createPoolImage(
                "fogSkyCompute-Full",
                sceneDepthZ.getExtent().width,
                sceneDepthZ.getExtent().height,
                VK_FORMAT_R16G16B16A16_SFLOAT,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

            PoolImageSharedRef fogSkyImage;
            {
                fogSkyImage = rtPool->createPoolImage(
                    "fogSkyCompute",
                    sceneDepthZ.getExtent().width / 4,
                    sceneDepthZ.getExtent().height / 4,
                    VK_FORMAT_R16G16B16A16_SFLOAT,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

                PushSetBuilder setBuilder(cmd);
                setBuilder
                    .addUAV(sceneColorHdr)
                    .addSRV(sceneColorHdr)
                    .addSRV(inGBuffers->chessboardHalfDepth)
                    .addSRV(inAtmosphere.transmittance) // 10
                    .addSRV(inAtmosphere.distantGrid, buildBasicImageSubresource(), VK_IMAGE_VIEW_TYPE_3D) // 11
                    .addBuffer(sunSDSMInfos.cascadeInfoBuffer)
                    .addSRV(m_history.cloudShadowDepthHistory)
                    .addBuffer(perFrameGPU) // 21
                    .addSRV(skyContext.skylightRadiance, buildBasicImageSubresourceCube(), VK_IMAGE_VIEW_TYPE_CUBE)
                    .addSRV(cloudDistanceLit)
                    .addUAV(fogSkyImage)
                    .addSRV(inGBuffers->hzbClosest)
                    .addSRV(cloudDistanceLit) // inFog
                    .addSRV(cloudDistanceLit) // inFogSky
                    .addSRV(inGBuffers->hzbFurthest)
                    .push(pass->fogPipe);

                std::vector<VkDescriptorSet> additionalSets =
                {
                    getContext()->getSamplerCache().getCommonDescriptorSet(),
                    getRenderer()->getBlueNoise().spp_1_buffer.set,
                    getContext()->getBindlessTexture().getSet()
                };
                pass->fogPipe->bindSet(cmd, additionalSets, 1);

                FogPush push{};
                push.kSkyPass = true;
                push.kGodRaySteps = cVarFogRayStep.get();
                push.cascadeCount = perframe.sunLightInfo.cascadeConfig.cascadeCount;
                for (int i = sunSDSMInfos.shadowDepths.size() - 1; i >= 0; i--)
                {
                    push.sdsmShadowDepthIndices[i] =
                        sunSDSMInfos.shadowDepths[i]->getImage().getOrCreateView(
                            RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT)).srvBindless;
                }

                {
                    fogSkyImage->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                    pass->fogPipe->bindAndPushConst(cmd, &push);
                    vkCmdDispatch(cmd,
                        getGroupCount(fogSkyImage->getImage().getExtent().width, 8),
                        getGroupCount(fogSkyImage->getImage().getExtent().height, 8), 1);
                    fogSkyImage->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
                }
            }

            PoolImageSharedRef fogImage;
            {
                fogImage = rtPool->createPoolImage(
                    "fogCompute",
                    sceneDepthZ.getExtent().width / 2,
                    sceneDepthZ.getExtent().height / 2,
                    VK_FORMAT_R16G16B16A16_SFLOAT,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

                PushSetBuilder setBuilder(cmd);
                setBuilder
                    .addUAV(sceneColorHdr)
                    .addSRV(sceneColorHdr)
                    .addSRV(inGBuffers->chessboardHalfDepth)
                    .addSRV(inAtmosphere.transmittance) // 10
                    .addSRV(inAtmosphere.distantGrid, buildBasicImageSubresource(), VK_IMAGE_VIEW_TYPE_3D) // 11
                    .addBuffer(sunSDSMInfos.cascadeInfoBuffer)
                    .addSRV(m_history.cloudShadowDepthHistory)
                    .addBuffer(perFrameGPU) // 21
                    .addSRV(skyContext.skylightRadiance, buildBasicImageSubresourceCube(), VK_IMAGE_VIEW_TYPE_CUBE)
                    .addSRV(cloudDistanceLit)
                    .addUAV(fogImage)
                    .addSRV(inGBuffers->hzbClosest)
                    .addSRV(cloudDistanceLit) // inFog
                    .addSRV(cloudDistanceLit) // inFogSky
                    .addSRV(inGBuffers->hzbFurthest)
                    .push(pass->fogPipe);

                std::vector<VkDescriptorSet> additionalSets =
                {
                    getContext()->getSamplerCache().getCommonDescriptorSet(),
                    getRenderer()->getBlueNoise().spp_1_buffer.set,
                    getContext()->getBindlessTexture().getSet()
                };
                pass->fogPipe->bindSet(cmd, additionalSets, 1);

                FogPush push{};
                push.kSkyPass = false;
                push.kGodRaySteps = cVarFogRayStep.get();
                push.cascadeCount = perframe.sunLightInfo.cascadeConfig.cascadeCount;
                for (int i = sunSDSMInfos.shadowDepths.size() - 1; i >= 0; i--)
                {
                    push.sdsmShadowDepthIndices[i] =
                        sunSDSMInfos.shadowDepths[i]->getImage().getOrCreateView(
                            RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT)).srvBindless;
                }

                {
                    fogImage->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                    pass->fogPipe->bindAndPushConst(cmd, &push);
                    vkCmdDispatch(cmd,
                        getGroupCount(fogImage->getImage().getExtent().width, 8),
                        getGroupCount(fogImage->getImage().getExtent().height, 8), 1);
                    fogImage->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
                }


            }


            {
                PushSetBuilder setBuilder(cmd);
                setBuilder
                    .addUAV(sceneColorHdr)
                    .addSRV(sceneColorHdr)
                    .addSRV(sceneDepthZ, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT))
                    .addSRV(inAtmosphere.transmittance) // 10
                    .addSRV(inAtmosphere.distantGrid, buildBasicImageSubresource(), VK_IMAGE_VIEW_TYPE_3D) // 11
                    .addBuffer(sunSDSMInfos.cascadeInfoBuffer)
                    .addSRV(m_history.cloudShadowDepthHistory)
                    .addBuffer(perFrameGPU) // 21
                    .addSRV(skyContext.skylightRadiance, buildBasicImageSubresourceCube(), VK_IMAGE_VIEW_TYPE_CUBE)
                    .addSRV(cloudDistanceLit)
                    .addUAV(fogFullImage)
                    .addSRV(inGBuffers->hzbClosest)
                    .addSRV(fogImage) // inFog
                    .addSRV(fogSkyImage) // inFogSky
                    .addSRV(inGBuffers->hzbFurthest)
                    .push(pass->fogPipe);

                std::vector<VkDescriptorSet> additionalSets =
                {
                    getContext()->getSamplerCache().getCommonDescriptorSet(),
                    getRenderer()->getBlueNoise().spp_1_buffer.set,
                    getContext()->getBindlessTexture().getSet()
                };
                pass->fogPipe->bindSet(cmd, additionalSets, 1);

                FogPush push{};
                push.kSkyPass = 2;
                push.kGodRaySteps = cVarFogRayStep.get();
                push.cascadeCount = perframe.sunLightInfo.cascadeConfig.cascadeCount;
                for (int i = sunSDSMInfos.shadowDepths.size() - 1; i >= 0; i--)
                {
                    push.sdsmShadowDepthIndices[i] =
                        sunSDSMInfos.shadowDepths[i]->getImage().getOrCreateView(
                            RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT)).srvBindless;
                }

                {
                    fogFullImage->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL);
                    pass->fogPipe->bindAndPushConst(cmd, &push);
                    vkCmdDispatch(cmd,
                        getGroupCount(fogFullImage->getImage().getExtent().width, 8),
                        getGroupCount(fogFullImage->getImage().getExtent().height, 8), 1);
                    fogFullImage->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                }
            }


            m_gpuTimer.getTimeStamp(cmd, "Volumetric fog");
        }



        auto computeCloud = [&]()
        {
            if (!shouldRenderCloud(m_perframe))
            {
                m_history.cloudReconstruction = nullptr;
                m_history.cloudReconstructionDepth = nullptr;
                return;
            }

            bExistCloud = true;

            // Quater resolution evaluate.
            auto computeCloud = rtPool->createPoolImage(
                "CloudCompute",
                sceneDepthZ.getExtent().width / kDownSampleSize,
                sceneDepthZ.getExtent().height / kDownSampleSize,
                VK_FORMAT_R16G16B16A16_SFLOAT,
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
            );
            auto computeCloudDepth = rtPool->createPoolImage(
                "CloudComputeDepth",
                sceneDepthZ.getExtent().width / kDownSampleSize,
                sceneDepthZ.getExtent().height / kDownSampleSize,
                VK_FORMAT_R32_SFLOAT,
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
            );

            inAtmosphere.transmittance->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            inAtmosphere.skyView->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            inAtmosphere.froxelScatter->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());



            auto newCloudReconstruction = computeCloud;
            auto newCloudReconstructionDepth = computeCloudDepth;

            if (bDownSample)
            {
                newCloudReconstruction = rtPool->createPoolImage(
                    "NewCloudReconstruction",
                    sceneDepthZ.getExtent().width,
                    sceneDepthZ.getExtent().height,
                    VK_FORMAT_R16G16B16A16_SFLOAT,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

                newCloudReconstructionDepth = rtPool->createPoolImage(
                    "NewCloudReconstructionDepth",
                    sceneDepthZ.getExtent().width,
                    sceneDepthZ.getExtent().height,
                    VK_FORMAT_R32_SFLOAT,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
            }


            VkClearColorValue zeroClear =
            {
                .uint32 = {0,0,0,0}
            };

            auto rangeClear = buildBasicImageSubresource();
            if (!m_history.cloudReconstruction)
            {
                m_history.cloudReconstruction = rtPool->createPoolImage(
                    "CloudReconstruction",
                    sceneDepthZ.getExtent().width,
                    sceneDepthZ.getExtent().height,
                    VK_FORMAT_R16G16B16A16_SFLOAT,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
                m_history.cloudReconstruction->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());


                m_bCameraCut = true;

                vkCmdClearColorImage(cmd, m_history.cloudReconstruction->getImage().getImage(), 
                    VK_IMAGE_LAYOUT_GENERAL, &zeroClear, 1, &rangeClear);

                m_history.cloudReconstruction->getImage().
                    transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

            }
            if (!m_history.cloudReconstructionDepth)
            {
                m_history.cloudReconstructionDepth = rtPool->createPoolImage(
                    "CloudReconstructionDepth",
                    sceneDepthZ.getExtent().width,
                    sceneDepthZ.getExtent().height,
                    VK_FORMAT_R32_SFLOAT,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
                m_history.cloudReconstructionDepth->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                m_bCameraCut = true;


                vkCmdClearColorImage(cmd, m_history.cloudReconstructionDepth->getImage().getImage(), 
                    VK_IMAGE_LAYOUT_GENERAL, &zeroClear, 1, &rangeClear);

                m_history.cloudReconstructionDepth->getImage().
                    transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            }

            auto weatherTexture = std::dynamic_pointer_cast<GPUImageAsset>(getContext()->getBuiltinTexture(EBuiltinTextures::cloudWeather));
            auto curlNoiseTexture = std::dynamic_pointer_cast<GPUImageAsset>(getContext()->getBuiltinTexture(EBuiltinTextures::cloudNoise));
            auto curlTexture = std::dynamic_pointer_cast<GPUImageAsset>(getContext()->getBuiltinTexture(EBuiltinTextures::curlNoise));


            CloudPush push{};
            push.cascadeCount = perframe.sunLightInfo.cascadeConfig.cascadeCount;

            for (int i = sunSDSMInfos.shadowDepths.size() - 1; i >= 0; i--)
            {
                push.sdsmShadowDepthIndices[i] =
                    sunSDSMInfos.shadowDepths[i]->getImage().getOrCreateView(
                        RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT)).srvBindless;
            }


            auto pushCommonSet = [&]()
            {
                PushSetBuilder setBuilder(cmd);
                setBuilder
                    .addUAV(sceneColorHdr)
                    .addSRV(sceneColorHdr)
                    .addUAV(computeCloud)
                    .addSRV(computeCloud)
                    .addSRV(inGBuffers->hzbFurthest)
                    .addSRV(curlTexture->getSelfImage())
                    .addSRV(*getRenderer()->getSharedTextures().cloudBasicNoise, buildBasicImageSubresource(), VK_IMAGE_VIEW_TYPE_3D)
                    .addSRV(*getRenderer()->getSharedTextures().cloudDetailNoise, buildBasicImageSubresource(), VK_IMAGE_VIEW_TYPE_3D)
                    .addSRV(weatherTexture->getSelfImage()) // 8
                    .addSRV(curlNoiseTexture->getSelfImage()) // 8
                    .addSRV(inAtmosphere.transmittance) // 10
                    .addSRV(inAtmosphere.froxelScatter, buildBasicImageSubresource(), VK_IMAGE_VIEW_TYPE_3D) // 11
                    .addUAV(newCloudReconstruction) // 12
                    .addSRV(newCloudReconstruction) // 13
                    .addUAV(computeCloudDepth) // 14
                    .addSRV(computeCloudDepth) // 15
                    .addUAV(newCloudReconstructionDepth) // 16
                    .addSRV(newCloudReconstructionDepth) // 17
                    .addSRV(m_history.cloudReconstruction) // 18
                    .addSRV(m_history.cloudReconstructionDepth) // 19
                    .addSRV(inAtmosphere.distantGrid, buildBasicImageSubresource(), VK_IMAGE_VIEW_TYPE_3D) // 20
                    .addBuffer(perFrameGPU) // 21
                    .addSRV(skyContext.skylightRadiance, buildBasicImageSubresourceCube(), VK_IMAGE_VIEW_TYPE_CUBE)
                    .addSRV(cloudDistanceLit)
                    .addUAV(m_history.cloudShadowDepthHistory)
                    .addSRV(m_history.cloudShadowDepthHistory)
                    .addBuffer(sunSDSMInfos.cascadeInfoBuffer)
                    .addSRV(fogFullImage)
                    .addSRV(fogFullImage)
                    .addSRV(m_history.prevHZBClosest ? m_history.prevHZBClosest : inGBuffers->hzbClosest)
                    .addSRV(inGBuffers->hzbClosest)
                    .push(pass->computeCloudPipeline);

                std::vector<VkDescriptorSet> additionalSets =
                {
                    getContext()->getSamplerCache().getCommonDescriptorSet(),
                    getRenderer()->getBlueNoise().spp_1_buffer.set,
                    getContext()->getBindlessTexture().getSet()
                };
                pass->computeCloudPipeline->bindSet(cmd, additionalSets, 1);
            };

            pass->computeCloudPipeline->pushConst(cmd, &push);

            cloudDistanceLit->getImage().transitionShaderReadOnly(cmd);

            pushCommonSet();
            {
                computeCloud->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                computeCloudDepth->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

                pass->computeCloudPipeline->bind(cmd);
                vkCmdDispatch(cmd,
                    getGroupCount(computeCloud->getImage().getExtent().width, 8),
                    getGroupCount(computeCloud->getImage().getExtent().height, 8), 1);

                computeCloud->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
                computeCloudDepth->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            }

            if(bDownSample)
            {
                newCloudReconstruction->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

                newCloudReconstructionDepth->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
                pass->reconstructionPipeline->bind(cmd);

                vkCmdDispatch(cmd,
                    getGroupCount(newCloudReconstruction->getImage().getExtent().width, 8),
                    getGroupCount(newCloudReconstruction->getImage().getExtent().height, 8), 1);

                newCloudReconstruction->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
                newCloudReconstructionDepth->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            }

            {
                sceneColorHdr.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

                pass->compositeCloudPipeline->bind(cmd);
                vkCmdDispatch(cmd, getGroupCount(sceneColorHdr.getExtent().width, 8), getGroupCount(sceneColorHdr.getExtent().height, 8), 1);
                sceneColorHdr.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

            }

            m_history.cloudReconstruction = newCloudReconstruction;
            m_history.cloudReconstructionDepth = newCloudReconstructionDepth;
        };

        computeCloud();

        auto lensSSBO = getContext()->getBufferParameters().getStaticStorage("SSBOLensBuffer", sizeof(float) * 4);
        {
            std::array<VkBufferMemoryBarrier2, 1> beginBufferBarriers
            {
                RHIBufferBarrier(lensSSBO->getBuffer()->getVkBuffer(),
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT),
            };
            RHIPipelineBarrier(cmd, 0, (uint32_t)beginBufferBarriers.size(), beginBufferBarriers.data(), 0, nullptr);

            PushSetBuilder setBuilder(cmd);
            setBuilder
                .addBuffer(lensSSBO)
                .addBuffer(perFrameGPU)
                .addSRV(sceneDepthZ, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT))
                .addSRV(fogFullImage)
                .addSRV(m_history.cloudReconstruction ? m_history.cloudReconstruction->getImage() : gbufferA)
                .addSRV(inAtmosphere.transmittance)
                .push(pass->lensPipe);

            std::vector<VkDescriptorSet> additionalSets =
            {
                getContext()->getSamplerCache().getCommonDescriptorSet(),
            };
            pass->lensPipe->bindSet(cmd, additionalSets, 1);


            LensPush push{
                .bCloud = bExistCloud ? 1U : 0U,
                .bFog   = 1U,
            };
            pass->lensPipe->bindAndPushConst(cmd, &push);

            vkCmdDispatch(cmd, 1, 1, 1);

            std::array<VkBufferMemoryBarrier2, 1> endBufferBarriers
            {
                RHIBufferBarrier(lensSSBO->getBuffer()->getVkBuffer(),
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT),
            };
            RHIPipelineBarrier(cmd, 0, (uint32_t)endBufferBarriers.size(), endBufferBarriers.data(), 0, nullptr);

            inGBuffers->lensBuffer = lensSSBO;
        }


        m_gpuTimer.getTimeStamp(cmd, "Volumetric Cloud");
    }
}