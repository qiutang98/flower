#include "../deferred_renderer.h"
#include "../render_scene.h"
#include "../renderer.h"
#include "../scene_textures.h"

namespace engine
{

    struct SSGIPush
    {
        vec3  probe0Pos;
        float probe0ValidFactor;
        vec3  probe1Pos;
        float probe1ValidFactor;

        vec4  boxExtentData0;
        vec4  boxExtentData1;
        vec4  boxExtentData2;
    };

    struct RTGIPush
    {
        float rayMinRange;
		float rayMaxRange;
    };

    struct SSGIReprojectPush
    {
        float kMinAlpha;
        float kMinAlphaMoment;
    };

    struct SSGIBlurPush
    {
        int kStepSize;
    };

    class SSGIPass : public PassInterface
    {
    public:
        std::unique_ptr<ComputePipeResources> intersect;
        std::unique_ptr<ComputePipeResources> filter;
        std::unique_ptr<ComputePipeResources> reproject;

        std::unique_ptr<ComputePipeResources> rtPipe;

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
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 4) // 
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 5) // 
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 6) // 
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 7) // 
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 8) // 
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 9) // 
                    .buildNoInfoPush(intersectLayout);

                std::vector<VkDescriptorSetLayout> intersectLayouts =
                {
                    intersectLayout,
                    getContext()->getSamplerCache().getCommonDescriptorSetLayout(),
                    getRenderer()->getBlueNoise().spp_1_buffer.setLayouts
                };

                intersect = std::make_unique<ComputePipeResources>("shader/ssgi_intersect.glsl", sizeof(SSGIPush), intersectLayouts);
            }
            {
                VkDescriptorSetLayout layout = VK_NULL_HANDLE;

                getContext()->descriptorFactoryBegin()
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 0) // 
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1) // 
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 2) // 
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3) // 
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,4) // 
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 5) // 
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 6) // 
                    .buildNoInfoPush(layout);

                std::vector<VkDescriptorSetLayout> layouts =
                {
                    layout,
                    getContext()->getSamplerCache().getCommonDescriptorSetLayout(),
                };

                filter = std::make_unique<ComputePipeResources>("shader/ssgi_blur.glsl", sizeof(SSGIBlurPush), layouts);
            }

            {
                VkDescriptorSetLayout layout = VK_NULL_HANDLE;

                getContext()->descriptorFactoryBegin()
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 0) // 
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1) // 
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,2) // 
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 3) // 
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 4) // 
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 5) // 
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 6) // 
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 7) // 
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 8) // 
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 9) // 
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,10) // 
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,11) // 
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,12) // 
                    .buildNoInfoPush(layout);

                std::vector<VkDescriptorSetLayout> layouts =
                {
                    layout,
                    getContext()->getSamplerCache().getCommonDescriptorSetLayout()
                };

                reproject = std::make_unique<ComputePipeResources>("shader/ssgi_reproject.glsl", sizeof(SSGIReprojectPush), layouts);
            }

			{
				VkDescriptorSetLayout rtSetLayout = VK_NULL_HANDLE;
				getContext()->descriptorFactoryBegin()
					.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 0) // 
					.bindNoInfo(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1) // inFrameData
					.bindNoInfo(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 2) // AS
					.bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 3) // inDepth
					.bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4)
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 5) // inDepth
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 6) // inDepth
					.buildNoInfoPush(rtSetLayout);

				std::vector<VkDescriptorSetLayout> layouts{
					rtSetLayout,
					m_context->getSamplerCache().getCommonDescriptorSetLayout(),
					getRenderer()->getBlueNoise().spp_1_buffer.setLayouts,
					m_context->getBindlessSSBOSetLayout()
					, m_context->getBindlessSSBOSetLayout()
					, m_context->getBindlessSamplerSetLayout()
					, m_context->getBindlessTextureSetLayout()
                };

				rtPipe = std::make_unique<ComputePipeResources>("shader/ssgi_rt.glsl", sizeof(RTGIPush), layouts);
			}
        }

        virtual void release() override
        {
            intersect.reset();
            filter.reset();
            reproject.reset();
            rtPipe.reset();
        }
    };


    PoolImageSharedRef DeferredRenderer::renderSSGI(
        VkCommandBuffer cmd,
        GBufferTextures* inGBuffers,
        RenderScene* scene,
        BufferParameterHandle perFrameGPU,
        PoolImageSharedRef inHiz,
        const SkyLightRenderContext& inSky,
        ReflectionProbeContext& reflectionProbeContext)
    {
        if (!inSky.skylightRadiance)
        {
            return nullptr;
        }


        const uint32_t objectCount = (uint32_t)scene->getObjectCollector().size();
        if (objectCount <= 0)
        {
            return nullptr;
        }

        auto* pass = getContext()->getPasses().get<SSGIPass>();
        auto* rtPool = &getContext()->getRenderTargetPools();

        auto gbufferB = inGBuffers->vertexNormal;



        auto historyColor = m_history.prevHdrBeforeAA ? m_history.prevHdrBeforeAA : inGBuffers->hdrSceneColor;

        auto& gbufferA = inGBuffers->gbufferA->getImage();
        auto& gbufferS = inGBuffers->gbufferS->getImage();
        auto& gbufferV = inGBuffers->gbufferV->getImage();
        auto& sceneDepthZ = inGBuffers->depthTexture->getImage();
        inGBuffers->gbufferId->getImage().transitionShaderReadOnly(cmd);

        gbufferA.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        gbufferB->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        gbufferS.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        gbufferV.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        sceneDepthZ.transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT));
        inHiz->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

        PoolImageSharedRef ssgiIntersect = rtPool->createPoolImage(
            "SSGI intersect",
            gbufferB->getImage().getExtent().width  / 2,
            gbufferB->getImage().getExtent().height / 2,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

        gbufferB->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        historyColor->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
        ssgiIntersect->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

        if (scene->isTLASValid() && false)
        {
            ScopePerframeMarker tonemapperMarker(cmd, "rt-intersect", { 1.0f, 1.0f, 0.0f, 1.0f }, &m_gpuTimer);
        
            RTGIPush pushConst;
            pushConst.rayMinRange = 0.1f;
            pushConst.rayMaxRange = 200.0f;

            pass->rtPipe->bindAndPushConst(cmd, &pushConst);

            PushSetBuilder(cmd)
                .addUAV(ssgiIntersect)
                .addBuffer(perFrameGPU)
                .addAS(scene->getTLAS())
                .addSRV(sceneDepthZ, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT))
                .addBuffer(scene->getObjectBufferGPU())
                .addSRV(inSky.skylightRadiance, buildBasicImageSubresourceCube(), VK_IMAGE_VIEW_TYPE_CUBE)
                .addSRV(gbufferB)
                .push(pass->rtPipe.get());

            pass->rtPipe->bindSet(cmd, std::vector<VkDescriptorSet>{
                getContext()->getSamplerCache().getCommonDescriptorSet(),
                getRenderer()->getBlueNoise().spp_8_buffer.set,
                getContext()->getBindlessSSBOSet(),
                getContext()->getBindlessSSBOSet(),
                getContext()->getBindlessSamplerSet(),
                getContext()->getBindlessTexture().getSet(),
            }, 1);

            vkCmdDispatch(cmd, getGroupCount(
                ssgiIntersect->getImage().getExtent().width, 8),
                getGroupCount(ssgiIntersect->getImage().getExtent().height, 8), 1);
        }
        else
        {
            ScopePerframeMarker tonemapperMarker(cmd, "ssgi-intersect", { 1.0f, 1.0f, 0.0f, 1.0f }, &m_gpuTimer);

            SSGIPush pushConst;

            pass->intersect->bindAndPushConst(cmd, &pushConst);
            PushSetBuilder(cmd)
                .addSRV(inHiz)
                .addSRV(inGBuffers->depthTexture, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT))
                .addSRV(gbufferB)
                .addSRV(inGBuffers->gbufferV)
                .addSRV(historyColor)
                .addBuffer(perFrameGPU)
                .addUAV(ssgiIntersect)
                .addSRV(reflectionProbeContext.probe0 ? reflectionProbeContext.probe0 : inSky.skylightRadiance, buildBasicImageSubresourceCube(), VK_IMAGE_VIEW_TYPE_CUBE)
                .addSRV(reflectionProbeContext.probe1 ? reflectionProbeContext.probe1 : inSky.skylightRadiance, buildBasicImageSubresourceCube(), VK_IMAGE_VIEW_TYPE_CUBE)
                .addSRV(inSky.skylightRadiance, buildBasicImageSubresourceCube(), VK_IMAGE_VIEW_TYPE_CUBE)
                .push(pass->intersect.get());

            pass->intersect->bindSet(cmd, std::vector<VkDescriptorSet>
            {
                getContext()->getSamplerCache().getCommonDescriptorSet(),
                getRenderer()->getBlueNoise().spp_1_buffer.set
            }, 1);

            vkCmdDispatch(cmd,
                getGroupCount(ssgiIntersect->getImage().getExtent().width, 8),
                getGroupCount(ssgiIntersect->getImage().getExtent().height, 8), 1);

        }

        ssgiIntersect->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());



        PoolImageSharedRef fullResIntersec = rtPool->createPoolImage(
            "SSGI intersect full",
            gbufferB->getImage().getExtent().width / 2,
            gbufferB->getImage().getExtent().height / 2,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

        PoolImageSharedRef fullResIntersec1 = rtPool->createPoolImage(
            "SSGI intersect full1",
            gbufferB->getImage().getExtent().width / 2,
            gbufferB->getImage().getExtent().height / 2,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);


        PoolImageSharedRef temporalImage = rtPool->createPoolImage(
            "SSGI temporal full",
            gbufferB->getImage().getExtent().width / 2,
            gbufferB->getImage().getExtent().height / 2,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

        PoolImageSharedRef moment = rtPool->createPoolImage(
            "SSGI moment full",
            gbufferB->getImage().getExtent().width / 2,
            gbufferB->getImage().getExtent().height / 2,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

        {


            ScopePerframeMarker tonemapperMarker(cmd, "ssgi-filter half", { 1.0f, 1.0f, 0.0f, 1.0f }, & m_gpuTimer);

            {
                temporalImage->getImage().transitionGeneral(cmd);
                moment->getImage().transitionGeneral(cmd);

                SSGIReprojectPush pushConst;
                pushConst.kMinAlpha = 0.01f;
                pushConst.kMinAlphaMoment = 0.2f;

                pass->reproject->bindAndPushConst(cmd, &pushConst);
                PushSetBuilder(cmd)
                    .addSRV(inGBuffers->depthTexture, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT))
                    .addSRV(inGBuffers->gbufferV)
                    .addBuffer(perFrameGPU)
                    .addSRV(ssgiIntersect)
                    .addSRV(m_history.ssgiHistory ? m_history.ssgiHistory : ssgiIntersect)
                    .addUAV(temporalImage)
                    .addSRV(m_history.ssgiMomentHistory ? m_history.ssgiMomentHistory : ssgiIntersect)
                    .addUAV(moment)
                    .addSRV(gbufferB)
                    .addSRV(m_history.prevGBufferB ? m_history.prevGBufferB : gbufferB)
                    .addSRV(m_history.prevDepth ? m_history.prevDepth : inGBuffers->depthTexture, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT))
                    .addSRV(inGBuffers->gbufferId)
                    .addSRV(m_history.prevGBufferID ? m_history.prevGBufferID : inGBuffers->gbufferId)
                    .push(pass->reproject.get());

                pass->reproject->bindSet(cmd, std::vector<VkDescriptorSet>
                {
                    getContext()->getSamplerCache().getCommonDescriptorSet(),
                }, 1);

                vkCmdDispatch(cmd,
                    getGroupCount(temporalImage->getImage().getExtent().width, 8),
                    getGroupCount(temporalImage->getImage().getExtent().height, 8), 1);

                temporalImage->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
                moment->getImage().transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
            }

            m_history.ssgiHistory = temporalImage;
            m_history.ssgiMomentHistory = moment;

            auto In = temporalImage;
            auto Out = fullResIntersec;

            const uint kFilterNum = 4;

            pass->filter->bind(cmd);
            pass->filter->bindSet(cmd, std::vector<VkDescriptorSet>
            {
                getContext()->getSamplerCache().getCommonDescriptorSet(),
            }, 1);
            for(uint i = 0; i < kFilterNum; i ++)
            {
                In->getImage().transitionShaderReadOnly(cmd);
                Out->getImage().transitionGeneral(cmd);

                SSGIBlurPush pushConst;
                pushConst.kStepSize = 1 << i;

                pass->filter->pushConst(cmd, &pushConst);

                PushSetBuilder(cmd)
                    .addSRV(inGBuffers->depthTexture, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT))
                    .addSRV(gbufferB)
                    .addSRV(In)
                    .addUAV(Out)
                    .addBuffer(perFrameGPU)
                    .addSRV(m_history.prevNormalVertex ? m_history.prevNormalVertex : gbufferB)
                    .addSRV(m_history.prevDepth ? m_history.prevDepth : inGBuffers->depthTexture, RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT))
                    .push(pass->filter.get());

                vkCmdDispatch(cmd,
                    getGroupCount(fullResIntersec->getImage().getExtent().width, 8),
                    getGroupCount(fullResIntersec->getImage().getExtent().height, 8), 1);


                auto tempOut = Out;
                Out = In;
                In = tempOut;
                if (i == 0)
                {
                    Out = fullResIntersec1;
                }
            }


            fullResIntersec = In;
            fullResIntersec->getImage().transitionShaderReadOnly(cmd);
        }


        return fullResIntersec;
    }
}