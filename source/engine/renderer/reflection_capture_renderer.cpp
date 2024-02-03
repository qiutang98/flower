#include "reflection_capture_renderer.h"
#include "renderer.h"
#include "render_scene.h"
#include "scene_textures.h"

namespace engine
{
    BufferParameterHandle getReflectionPerframe(
        PerFrameData& result,
        uint dimension,
        VkCommandBuffer graphicsCmd,
        CameraInterface* camera,
        const RuntimeModuleTickData& tickData)
    {
        result = { };

        result.appTime = 
        {
            tickData.runTime,
            glm::sin(tickData.runTime),
            glm::cos(tickData.runTime),
            0.0f
        };

        // Frame index always zero for reflection capture.
        result.frameIndex = { 0, 0, 0, 0 };

        result.renderWidth  = dimension;
        result.renderHeight = dimension;
        result.postWidth    = dimension;
        result.postHeight   = dimension;

        // Always camera cut.
        result.bCameraCut = true;

        // No jitter.
        result.jitterData.x        = 0.0f;
        result.jitterData.y        = 0.0f;
        result.jitterPeriod        = 1;
        result.bEnableJitter       = false;
        result.basicTextureLODBias = 0.0f;
        result.bTAAU               = false;

        // Fill camera matrix.
        camera->fillPerframe(result);
        // Don't care prev frame infos.

        // Now fill scene data.
        getRenderer()->getScene()->fillPerframe(result, tickData);

        // 
        auto perFrameGPU = getContext()->getBufferParameters().getStaticUniform("FrameData", sizeof(result));
        {
            result.renderType = ERendererType_ReflectionCapture;

            // hack and do some optimize.
            result.postprocessing.ssao_enable = 0;

            // Cascade only set to one.
            result.sunLightInfo.cascadeConfig.maxDrawDepthDistance = 2000.0f;
            result.sunLightInfo.cascadeConfig.bSDSM           = false;
            result.sunLightInfo.cascadeConfig.cascadeCount    = 1;
            result.sunLightInfo.cascadeConfig.percascadeDimXY = math::clamp(dimension * 4U, 1024u, 2048U);
            result.sunLightInfo.cascadeConfig.filterSize      = 4.0f;
            result.sunLightInfo.cascadeConfig.bContactShadow  = false;
        }


        perFrameGPU->updateData(result);

        return perFrameGPU;
    }

    PoolImageSharedRef ReflectionCaptureRenderer::render(
        uint dimension,
        VkCommandBuffer graphicsCmd, 
        CameraInterface* camera,
        const RuntimeModuleTickData& tickData)
    {
        auto* renderScene = getRenderer()->getScene();
        PerFrameData perframe { };
        auto perframeGPU = getReflectionPerframe(perframe, dimension, graphicsCmd, camera, tickData);

        auto gbuffer = GBufferTextures::build(dimension, dimension, dimension, dimension);
        gbuffer.clearValue(graphicsCmd);

        renderStaticMeshPrepass(
            graphicsCmd,
            &gbuffer,
            getRenderer()->getScene(),
            perframeGPU,
            nullptr);

        AtmosphereTextures atmosphereTextures{ };
        renderAtmosphere(
            graphicsCmd,
            &gbuffer,
            getRenderer()->getScene(),
            perframeGPU,
            perframe,
            atmosphereTextures,
            false,
            nullptr);

        SkyLightRenderContext skylightContext { };
        ReflectionProbeContext probeContext{ };
        renderSkylight(
            graphicsCmd,
            atmosphereTextures,
            perframe,
            getRenderer()->getScene(),
            skylightContext,
            probeContext,
            nullptr);

        prepareTerrainLODS(
            graphicsCmd,
            &gbuffer,
            getRenderer()->getScene(),
            perframeGPU,
            nullptr);

        renderTerrainGbuffer(
            graphicsCmd,
            &gbuffer,
            getRenderer()->getScene(),
            perframeGPU,
            nullptr);

        PoolImageSharedRef hzbClosest;
        PoolImageSharedRef hzbFurthest;
        renderHzb(
            hzbClosest,
            hzbFurthest,
            graphicsCmd,
            &gbuffer,
            getRenderer()->getScene(),
            perframeGPU,
            nullptr);

        renderStaticMeshGBuffer(
            graphicsCmd,
            &gbuffer,
            getRenderer()->getScene(),
            perframeGPU,
            hzbFurthest,
            nullptr,
            nullptr);

        auto sceneDepthRangeBuffer = sceneDepthRangePass(
            graphicsCmd,
            &gbuffer,
            perframeGPU,
            getRenderer()->getScene(),
            nullptr);

        SDSMInfos sunSDSMInfos{ };
        SDSMInfos moonSDSMInfos{ };
        renderSDSM(
            graphicsCmd,
            &gbuffer,
            perframe.sunLightInfo,
            getRenderer()->getScene(),
            perframeGPU,
            sunSDSMInfos,
            moonSDSMInfos,
            sceneDepthRangeBuffer,
            nullptr,
            nullptr);

        renderDirectLighting(
            graphicsCmd,
            &gbuffer,
            getRenderer()->getScene(),
            perframeGPU,
            atmosphereTextures,
            sunSDSMInfos,
            moonSDSMInfos,
            nullptr, 
            nullptr);

        // Composite sky.
        renderAtmosphere(
            graphicsCmd, 
            &gbuffer, 
            getRenderer()->getScene(), 
            perframeGPU, 
            perframe,
            atmosphereTextures, 
            true,
            nullptr);

        renderGIDiffuse(
            graphicsCmd,
            &gbuffer, getRenderer()->getScene(),
            perframeGPU,
            nullptr,
            nullptr,
            skylightContext,
            nullptr);

        renderGIReflection(
            graphicsCmd,
            &gbuffer,
            getRenderer()->getScene(),
            perframe,
            perframeGPU,
            skylightContext,
            nullptr);

        gbuffer.hdrSceneColor->getImage().transitionShaderReadOnly(graphicsCmd);

        // Now hdr texture prepare.
        return gbuffer.hdrSceneColor;
    }
}