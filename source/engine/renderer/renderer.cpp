#include "renderer.h"
#include "../engine.h"
#include "render_scene.h"
#include "scene_textures.h"

namespace engine
{
    RendererManager* getRenderer()
    {
        static RendererManager* renderer = Engine::get()->getRuntimeModule<RendererManager>();
        return renderer;
    }

    void RendererManager::registerCheck(Engine* engine)
    {
        ASSERT(engine->existRegisteredModule<VulkanContext>(),
            "When renderer enable, you must register vulkan context module before renderer.");
    }

    bool RendererManager::init()
    {
        m_renderScene = new RenderScene();

        // Init global render resource.
        m_temporalBlueNoise = new TemporalBlueNoise();
        m_sharedTextures = new SharedTextures();

        m_fallbackSSBO = std::make_unique<VulkanBuffer>(
            getContext()->getVMABuffer(),
            "dump ssbo",
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            0,
            sizeof(uint)
        );

        // Prepare windows present need datas.
        if (m_engine->isWindowApplication())
        {
            // Preapre command buffers and semaphore, this is useful when render and present to surface.
            initWindowCommandContext();

            m_imguiManager.init();
        }

        return true;
    }

    bool RendererManager::tick(const RuntimeModuleTickData& tickData)
    {
        // Window present render tick.
        if (m_engine->isWindowApplication())
        {
            // Imgui new frame.
            m_imguiManager.newFrame();

            // Broadcast tick functions and render imgui.
            tickFunctions.broadcast(tickData, getContext());

            // Prepare render data.
            m_imguiManager.render();

            auto rendererTick = [&](VkCommandBuffer graphicsCmd)
            {
                RHICheck(vkResetCommandBuffer(graphicsCmd, 0));
                VkCommandBufferBeginInfo cmdBeginInfo = 
                    RHICommandbufferBeginInfo(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

                // Record tick command functions.
                RHICheck(vkBeginCommandBuffer(graphicsCmd, &cmdBeginInfo));
                {
                    // Render scene update and collect data.
                    m_renderScene->tick(tickData, graphicsCmd);

                    // Tick delegates functions.
                    tickCmdFunctionsBefore.broadcast(tickData, graphicsCmd, getContext());
                    tickCmdFunctions.broadcast(tickData, graphicsCmd, getContext());
                }
                RHICheck(vkEndCommandBuffer(graphicsCmd));
            };

            VkPipelineStageFlags waitFlags = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;

            // Check main imgui minimized state to decide wether should we present current frame.
            const bool bMainMinimized = m_imguiManager.isMainMinimized();
            if (!bMainMinimized)
            {
                // Acquire next present image.
                uint32_t backBufferIndex = getContext()->acquireNextPresentImage();
                ASSERT(backBufferIndex < getContext()->getBackBufferCount(), "Swapchain backbuffer count should equal to flighting count.");

                // Prepare current
                VkCommandBuffer graphicsCmd = m_windowCmdContext.mainCmdRing.at(backBufferIndex);

                // Record.
                rendererTick(graphicsCmd);

                // Record ui render.
                m_imguiManager.renderFrame(backBufferIndex);

                // Load all semaphores.
                auto frameStartSemaphore = getContext()->getCurrentFrameWaitSemaphore();
                auto graphicsCmdEndSemaphore = m_windowCmdContext.mainSemaphoreRing[backBufferIndex];
                auto frameEndSemaphore = getContext()->getCurrentFrameFinishSemaphore();

                // Submit with semaphore.
                RHISubmitInfo graphicsCmdSubmitInfo{};
                graphicsCmdSubmitInfo.setWaitStage(&waitFlags)
                    .setWaitSemaphore(&frameStartSemaphore, 1)
                    .setSignalSemaphore(&graphicsCmdEndSemaphore, 1)
                    .setCommandBuffer(&graphicsCmd, 1);

                RHISubmitInfo uiCmdSubmitInfo{};
                VkCommandBuffer uiCmdBuffer = m_imguiManager.getCommandBuffer(backBufferIndex);
                uiCmdSubmitInfo.setWaitStage(&waitFlags)
                    .setWaitSemaphore(&graphicsCmdEndSemaphore, 1)
                    .setSignalSemaphore(&frameEndSemaphore, 1)
                    .setCommandBuffer(&uiCmdBuffer, 1);

                std::vector<VkSubmitInfo> infosRawSubmit{ graphicsCmdSubmitInfo, uiCmdSubmitInfo };

                getContext()->resetFence();
                getContext()->submit((uint32_t)infosRawSubmit.size(), infosRawSubmit.data());
            }
            else
            {
                getContext()->waitDeviceIdle();
            }

            m_imguiManager.updateAfterSubmit();

            if (!bMainMinimized)
            {
                getContext()->present();
            }
        }

        return true;
    }

    bool RendererManager::beforeRelease()
    {
        // Wait device work finish all work before release.
        getContext()->waitDeviceIdle();

        return true;
    }

    bool RendererManager::release()
    {
        m_fallbackSSBO.reset();

        // Clear render scene.
        delete m_renderScene;

        // Clear global render resource.
        delete m_temporalBlueNoise;
        delete m_sharedTextures;

        if (m_engine->isWindowApplication())
        {
            m_imguiManager.release();

            destroyWindowCommandContext();
        }

        return true;
    }

    void RendererManager::initWindowCommandContext()
    {
        // prepare common cmdbuffer and semaphore.
        VkSemaphoreCreateInfo semaphoreInfo{ };
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        // Main window prepare.
        m_windowCmdContext.mainCmdRing.resize(getContext()->getBackBufferCount());
        m_windowCmdContext.mainSemaphoreRing.resize(getContext()->getBackBufferCount());
        for (size_t i = 0; i < getContext()->getBackBufferCount(); i++)
        {
            m_windowCmdContext.mainCmdRing[i] = getContext()->createMajorGraphicsCommandBuffer();
            RHICheck(vkCreateSemaphore(getDevice(), &semaphoreInfo, nullptr, &m_windowCmdContext.mainSemaphoreRing[i]));
        }
    }

    void RendererManager::destroyWindowCommandContext()
    {
        // Main window release.
        for (size_t i = 0; i < getContext()->getBackBufferCount(); i++)
        {
            vkDestroySemaphore(getContext()->getDevice(), m_windowCmdContext.mainSemaphoreRing[i], nullptr);
            getContext()->freeMajorGraphicsCommandBuffer(m_windowCmdContext.mainCmdRing[i]);
        }
    }
}