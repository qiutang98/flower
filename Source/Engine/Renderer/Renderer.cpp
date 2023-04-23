#include "renderer.h"
#include "render_scene.h"
#include <scene/scene.h>
#include "scene_textures.h"

namespace engine
{
    Renderer* getRenderer()
    {
        static Renderer* renderer = Framework::get()->getEngine().getRuntimeModule<Renderer>();
        return renderer;
    }

	void Renderer::registerCheck(Engine* engine)
    {
        ASSERT(engine->existRegisteredModule<VulkanContext>(),
            "When renderer enable, you must register vulkan context module before renderer.");

        ASSERT(engine->existRegisteredModule<SceneManager>(),
            "When renderer enable, you must register vulkan context module before renderer.")
    }

    bool Renderer::init()
    {
        // Fetch and cache vulkan context handle.
        m_context = m_engine->getRuntimeModule<VulkanContext>();
        m_renderScene = new RenderScene(m_context, m_engine->getRuntimeModule<SceneManager>());

        m_temporalBlueNoise = new TemporalBlueNoise();
        m_sharedTextures = new SharedTextures();

        // Prepare windows present need datas.
        if (m_engine->isWindowApp())
        {
            // Preapre command buffers and semaphore, this is useful when render and present to surface.
            initWindowCommandContext();

            m_imguiManager.init(m_context);
        }

        return true;
    }

    bool Renderer::tick(const RuntimeModuleTickData& tickData)
    {
        // Window present render tick.
        if (m_engine->isWindowApp())
        {
            // Imgui new frame.
            m_imguiManager.newFrame();

            // Broadcast tick functions and render imgui.
            tickFunctions.broadcast(tickData, m_context);

            // Prepare render data.
            m_imguiManager.render();

            auto rendererTick = [&](VkCommandBuffer graphicsCmd)
            {
                RHICheck(vkResetCommandBuffer(graphicsCmd, 0));
                VkCommandBufferBeginInfo cmdBeginInfo = RHICommandbufferBeginInfo(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

                // Record tick command functions.
                RHICheck(vkBeginCommandBuffer(graphicsCmd, &cmdBeginInfo));
                {
                    m_renderScene->tick(tickData, graphicsCmd);
                    tickCmdFunctions.broadcast(tickData, graphicsCmd, m_context);
                }
                RHICheck(vkEndCommandBuffer(graphicsCmd));
            };

            VkPipelineStageFlags waitFlags = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;

            // Check main imgui minimized state to decide wether should we present current frame.
            const bool bMainMinimized = m_imguiManager.isMainMinimized();
            if (!bMainMinimized)
            {
                // Acquire next present image.
                uint32_t backBufferIndex = m_context->acquireNextPresentImage();
                ASSERT(backBufferIndex < m_context->getBackBufferCount(), "Swapchain backbuffer count should equal to flighting count.");

                // Prepare current
                VkCommandBuffer graphicsCmd = m_windowCmdContext.mainCmdRing.at(backBufferIndex);

                // Record.
                rendererTick(graphicsCmd);

                // Record ui render.
                m_imguiManager.renderFrame(backBufferIndex);

                // Load all semaphores.
                auto frameStartSemaphore     = m_context->getCurrentFrameWaitSemaphore();
                auto graphicsCmdEndSemaphore = m_windowCmdContext.mainSemaphoreRing[backBufferIndex];
                auto frameEndSemaphore       = m_context->getCurrentFrameFinishSemaphore();

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

                m_context->resetFence();
                m_context->submit((uint32_t)infosRawSubmit.size(), infosRawSubmit.data());
            }
            else
            {
                m_context->waitDeviceIdle();
            }

            m_imguiManager.updateAfterSubmit();

            if (!bMainMinimized)
            {
                m_context->present();
            }
        }

        return true;
    }

    void Renderer::release()
    {
        // Wait device finish all work before release.
        m_context->waitDeviceIdle();

        // Release blue noise.
        delete m_temporalBlueNoise;
        m_temporalBlueNoise = nullptr;

        delete m_sharedTextures;
        m_sharedTextures = nullptr;

        delete m_renderScene; 
        m_renderScene = nullptr;

        if (m_engine->isWindowApp())
        {
            m_imguiManager.release();

            destroyWindowCommandContext();
        }
    }

    void Renderer::initWindowCommandContext()
    {
        ASSERT(m_engine->isWindowApp(), "Only init these context for windows app.");

        // prepare common cmdbuffer and semaphore.
        VkSemaphoreCreateInfo semaphoreInfo{ };
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        // Main window prepare.
        m_windowCmdContext.mainCmdRing.resize(m_context->getBackBufferCount());
        m_windowCmdContext.mainSemaphoreRing.resize(m_context->getBackBufferCount());
        for (size_t i = 0; i < m_context->getBackBufferCount(); i++)
        {
            m_windowCmdContext.mainCmdRing[i] = m_context->createMajorGraphicsCommandBuffer();
            RHICheck(vkCreateSemaphore(m_context->getDevice(), &semaphoreInfo, nullptr, &m_windowCmdContext.mainSemaphoreRing[i]));
        }

        // Second context prepare.
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        RHICheck(vkCreateFence(m_context->getDevice(), &fenceInfo, nullptr, &m_windowCmdContext.secondCmdFence));
        m_windowCmdContext.secondCmd = m_context->createMajorGraphicsCommandBuffer();
    }

    void Renderer::destroyWindowCommandContext()
    {
        ASSERT(m_engine->isWindowApp(), "Only destroy these context for windows app.");

        // Main window release.
        for (size_t i = 0; i < m_context->getBackBufferCount(); i++)
        {
            vkDestroySemaphore(m_context->getDevice(), m_windowCmdContext.mainSemaphoreRing[i], nullptr);
            m_context->freeMajorGraphicsCommandBuffer(m_windowCmdContext.mainCmdRing[i]);
        }

        // Second context destroy.
        m_context->freeMajorGraphicsCommandBuffer(m_windowCmdContext.secondCmd);
        vkDestroyFence(m_context->getDevice(), m_windowCmdContext.secondCmdFence, nullptr);
    }

}