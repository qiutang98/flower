#include "editor.h"

#if _WIN32
#include <Windows.h>
#endif

#include "widgets/hub.h"
#include "widgets/dockspace.h"
#include "widgets/downbar.h"
#include "widgets/console.h"
#include "widgets/content.h"
#include "widgets/scene_outliner.h"
#include "widgets/detail.h"
#include "builtin_resources.h"
#include "widgets/viewport.h"
#include <asset/asset_manager.h>
#include <ui/imgui/imgui_impl_vulkan.h>
#include <profile/profile.h>

using namespace engine;
using EWindowModeType = GLFWWindows::InitConfig::EWindowMode;

bool Editor::init()
{
#if _WIN32
    // Console output set to utf8 encode.
    SetConsoleOutputCP(CP_UTF8);
#endif 

    m_renderer = getRenderer();

    // Add hub widget when init.
    m_hubHandle = m_widgetManager.addWidget<HubWidget>();

    m_tickFunctionHandle = m_renderer->tickFunctions.addRaw(this, &Editor::tick);
    m_tickCmdFunctionHandle = m_renderer->tickCmdFunctions.addRaw(this, &Editor::tickWithCmd);

    m_onWindowRequireColosedHandle = m_windows.registerClosedEventBody([&](const GLFWWindows* windows)
    { 
        return  this->onWindowRequireClosed(windows);
    });

    m_builtinResources = std::make_unique<EditorBuiltinResource>();

    return true;
}

bool Editor::release()
{
    m_windows.unregisterClosedEventBody(m_onWindowRequireColosedHandle);

    CHECK(m_renderer->tickFunctions.remove(m_tickFunctionHandle));
    CHECK(m_renderer->tickCmdFunctions.remove(m_tickCmdFunctionHandle));

    m_widgetManager.release();

    getContext()->waitDeviceIdle();
    m_builtinResources = nullptr;

    if (m_projectContentModel)
    {
        m_projectContentModel->release();
    }

    return true;
}

Editor* Editor::get()
{
	static Editor editor;
	return &editor;
}

int Editor::run(int argc, char** argv)
{
    // Init log infos.
    LoggerSystem::initBasicConfig(
    {
        .bOutputLog = true,
        .outputLogPath = "editor",
    });

    LOG_TRACE("Init reflection compile trace uuid {}.", kRelfectionCompilePlayHolder);

    // Init cvar configs.
    initBasicCVarConfigs();

    // Install engine hook.
    Engine::get()->initGLFWWindowsHook(m_windows);

    GLFWWindows::InitConfig windowInitConfig =
    {
        .appName        = "dark editor",
        .appIcon        = "image/editorIcon.png",
        .windowShowMode = EWindowModeType::Free,
        .bResizable     = false,
        .initWidth      = 1600U,
        .initHeight     = 900U,
    };

    // Windows init.
    if (m_windows.init(windowInitConfig))
    {
        // Editor init after windows init.
        ASSERT(this->init(), "Fail to init editor!");

        // Windows loop.
        m_windows.loop();

        // Editor release before windows exit.
        ASSERT(this->release(), "Fail to release editor!");

        // Windows exit.
        ASSERT(m_windows.release(), "Fail to release all part of application!");
    }

    // Uninstall engine hook.
    Engine::get()->releaseGLFWWindowsHook();

	return 0;
}

void Editor::setTitleName(const std::u16string& name) const
{
    std::string newTitleName = Engine::get()->getGLFWWindows()->getName() + " - " + utf8::utf16to8(name);

    glfwSetWindowTitle(Engine::get()->getGLFWWindows()->getGLFWWindowHandle(), newTitleName.c_str());
}

ImTextureID Editor::getImGuiTexture(VulkanImage* image, const VkSamplerCreateInfo& sampler)
{
    size_t hash;
    {
        hash = (size_t)crc::crc32(&sampler, sizeof(sampler));
        hash = hashCombine(hash, std::hash<UUID64u>{}(image->getRuntimeUUID()));
    }

    if (VK_NULL_HANDLE == m_cacheImGuiImage[hash].view || VK_NULL_HANDLE == m_cacheImGuiImage[hash].sampler)
    {
        m_cacheImGuiImage[hash].view = image->getOrCreateView(buildBasicImageSubresource()).view;
        m_cacheImGuiImage[hash].sampler = getContext()->getSamplerCache().createSampler(sampler);
    }

    return &m_cacheImGuiImage.at(hash);
}

ImTextureID Editor::getClampToTransparentBorderImGuiTexture(VulkanImage* image)
{
    static const VkSamplerCreateInfo info = SamplerFactory::pointClampBorder0000();
    return getImGuiTexture(image, info);
}

void Editor::updateApplicationTitle()
{
    ZoneScopedN("Editor::updateApplicationTitle()");
    
    const auto& projectConfig = getAssetManager()->getProjectConfig();
    if (!projectConfig.projectName.empty())
    {
        auto activeScene = getSceneManager()->getActiveScene();

        std::u16string showName = projectConfig.projectName + u" - " + utf8::utf8to16(activeScene->getName());
        if (activeScene->isDirty())
        {
            showName += u"*";
        }

        // Update title name.
        Editor::get()->setTitleName(showName);
    }
}

void Editor::shortcutHandle()
{
    ZoneScopedN("Editor::shortcutHandle()");
    // Handle undo shortcut. ctrl_z and ctrl_y.
    if (ImGui::IsKeyDown(ImGuiKey_LeftCtrl) || ImGui::IsKeyDown(ImGuiKey_RightCtrl))
    {
        if (ImGui::IsKeyPressed(ImGuiKey_S))
        {
            if (!getAssetManager()->getDirtyAsset<Scene>().empty())
            {
                m_dockSpace->sceneAssetSave.open();
            }
        }
    }
}

void Editor::closedHubWidget()
{
    ZoneScopedN("Editor::closedHubWidget()");

    if (m_hubHandle)
    {
        onceEventAfterTick.add([&]() 
        {
            CHECK(m_widgetManager.removeWidget(m_hubHandle));
            m_hubHandle = nullptr;

            // Then create all widgets.
            m_dockSpace = m_widgetManager.addWidget<MainViewportDockspaceAndMenu>();
            CHECK(m_widgetManager.addWidget<DownbarWidget>());

            // Console.
            {
                m_consoleHandle = m_widgetManager.addWidget<WidgetConsole>();

                // Register console in view.
                {
                    WidgetInView consoleView = { .bMultiWindow = false, .widgets = { m_consoleHandle } };
                    m_dockSpace->widgetInView.add(consoleView);
                }
            }

            // Content
            {
                m_projectContentModel = std::make_unique<ProjectContentModel>();

                WidgetInView contentView = { .bMultiWindow = true };
                for (size_t i = 0; i < kMultiWidgetMaxNum; i++)
                {
                    m_contents[i] = m_widgetManager.addWidget<WidgetContent>(i, m_projectContentModel.get());
                    contentView.widgets[i] = m_contents[i];
                }
                m_contents[0]->setVisible(true);

                m_dockSpace->widgetInView.add(contentView);
            }

            // Outliner
            {
                m_outliner = m_widgetManager.addWidget<SceneOutlinerWidget>();

                // Register console in view.
                {
                    WidgetInView outlinerView = { .bMultiWindow = false, .widgets = { m_outliner } };
                    m_dockSpace->widgetInView.add(outlinerView);
                }
            }

            // Detail
            {
                WidgetInView detailView = { .bMultiWindow = true };
                for (size_t i = 0; i < kMultiWidgetMaxNum; i++)
                {
                    m_details[i] = m_widgetManager.addWidget<WidgetDetail>(i);
                    detailView.widgets[i] = m_details[i];
                    m_details[i]->setVisible(false);
                }
                m_details[0]->setVisible(true);

                m_dockSpace->widgetInView.add(detailView);
            }

            // Viewport
            {
                WidgetInView viewportView = { .bMultiWindow = true };
                for (size_t i = 0; i < kMultiWidgetMaxNum; i++)
                {
                    m_viewports[i] = m_widgetManager.addWidget<ViewportWidget>(i);
                    viewportView.widgets[i] = m_viewports[i];
                    m_viewports[i]->setVisible(false);
                }
                m_viewports[0]->setVisible(true);

                m_dockSpace->widgetInView.add(viewportView);
            }
        });
    }
}

void Editor::tick(const RuntimeModuleTickData& tickData, VulkanContext* context)
{
    ZoneScopedN("Editor::tick(const RuntimeModuleTickData&, VulkanContext*)");

    // Clear cache imgui image before tick.
    m_cacheImGuiImage.clear();

    if (m_projectContentModel)
    {
        m_projectContentModel->tick();
    }

    m_widgetManager.tick(tickData, context);
    tickFunctions.broadcast(tickData, context);

    updateApplicationTitle();

    shortcutHandle();
}

void Editor::tickWithCmd(const RuntimeModuleTickData& tickData, VkCommandBuffer cmd, VulkanContext* context)
{
    m_widgetManager.tickWithCmd(tickData, cmd, context);
    tickCmdFunctions.broadcast(tickData, cmd, context);

    onceEventAfterTick.brocast();
}

bool Editor::onWindowRequireClosed(const GLFWWindows* windows)
{
    if (!getAssetManager()->isProjectSetup())
    {
        return false;
    }

    bool bContinue = false;

    if (!getAssetManager()->getDirtyAsset<Scene>().empty())
    {
        bContinue = true;
        if (m_dockSpace->sceneAssetSave.open())
        {
            CHECK(!m_dockSpace->sceneAssetSave.afterEventAccept);
            m_dockSpace->sceneAssetSave.afterEventAccept = [windows]()
            {
                glfwSetWindowShouldClose(windows->getGLFWWindowHandle(), 1);
            };
        }
    }

    return bContinue;
}
