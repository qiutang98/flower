#include "dockspace.h"
#include "console.h"
#include "content.h"
#include <asset/asset_manager.h>
#include <nfd.h>
#include <scene/scene_manager.h>
using namespace engine;
using namespace engine::ui;

const static std::string MAINMENU_GCloseIcon = ICON_FA_POWER_OFF;

MainViewportDockspaceAndMenu::MainViewportDockspaceAndMenu()
	: WidgetBase("MainViewportDockspaceAndMenu", "MainViewportDockspaceAndMenu")
    , sceneAssetSave(combineIcon("Save edited scenes...", ICON_FA_MESSAGE))
    , contentAssetImport(combineIcon("Imported assets config...", ICON_FA_MESSAGE))
{
    m_bShow = false;
}



void MainViewportDockspaceAndMenu::dockspace(
    bool bNewWindow,
    const std::string& name, 
    const RuntimeModuleTickData& tickData, 
    VulkanContext* context, 
    ImGuiViewport* viewport, 
    std::function<void()>&& menu)
{
    ZoneScoped;

    ImGuiIO& io = ImGui::GetIO();
    if (!(io.ConfigFlags & ImGuiConfigFlags_DockingEnable))
    {
        return;
    }
    static bool bShow = true;

    static ImGuiDockNodeFlags dockspaceFlags = ImGuiDockNodeFlags_None;

    ImGuiWindowFlags windowFlags
        = ImGuiWindowFlags_MenuBar
        | ImGuiWindowFlags_NoDocking
        | ImGuiWindowFlags_NoTitleBar
        | ImGuiWindowFlags_NoCollapse
        | ImGuiWindowFlags_NoResize
        | ImGuiWindowFlags_NoMove
        | ImGuiWindowFlags_NoBringToFrontOnFocus
        | ImGuiWindowFlags_NoNavFocus;
    if (dockspaceFlags & ImGuiDockNodeFlags_PassthruCentralNode)
    {
        windowFlags |= ImGuiWindowFlags_NoBackground;
    }

    {
        ImGui::SetNextWindowPos(viewport->WorkPos);
        ImGui::SetNextWindowSize(viewport->WorkSize);
        ImGui::SetNextWindowViewport(viewport->ID);

        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 6.0f));
        {
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
            if(bNewWindow)
            {
                ImGui::Begin(name.c_str(), &bShow, windowFlags);
            }
            ImGui::PopStyleVar();

            ImGuiID dockspaceId = ImGui::GetID(name.c_str());
            ImGui::DockSpace(dockspaceId, ImVec2(0.0f, 0.0f), dockspaceFlags);

            if (ImGui::BeginMenuBar())
            {
                menu();

                ImGui::EndMenuBar();
            }
        }
        ImGui::PopStyleVar(3);

        if (bNewWindow)
        {
            ImGui::End();
        }
    }
}

void MainViewportDockspaceAndMenu::onTick(const RuntimeModuleTickData& tickData, VulkanContext* context)
{
    static std::string dockspaceName = "Engine MainViewport Dockspace";
    dockspace(true, dockspaceName, tickData, context, ImGui::GetMainViewport(), [this]() { drawDockspaceMenu(); });

    sceneAssetSave.draw();
    contentAssetImport.draw();
}


void MainViewportDockspaceAndMenu::drawDockspaceMenu()
{
    if (ImGui::BeginMenu("  FILE  "))
    {
        const bool bSceneDirty = !getAssetManager()->getDirtyAsset<Scene>().empty();
        if (ImGui::MenuItem(combineIcon("New Scene", ICON_NONE).c_str()))
        {
            if (bSceneDirty)
            {
                if (sceneAssetSave.open())
                {
                    CHECK(!sceneAssetSave.afterEventAccept);
                    sceneAssetSave.afterEventAccept = []()
                    {
                        getSceneManager()->releaseScene();
                    };
                }
            }
            else
            {
                getSceneManager()->releaseScene();
            }
        }

        if (!bSceneDirty) { ImGui::BeginDisabled(); }
        if (ImGui::MenuItem(combineIcon("Save Scene", ICON_NONE).c_str()))
        {
            if (bSceneDirty)
            {
                sceneAssetSave.open();
            }
        }
        if (!bSceneDirty) { ImGui::EndDisabled(); }

        ImGui::Separator();

        if (ImGui::MenuItem(combineIcon(" Exit", MAINMENU_GCloseIcon).c_str()))
        {
            m_engine->getGLFWWindows()->close();
        }



        ImGui::EndMenu();
    }
    ImGui::Separator();

    if (ImGui::BeginMenu("  EDIT  "))
    {
        if (ImGui::MenuItem(combineIcon("Undo", ICON_FA_ARROW_ROTATE_LEFT).c_str()))
        {

        }

        if (ImGui::MenuItem(combineIcon("Redo", ICON_FA_ARROW_ROTATE_RIGHT).c_str()))
        {

        }


        ImGui::EndMenu();
    }
    ImGui::Separator(); 

    if (ImGui::BeginMenu("  VIEW  "))
    {
        widgetInView.loop([](WidgetInView& widgetInView) 
        {
            if(!widgetInView.bMultiWindow)
            {
                auto* widget = widgetInView.widgets[0];
                if (ImGui::MenuItem(widget->getName().c_str(), NULL, widget->getVisible()))
                {
                    widget->setVisible(!widget->getVisible());
                }
            }
            else
            {
                const char* name = widgetInView.widgets[0]->getWidgetName().c_str();
                if (ImGui::BeginMenu(name))
                {
                    for (size_t i = 0; i < widgetInView.widgets.size(); i++)
                    {
                        auto* widget = widgetInView.widgets[i];
                        std::string secondName = widgetInView.widgets[i]->getName();
                        if (ImGui::MenuItem(secondName.c_str(), NULL, widget->getVisible()))
                        {
                            widget->setVisible(!widget->getVisible());
                        }
                    }

                    ImGui::EndMenu();
                }
            }
        });

        ImGui::EndMenu();
    }
    ImGui::Separator();

    if (ImGui::BeginMenu("  HELP  "))
    {
        if (ImGui::MenuItem(combineIcon(" About", ICON_FA_CIRCLE_QUESTION).c_str()))
        {

        }

        ImGui::Separator();

        if (ImGui::MenuItem(combineIcon("Developer", ICON_NONE).c_str()))
        {

        }

        if (ImGui::MenuItem(combineIcon("SDKs", ICON_NONE).c_str()))
        {

        }

        ImGui::EndMenu();
    }
    ImGui::Separator();
}


SceneAssetSaveWidget::SceneAssetSaveWidget(const std::string& titleName)
    : ImGuiPopupSelfManagedOpenState(titleName, ImGuiWindowFlags_AlwaysAutoResize)
{

}

void SceneAssetSaveWidget::onDraw()
{
    auto scenes = getAssetManager()->getDirtyAsset<Scene>();

    // Current only support one scene edit.
    CHECK(scenes.size() == 1);

    auto scene = scenes[0].lock();
    const bool bTemp = scene->getSaveInfo().isTemp();

    if (m_processingAsset != scene->getSaveInfo())
    {
        m_processingAsset = scene->getSaveInfo();
        m_bSelected = true;
    }

    ImGui::TextDisabled("Scene still un-save after edited, please decide discard or save.");
    ImGui::NewLine();
    ImGui::Indent();
    {
        std::string showName = Scene::uiGetAssetReflectionInfo().decoratedName + ":  " + scene->getName() +
            (bTemp ? "*  (Created)" : "*  (Edited)");

        if (bTemp) ImGui::PushStyleColor(ImGuiCol_Text, (ImVec4)ImColor::HSV(0.15f, 0.6f, 1.0f));
        {
            ImGui::Checkbox("##HideCheckBox", &m_bSelected); ImGui::SameLine();
            ImGui::Selectable(showName.c_str(), &m_bSelected, ImGuiSelectableFlags_DontClosePopups);
        }
        if (bTemp) ImGui::PopStyleColor();
    }
    ImGui::Unindent();
    ImGui::NewLine();
    ImGui::NewLine();
    ImGui::NewLine();

    bool bAccept = false;

    if (ImGui::Button("Save", ImVec2(120, 0)))
    {
        bAccept = true;
        if (m_bSelected)
        {
            if (bTemp)
            {
                std::string path;

                const auto assetStartFolder = utf8::utf16to8(getAssetManager()->getProjectConfig().assetPath);
                nfdchar_t* outPathChars;

                std::string suffix = std::string(scene->getSuffix()).erase(0, 1) + "\0";
                nfdchar_t* filterList = suffix.data();
                nfdresult_t result = NFD_SaveDialog(filterList, assetStartFolder.c_str(), &outPathChars);
                if (result == NFD_OKAY)
                {
                    path = outPathChars;
                    free(outPathChars);
                }

                auto u16PathString = utf8::utf8to16(path);
                std::filesystem::path fp(u16PathString);
                if (!path.empty())
                {
                    std::filesystem::path assetName = fp.filename();
                    std::string assetNameUtf8 = utf8::utf16to8(assetName.u16string()) + Scene::getCDO()->getSuffix();

                    const auto relativePath = buildRelativePathUtf8(getAssetManager()->getProjectConfig().assetPath, fp.remove_filename());

                    const AssetSaveInfo newInfo(assetNameUtf8, relativePath);
                    scene->changeSaveInfo(newInfo);

                    if (!scene->save())
                    {
                        LOG_ERROR("Fail to save new created scene {0} in path {1}.", scene->getName(), utf8::utf16to8(scene->getSavePath().u16string()));
                    }
                }
            }
            else
            {
                if (!scene->save())
                {
                    LOG_ERROR("Fail to save edited scene {0} in path {1}.", scene->getName(), utf8::utf16to8(scene->getSavePath().u16string()));
                }
            }
        }

        m_bSelected = true;
        ImGui::CloseCurrentPopup();
    }

    ImGui::SetItemDefaultFocus();
    ImGui::SameLine();
    if (ImGui::Button("Discard", ImVec2(120, 0)))
    {
        bAccept = true;
        if (m_bSelected)
        {
            const auto saveInfo = scene->getSaveInfo();
            scene->discardChanged();

            if (!saveInfo.isTemp())
            {
                // Reload src scene in disk.
                getSceneManager()->loadScene(saveInfo.toPath());
            }
        }
        m_bSelected = true;
        ImGui::CloseCurrentPopup();
    }

    if (bAccept)
    {
        if (afterEventAccept)
        {
            afterEventAccept();
        }
        onClosed();
    }
}

ContentAssetImportWidget::ContentAssetImportWidget(const std::string& titleName)
    : ImGuiPopupSelfManagedOpenState(titleName, ImGuiWindowFlags_AlwaysAutoResize)
{

}

void ContentAssetImportWidget::onDraw()
{
    if (m_bImporting)
    {
        onDrawImporting();
    }
    else
    {
        onDrawState();
    }
}

void ContentAssetImportWidget::onDrawState()
{
    CHECK(!importConfigs.empty());
    CHECK(!typeName.empty());

    auto type = rttr::type::get_by_name(typeName);

    const auto& method = type.get_method("uiGetAssetReflectionInfo");
    rttr::variant returnValue = method.invoke({});
    const auto& meta = returnValue.get_value<AssetReflectionInfo>();

    CHECK(meta.importConfig.drawAssetImportConfig);
    for (auto& ptr : importConfigs)
    {
        meta.importConfig.drawAssetImportConfig(ptr);
    }

    bool bAccept = false;

    if (ImGui::Button("OK", ImVec2(120, 0)))
    {
        m_bImporting = true;
        {
            CHECK(!m_importProgress.logHandle.isValid());
            {
                m_importProgress.logHandle = LoggerSystem::get()->pushCallback([&](const std::string& info, ELogType type)
                {
                    m_importProgress.logItems.push_back({ type, info });
                    if (static_cast<uint32_t>(m_importProgress.logItems.size()) >= 60)
                    {
                        m_importProgress.logItems.pop_front();
                    }
                });
            }

            CHECK(m_executeFutures.futures.empty());
            {
                auto type = rttr::type::get_by_name(typeName);

                const auto& method = type.get_method("uiGetAssetReflectionInfo");
                rttr::variant returnValue = method.invoke({});
                const auto& meta = returnValue.get_value<AssetReflectionInfo>();

                CHECK(meta.importConfig.importAssetFromConfigThreadSafe);

                const auto loop = [this, meta](const size_t loopStart, const size_t loopEnd)
                {
                    for (size_t i = loopStart; i < loopEnd; ++i)
                    {
                        meta.importConfig.importAssetFromConfigThreadSafe(importConfigs[i]);
                    }
                };
                m_executeFutures = Engine::get()->getThreadPool()->parallelizeLoop(0, importConfigs.size(), loop);
            }
        }
    }

    ImGui::SetItemDefaultFocus();
    ImGui::SameLine();
    if (ImGui::Button("Cancel", ImVec2(120, 0)))
    {
        bAccept = true;
        ImGui::CloseCurrentPopup();
    }

    if (bAccept)
    {
        if (afterEventAccept)
        {
            afterEventAccept();
        }
        onClosed();
    }
}

void ContentAssetImportWidget::onDrawImporting()
{
    CHECK(!m_executeFutures.futures.empty());
    CHECK(m_bImporting);

    ImGui::Indent();
    ImGui::Text("Asset  Importing ...    ");
    ImGui::SameLine();

    float progress = m_executeFutures.getProgress();
    ImGui::ProgressBar(progress, ImVec2(0.0f, 0.0f));

    ImGui::Unindent();
    ImGui::Separator();

    ImGui::BeginDisabled();
    for (int i = 0; i < m_importProgress.logItems.size(); i++)
    {
        ImVec4 color;
        if (m_importProgress.logItems[i].first == ELogType::Error || 
            m_importProgress.logItems[i].first == ELogType::Fatal)
        {
            color = ImVec4(1.0f, 0.08f, 0.08f, 1.0f);
        }
        else if (m_importProgress.logItems[i].first == ELogType::Warn)
        {
            color = ImVec4(1.0f, 1.0f, 0.1f, 1.0f);
        }
        else
        {
            color = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
        }


        ImGui::PushStyleColor(ImGuiCol_Text, color);
        ImGui::Selectable(m_importProgress.logItems[i].second.c_str());
        ImGui::PopStyleColor();
    }
    ImGui::EndDisabled();

    bool bAccept = false;
    if (progress > 0.99f)
    {
        m_executeFutures.wait();

        // Clean state.
        m_bImporting = false;
        m_executeFutures = {};
        if (m_importProgress.logHandle.isValid())
        {
            LoggerSystem::get()->popCallback(m_importProgress.logHandle);
            m_importProgress.logHandle.reset();
            m_importProgress.logItems.clear();
        }

        bAccept = true;
        ImGui::CloseCurrentPopup();
    }

    if (bAccept)
    {
        if (afterEventAccept)
        {
            afterEventAccept();
        }
        onClosed();
    }
}
