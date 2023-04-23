#include "dockspace.h"

#include "../editor.h"

#include <imgui/region_string.h>

using namespace engine;
using namespace engine::ui;

#include <imgui/imgui.h>

RegionStringInit MainMenu_File("File", "  FILE  ", "MainMenu");
RegionStringInit MainMenu_Edit("Edit", "  EDIT  ", "MainMenu");
RegionStringInit MainMenu_View("View", "  VIEW  ", "MainMenu");
RegionStringInit MainMenu_Help("Help", "  HELP  ", "MainMenu");

MainViewportDockspaceAndMenu::MainViewportDockspaceAndMenu(Editor* editor)
	: Widget(editor, "MainViewportDockspaceAndMenu")
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
}

void MainViewportDockspaceAndMenu::drawDockspaceMenu()
{
    if (ImGui::BeginMenu(MainMenu_File.imgui()))
    {

        ImGui::EndMenu();
    }
    ImGui::Separator();

    if (ImGui::BeginMenu(MainMenu_Edit.imgui()))
    {



        ImGui::EndMenu();
    }
    ImGui::Separator(); 

    if (ImGui::BeginMenu(MainMenu_View.imgui()))
    {


        ImGui::EndMenu();
    }
    ImGui::Separator();

    if (ImGui::BeginMenu(MainMenu_Help.imgui()))
    {


        ImGui::EndMenu();
    }
    ImGui::Separator();

}