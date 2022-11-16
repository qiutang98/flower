#include "Pch.h"
#include "MainMenu.h"
#include "../Editor.h"

using namespace Flower;
using namespace Flower::UI;

const static std::string MAINMENU_GCloseIcon = ICON_FA_POWER_OFF;
const static std::string MAINMENU_GNoneIcon  = ICON_NONE;

void MainMenu::build()
{
    if (ImGui::BeginMenuBar())
    {
        if (ImGui::BeginMenu("  FILE  "))
        {
            this->buildFilesCommand();
            ImGui::EndMenu();
        }

        ImGui::Separator();

        if (ImGui::BeginMenu("  EDIT  "))
        {

            ImGui::EndMenu();
        }
        ImGui::Separator();

        if (ImGui::BeginMenu("  VIEW  "))
        {


            ImGui::EndMenu();
        }
        ImGui::Separator();
        if (ImGui::BeginMenu("  HELP  "))
        {


            ImGui::EndMenu();
        }
        ImGui::Separator();
        ImGui::EndMenuBar();
    }
}

void MainMenu::buildFilesCommand()
{
    static const std::string sSelectProjectName = MAINMENU_GNoneIcon + "    Select  Project ";
    if (ImGui::MenuItem(sSelectProjectName.c_str(), NULL, false, true))
    {
        GEditor->getProjectSelect()->setVisible(true);
    }

    ImGui::Separator();

    static const std::string sExitName = MAINMENU_GCloseIcon + "    Exit  ";
    if (ImGui::MenuItem(sExitName.c_str(), NULL, false, true))
    {
        GLFWWindowData::get()->setShouldRun(false);
    }
}