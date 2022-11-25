#include "Pch.h"
#include "ProjectSelect.h"
#include "../Editor.h"

using namespace Flower;
using namespace Flower::UI;

static const std::string PROJECTSELECT_SelectProjectIcon = ICON_FA_ANCHOR_CIRCLE_XMARK;
static const std::string PROJECTSELECT_SwitchProjectIcon = ICON_FA_ANCHOR_CIRCLE_EXCLAMATION;

WidgetProjectSelect::WidgetProjectSelect()
    : Widget("SelectProject")
{
    m_bShow = false;
}

void WidgetProjectSelect::onTick(const RuntimeModuleTickData& tickData)
{
    if (!m_bActive)
    {
        return;
    }

    static bool bUseWorkArea = true;
    static ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings;
    const ImGuiViewport* viewport = ImGui::GetMainViewport();

    ImGui::SetNextWindowPos(bUseWorkArea ? viewport->WorkPos : viewport->Pos);
    ImGui::SetNextWindowSize(bUseWorkArea ? viewport->WorkSize : viewport->Size);

    const float footerHeightToReserve =
        ImGui::GetStyle().ItemSpacing.y +
        ImGui::GetFrameHeightWithSpacing() * (m_bActiveCreateDetail ? 3.6f : 2.4f);

    if (ImGui::Begin("ProjectSelectedWindow", &m_bActive, flags))
    {
        ImGui::Spacing();

        ImGui::BeginChild("ContenProjectSetup", { ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y - footerHeightToReserve }, true);
        if (ProjectContext::get()->isValid())
        {
            std::stringstream ss;
            ss << PROJECTSELECT_SwitchProjectIcon;
            ss << "\t EXIST  ACTIVE  PROJECT:  \n";
            ss << "\t\t\t\t [ " << ProjectContext::get()->project.getName() << " ]";
            ss << "\n\t\t SELECTING NEW PROJECT!\n\n\n";

            ImGui::Text(ss.str().c_str());
        }
        else
        {
            static std::string cOutText =
                PROJECTSELECT_SelectProjectIcon + "\t NO  ACTIVE  PROJECT!\n\n"
                "\t\t SELECT NEW PROJECT PLEASE!\n\n\n";

            ImGui::Text(cOutText.c_str());
        }
        ImGui::EndChild();
        ImGui::Spacing();

        if (ProjectContext::get()->isValid())
        {
            if (ImGui::Button("Close"))
            {
                m_bActive = false;
                m_bActiveCreateDetail = false;
                ImGui::CloseCurrentPopup();
            }
        }
        else
        {
            ImGui::BeginDisabled();
            ImGui::Button("Close");
            ImGui::EndDisabled();
        }
        ImGui::SameLine();
        if (m_bActiveCreateDetail)
        {
            if (ImGui::Button("Back"))
            {
                m_bActiveCreateDetail = false;
            }
        }
        else
        {
            ImGui::BeginDisabled();
            ImGui::Button("Back");
            ImGui::EndDisabled();
        }
        ImGui::Spacing();

        if (!m_bActiveCreateDetail)
        {
            // Try load project.
            if (ImGui::Button("  Load  Project  "))
            {
                if (loadProject())
                {
                    m_bActive = false;
                    ImGui::CloseCurrentPopup();
                }
            }
            ImGui::SetItemDefaultFocus();

            // Try create project.
            ImGui::SameLine();
            if (ImGui::Button("  Create  Project  "))
            {
                m_bActiveCreateDetail = true;
            }
        }

        if (m_bActiveCreateDetail)
        {
            ImGui::InputText("Path  ", m_createProjectPath, GCreateProjectPathSize);
            ImGui::SameLine();
            if (ImGui::Button("     .......     "))
            {
                nfdchar_t* outPath = NULL;
                nfdresult_t result = NFD_PickFolder(NULL, &outPath);
                if (result == NFD_OKAY)
                {
                    std::string path = outPath;
                    if (!path.empty())
                    {
                        std::copy(path.begin(), path.end(), m_createProjectPath);
                        m_createProjectPath[path.size()] = '\0';
                    }

                    free(outPath);
                }
            }

            ImGui::InputText("Name", m_createProjectName, GCreateProjectPathSize);
            ImGui::SameLine();

            std::string projectName = m_createProjectName;
            std::string folderName = m_createProjectPath;
            std::filesystem::path fn = folderName;

            bool bValidCanCreate = false;
            if (!projectName.empty() && !folderName.empty())
            {
                if (std::filesystem::exists(fn) && !std::filesystem::exists(fn / projectName))
                {
                    if (std::all_of(projectName.begin(), projectName.end(), [](char c)
                        {
                            return
                                (c >= 'a' && c <= 'z') ||
                                (c >= 'A' && c <= 'Z') ||
                                (c >= '0' && c <= '9');
                        }))
                    {
                        bValidCanCreate = true;
                    }
                }
            }

            if (!bValidCanCreate)
            {
                ImGui::BeginDisabled();
            }
            if (ImGui::Button("  Create  "))
            {
                createProject(fn);

                m_bActiveCreateDetail = false;
                m_bActive = false;
                ImGui::CloseCurrentPopup();
            }
            if (!bValidCanCreate)
            {
                ImGui::EndDisabled();
            }
        }
    }
    ImGui::End();
}

bool WidgetProjectSelect::loadProject()
{
    bool bResult = false;

    std::string readPathString;
    nfdchar_t* readPath = NULL;
    nfdresult_t result = NFD_OpenDialog("flower", NULL, &readPath);
    if (result == NFD_OKAY)
    {
        readPathString = readPath;
        free(readPath);
    }
    else if (result == NFD_CANCEL)
    {
        return false;
    }

    std::filesystem::path fp{ readPathString };

    if (!fp.empty())
    {
        if (std::filesystem::exists(fp))
        {
            // Load project info.
            std::ifstream is(fp);
            cereal::JSONInputArchive archive(is);
            archive(ProjectContext::get()->project);

            bResult = GEditor->setProjectPath(fp);
        }
    }

    return bResult;
}

void WidgetProjectSelect::createProject(const std::filesystem::path& path)
{
    std::filesystem::path fn = path;

    // Create new project and set to global item.
    std::string projectName = m_createProjectName;
    ProjectContext::get()->project = Project(projectName);

    fn /= projectName;
    std::filesystem::create_directory(fn);

    fn /= projectName;
    std::string projectFilePath = fn.string() + ".flower";

    // Also archive.
    saveActiveProject(projectFilePath);

    // This process should always sucess.
    CHECK(GEditor->setProjectPath(projectFilePath));
}
