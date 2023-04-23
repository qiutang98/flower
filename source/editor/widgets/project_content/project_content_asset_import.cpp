#include "../project_content.h"
#include "imgui/ui.h"
#include "imgui/region_string.h"
#include "../../editor.h"
#include "../../editor_asset.h"

#include <nfd.h>
#include <utfcpp/source/utf8/cpp17.h>
#include <imgui/imgui.h>

using namespace engine;
using namespace engine::ui;

bool ProjectContentWidget::importAssetAction(EAssetType type)
{
	std::filesystem::path saveFileFolder = m_editor->getProjectRootPathUtf16();

	const auto& typeNameMap = EditorAsset::get()->getTypeNameMap();
	const auto& assetMap = EditorAsset::get()->getRegisterMap();

	nfdpathset_t pathSet;
	nfdresult_t result = NFD_OpenDialogMultiple(assetMap.at(typeNameMap.at(type)).rawResourceExtensions, NULL, &pathSet);
	if (result != NFD_OKAY)
	{
		return false;
	}

	m_assetImportPayload.cleanState();
	m_assetImportPayload.type = type;

	for (size_t i = 0; i < NFD_PathSet_GetCount(&pathSet); ++i)
	{
		nfdchar_t* path = NFD_PathSet_GetPath(&pathSet, i);
		std::string utf8Path = path;

		m_assetImportPayload.srcPaths.push_back(utf8::utf8to16(utf8Path));
		m_assetImportPayload.savePaths.push_back( m_activeFolder / m_assetImportPayload.srcPaths.back().filename().replace_extension());
	}

	NFD_PathSet_Free(&pathSet);
	return true;
}

void ProjectContentWidget::drawImageImportModalContent()
{
	if (m_assetImportPayload.srcPaths.empty())
	{
		return;
	}

	if (!m_assetImportPayload.bConfigInit)
	{
		m_assetImportPayload.imageConfigs.clear();
		m_assetImportPayload.imageConfigs.resize(m_assetImportPayload.srcPaths.size());

		m_assetImportPayload.bConfigInit = true;
	}

	for (size_t i = 0; i < m_assetImportPayload.srcPaths.size(); i++)
	{
		auto& config = m_assetImportPayload.imageConfigs.at(i);

		ImGui::Spacing();
		ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 0));
		ImGui::PushID((int)i);
		ImGui::Indent();
		{
			std::string utf8Name = utf8::utf16to8(m_assetImportPayload.srcPaths[i].u16string());
			std::string saveUtf8 = utf8::utf16to8(m_assetImportPayload.savePaths[i].u16string());

			ImGui::TextDisabled(std::format("Load from: {}",utf8Name).c_str());
			ImGui::TextDisabled(std::format("Save to: {}", saveUtf8).c_str());
			ImGui::Spacing();

			if (ImGui::BeginTable("##ConfigTable", 2, ImGuiTableFlags_Resizable | ImGuiTableFlags_Borders))
			{
				ImGui::TableNextRow(); ImGui::TableNextColumn(); ImGui::Text("Is sRGB"); ImGui::TableNextColumn(); ImGui::Checkbox("##SRGB", &config.bSRGB);
				ImGui::TableNextRow(); ImGui::TableNextColumn(); ImGui::Text("Build Mipmap"); ImGui::TableNextColumn(); ImGui::Checkbox("##MipMap", &config.bGenerateMipmap);
				ImGui::TableNextRow(); ImGui::TableNextColumn(); ImGui::Text("Compressed"); ImGui::TableNextColumn(); ImGui::Checkbox("##Compressed", &config.bCompressed);
				ImGui::TableNextRow(); ImGui::TableNextColumn(); ImGui::Text("Alpha Cutoff"); ImGui::TableNextColumn(); ImGui::DragFloat("##AlphaCutoff", &config.cutoffAlpha, 0.01f, 0.0f, 1.0f);
				ImGui::TableNextRow(); ImGui::TableNextColumn(); ImGui::Text("Half Fixed"); ImGui::TableNextColumn(); ImGui::Checkbox("##HalfFixed", &config.bHalfFixed);
				ImGui::TableNextRow(); ImGui::TableNextColumn(); ImGui::Text("Exr"); ImGui::TableNextColumn(); ImGui::Checkbox("##Exr", &config.bExr);

				enum ChannelMode
				{
					Mode_RGBA,
					Mode_R,
					Mode_G,
					Mode_B,
					Mode_A,
					Mode_RGB,
				};
				int mode = (int)config.channel;
				if (ImGui::RadioButton("RGBA", mode == Mode_RGBA)) { mode = (int)Mode_RGBA; } ImGui::SameLine();
				if (ImGui::RadioButton("RGB", mode == Mode_RGB)) { mode = (int)Mode_RGB; } ImGui::SameLine();
				if (ImGui::RadioButton("R", mode == Mode_R)) { mode = (int)Mode_R; } ImGui::SameLine();
				if (ImGui::RadioButton("G", mode == Mode_G)) { mode = (int)Mode_G; } ImGui::SameLine();
				if (ImGui::RadioButton("B", mode == Mode_B)) { mode = (int)Mode_B; } ImGui::SameLine();
				if (ImGui::RadioButton("A", mode == Mode_A)) { mode = (int)Mode_A; } ImGui::SameLine();
				config.channel = (engine::AssetTexture::ImportConfig::EChannel)mode;
				ImGui::EndTable();
			}
		}
		ImGui::Unindent();
		ImGui::PopStyleVar();
		ImGui::PopID();
		
		ImGui::NewLine();
		ImGui::Separator();
	}

}

void ProjectContentWidget::drawStaticMeshImportModalContent()
{
	if (m_assetImportPayload.srcPaths.empty())
	{
		return;
	}

	if (!m_assetImportPayload.bConfigInit)
	{
		m_assetImportPayload.staticmeshConfigs.clear();
		m_assetImportPayload.staticmeshConfigs.resize(m_assetImportPayload.srcPaths.size());

		m_assetImportPayload.bConfigInit = true;
	}

	for (size_t i = 0; i < m_assetImportPayload.srcPaths.size(); i++)
	{
		auto& config = m_assetImportPayload.staticmeshConfigs.at(i);

		ImGui::Spacing();
		ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 0));
		ImGui::PushID((int)i);
		ImGui::Indent();
		{
			std::string utf8Name = utf8::utf16to8(m_assetImportPayload.srcPaths[i].u16string());
			std::string saveUtf8 = utf8::utf16to8(m_assetImportPayload.savePaths[i].u16string());

			ImGui::TextDisabled(std::format("Load from: {}", utf8Name).c_str());
			ImGui::TextDisabled(std::format("Save to: {}", saveUtf8).c_str());
			ImGui::Spacing();

		}
		ImGui::Unindent();
		ImGui::PopStyleVar();
		ImGui::PopID();

		ImGui::Spacing();
		ImGui::Spacing();
		ImGui::Separator();
	}
}


void ProjectContentWidget::executeImport()
{
	// Pre-return if no config state.
	if (!m_assetImportPayload.bConfigInit || m_assetImportPayload.srcPaths.empty())
	{
		return;
	}

	if (m_assetImportPayload.bExecuteImportConfigs)
	{
		ImGui::OpenPopup("##AssetImportProgressDetail");

		if (!m_importProgress.logHandle.isValid())
		{
			m_importProgress.logHandle = LoggerSystem::getDefaultLoggerSystem()->pushCallback([&](const std::string& info, ELogType type)
			{
				m_importProgress.logItems.push_back(info);
				if (static_cast<uint32_t>(m_importProgress.logItems.size()) >= 60)
				{
					m_importProgress.logItems.pop_front();
				}
			});
		}
	}

	ImGuiIO& io = ImGui::GetIO();
	ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.5f), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
	if (ImGui::BeginPopupModal("##AssetImportProgressDetail", NULL, ImGuiWindowFlags_AlwaysAutoResize))
	{
		// Submit import task.
		if (m_assetImportPayload.bExecuteImportConfigs)
		{
			const auto loop = [this](const size_t loopStart, const size_t loopEnd)
			{
				for (size_t i = loopStart; i < loopEnd; ++i)
				{
					auto& pl = m_assetImportPayload;
					if (pl.type == EAssetType::Texture)
					{
						AssetTexture meta{};
						if (AssetTexture::buildFromConfigs(pl.imageConfigs[i], m_editor->getProjectRootPathUtf16(), pl.savePaths[i], pl.srcPaths[i], meta))
						{
							LOG_TRACE("Import image from {} to {}.", utf8::utf16to8(pl.srcPaths[i].u16string()), utf8::utf16to8(pl.savePaths[i].u16string()));
						}
						else
						{
							LOG_ERROR("Fail to import image from {} to {}.", utf8::utf16to8(pl.srcPaths[i].u16string()), utf8::utf16to8(pl.savePaths[i].u16string()));
						}
					}
					else if (pl.type == EAssetType::StaticMesh)
					{
						AssetStaticMesh meta{};
						if (AssetStaticMesh::buildFromConfigs(pl.staticmeshConfigs[i], m_editor->getProjectRootPathUtf16(), pl.savePaths[i], pl.srcPaths[i], meta))
						{
							LOG_TRACE("Import static mesh from {} to {}.", utf8::utf16to8(pl.srcPaths[i].u16string()), utf8::utf16to8(pl.savePaths[i].u16string()));
						}
						else
						{
							LOG_ERROR("Fail to import static mesh from {} to {}.", utf8::utf16to8(pl.srcPaths[i].u16string()), utf8::utf16to8(pl.savePaths[i].u16string()));
						}
					}
					else
					{
						CHECK_ENTRY();
					}
				}
			};
			m_assetImportPayload.executeFutures = ThreadPool::getDefault()->parallelizeLoop(0, m_assetImportPayload.srcPaths.size(), loop);
		}
		m_assetImportPayload.bExecuteImportConfigs = false;

		ImGui::Indent();
		ImGui::Text("Asset  Importing ...    ");
		ImGui::SameLine();

		float progress = m_assetImportPayload.executeFutures.getProgress();
		ImGui::ProgressBar(progress, ImVec2(0.0f, 0.0f));

		ImGui::Unindent();
		ImGui::Separator();

		ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
		for (int i = 0; i < m_importProgress.logItems.size(); i++)
		{
			ImGui::Selectable(m_importProgress.logItems[i].c_str());
		}
		ImGui::PopStyleColor();

		if (progress > 0.99f)
		{
			m_assetImportPayload.executeFutures.wait();
			ImGui::CloseCurrentPopup();

			// All import ready. clean state.
			m_assetImportPayload.cleanState();

			// Rebuild project tree.
			setupProject(m_editor->getProjectFilePathUtf16());
			m_assetSystem->setupProject(m_editor->getProjectFilePathUtf16());

			if(m_importProgress.logHandle.isValid())
			{
				LoggerSystem::getDefaultLoggerSystem()->popCallback(m_importProgress.logHandle);
				m_importProgress.logHandle.reset();
				m_importProgress.logItems.clear();
			}
		}

		ImGui::EndPopup();
	}
}

void ProjectContentWidget::drawAssetImportModal()
{
	if (m_assetImportPayload.srcPaths.empty())
	{
		return;
	}

	if (m_assetImportPayload.bShouldDrawImportModal)
	{
		m_assetImportPayload.bShouldDrawImportModal = false;
		ImGui::OpenPopup("##AssetImportDetail");
	}

    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

    if (ImGui::BeginPopupModal("##AssetImportDetail", NULL, ImGuiWindowFlags_AlwaysAutoResize))
    {
		if (m_assetImportPayload.type == EAssetType::Texture)
		{
			drawImageImportModalContent();
		}
		else if (m_assetImportPayload.type == EAssetType::StaticMesh)
		{
			drawStaticMeshImportModalContent();
		}
		else
		{
			CHECK_ENTRY();
		}

        if (ImGui::Button("OK", ImVec2(120, 0))) 
        { 
			m_assetImportPayload.bExecuteImportConfigs = true;
            ImGui::CloseCurrentPopup(); 
        }

        ImGui::SetItemDefaultFocus();
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(120, 0))) 
        { 
            ImGui::CloseCurrentPopup(); 
        }
        ImGui::EndPopup();
    }
}