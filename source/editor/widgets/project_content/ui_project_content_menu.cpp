#include "../project_content.h"
#include "imgui/ui.h"
#include "imgui/region_string.h"
#include "../../editor.h"
#include "../../editor_asset.h"

using namespace engine;
using namespace engine::ui;

static const std::string CONTENTVIEWR_ImportIcon      = std::string(" ") + ICON_FA_FILE_IMPORT + std::string(" Import ");
static const std::string CONTENTVIEWR_NewIcon         = std::string(" ") + ICON_FA_FILE_CIRCLE_PLUS + std::string(" New ");
static const std::string CONTENTVIEWR_SearchAssetIcon = ICON_FA_MAGNIFYING_GLASS;

static const std::string CONTENTVIEWR_SaveIcon = std::string(" ") + ICON_FA_FILE_EXPORT + std::string(" Save ");
static const std::string CONTENTVIEWR_SaveAllIcon        = std::string(" ") + ICON_FA_FILE_SIGNATURE + std::string(" Save All ");

void ProjectContentWidget::drawMenu()
{
	const float sizeLable = ImGui::GetFontSize();
	if (ImGui::BeginTable("Import UIC##", 5))
	{
		ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, sizeLable * 5.0f);
		ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, sizeLable * 4.0f);
		ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, sizeLable * 4.0f);
		ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, sizeLable * 6.0f);
		ImGui::TableSetupColumn("", ImGuiTableColumnFlags_None);

		ImGui::TableNextColumn();

		static const char* kImport = "##XAssetMenu_Import";
		static const char* kCreate = "##XAssetMenu_Create";
		if (ImGui::Button((CONTENTVIEWR_ImportIcon).c_str()))
		{
			ImGui::OpenPopup(kImport);
		}
		hoverTip("Import new asset from disk.");

		if (ImGui::BeginPopup(kImport))
		{
			ImGui::TextDisabled("Import  Assets:");
			ImGui::Separator();

			drawAssetImport();

			ImGui::EndPopup();
		}

		ImGui::TableNextColumn();

		if (ImGui::Button((CONTENTVIEWR_NewIcon).c_str()))
		{
			ImGui::OpenPopup(kCreate);

		}
		hoverTip("Create new asset.");

		if (ImGui::BeginPopup(kCreate))
		{
			ImGui::TextDisabled("Create  Assets:");
			ImGui::Separator();
			
			drawAssetCreate();

			ImGui::EndPopup();
		}

		ImGui::TableNextColumn();

		if (ImGui::Button((CONTENTVIEWR_SaveIcon).c_str()))
		{
			const auto& assets = m_editor->getAssetSelected();
			for (const auto& assetPath : assets)
			{
				auto pathCopy = assetPath;
				const auto relativePath = buildRelativePathUtf8(m_editor->getProjectRootPathUtf16(), pathCopy.replace_extension());
				getAssetSystem()->getAssetByRelativeMap(relativePath)->saveAction();
			}
		}
		hoverTip("Save select asset.");
		ImGui::TableNextColumn();

		if (ImGui::Button((CONTENTVIEWR_SaveAllIcon).c_str()))
		{

		}
		hoverTip("Save all assets.");
		ImGui::TableNextColumn();

		m_filter.Draw((CONTENTVIEWR_SearchAssetIcon).c_str());

		ImGui::EndTable();
	}

}

void ProjectContentWidget::drawAssetImport()
{
	const auto& filterMaps = EditorAsset::get()->getRegisterMap();
	for (const auto& pair : filterMaps)
	{
		if (pair.second.type != EAssetType::Max)
		{
			bool bPreReturn = false;
			ImGui::PushID(pair.first.c_str());
			if (ImGui::Selectable(pair.second.decoratorName.c_str()))
			{
				if (importAssetAction(pair.second.type))
				{
					m_assetImportPayload.bShouldDrawImportModal = true;
				}
				bPreReturn = true;
			}
			ImGui::PopID();

			if (bPreReturn) { return; }
		}
	}
}

void ProjectContentWidget::drawAssetCreate()
{
	const auto& filterMaps = EditorAsset::get()->getRegisterMap();

	for (const auto& pair : filterMaps)
	{
		bool bPreReturn = false;
		if (pair.second.type != EAssetType::Max)
		{
			ImGui::PushID(pair.first.c_str());
			if (ImGui::Selectable(pair.second.decoratorName.c_str()))
			{
				bPreReturn = true;



			}
			ImGui::PopID();

			if (bPreReturn) { return; }
		}
	}
}