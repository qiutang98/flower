#include "project_content.h"
#include "imgui/ui.h"
#include "imgui/region_string.h"
#include "../editor.h"

using namespace engine;
using namespace engine::ui;

RegionStringInit ProjectContent_Title("ProjectContent_Title", "Content", "ProjectContent");
const static std::string ICON_PROJECT_CONTENT = ICON_FA_FOLDER_CLOSED;

ProjectContentWidget::ProjectContentWidget(Editor* editor)
	: Widget(editor, "Content")
{

}


void ProjectContentWidget::onInit()
{
	m_name = combineIcon(ProjectContent_Title, ICON_PROJECT_CONTENT);
	m_flags = ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoScrollbar;
}

void ProjectContentWidget::onRelease()
{

}

void ProjectContentWidget::onTick(const RuntimeModuleTickData& tickData, VulkanContext* context)
{

}


void ProjectContentWidget::onVisibleTick(const RuntimeModuleTickData& tickData)
{
	ImGui::Separator();

	// Draw content menu.
	drawMenu();


	ImGui::Separator();
	ImGui::TextDisabled("Working Project %s and Working Path: %s.", m_editor->getProjectNameUtf8().c_str(), m_editor->getProjectRootPathUtf8().c_str());

	ImGui::SameLine();
	ImGui::Text("Inspecting folder path: %s.", utf8::utf16to8(m_activeFolder.u16string()).c_str());



	drawContent();


	drawAssetImportModal();
	executeImport();

}

VkDescriptorSet ProjectAssetTreeEntry::getSet(Editor* editor, ImVec2& uv0, ImVec2& uv1)
{
	VkDescriptorSet result = m_drawDetail.set;

	if (m_drawDetail.set == VK_NULL_HANDLE)
	{
		m_drawDetail.uv0 = { 0.0f, 0.0f };
		m_drawDetail.uv1 = { 1.0f, 1.0f };
		static const float kUvScale = 0.02f;

		m_drawDetail.uv0.x = -kUvScale;
		m_drawDetail.uv1.x = 1.0f + kUvScale;
		m_drawDetail.uv0.y = -kUvScale;
		m_drawDetail.uv1.y = 1.0f + kUvScale;

		if (m_bFolder)
		{
			m_drawDetail.set = editor->getClampToTransparentBorderSet(editor->getFolderImage());

			result = m_drawDetail.set;
		}
		else
		{
			m_drawDetail.set = editor->getClampToTransparentBorderSet(editor->getFileImage());
			result = m_drawDetail.set;

			// Copy path.
			auto path = m_path;
			const auto extension = m_path.extension().string();
			if (isEngineMetaAsset(extension))
			{
				const auto relativePath = buildRelativePathUtf8(editor->getProjectRootPathUtf16(), path.replace_extension());
				if (auto asset = editor->getAssetSystem()->getAssetByRelativeMap(relativePath))
				{
					if (asset->existSnapshot())
					{
						if (auto image = asset->getOrCreateLRUSnapShot(editor->getContext()))
						{
							if (image->isAssetReady())
							{
								m_drawDetail.set = editor->getClampToTransparentBorderSet(&image->getImage());


								const auto w = asset->getSnapshotWidth();
								const auto h = asset->getSnapshotHeight();

								if (w < h)
								{
									m_drawDetail.uv0.x = 0.0f - (1.0f - float(w) / float(h)) * 0.5f;
									m_drawDetail.uv1.x = 1.0f + (1.0f - float(w) / float(h)) * 0.5f;

									m_drawDetail.uv0.y = -kUvScale;
									m_drawDetail.uv1.y = 1.0f + kUvScale;

								}
								else if (w > h)
								{
									m_drawDetail.uv0.y = 0.0f - (1.0f - float(h) / float(w)) * 0.5f;
									m_drawDetail.uv1.y = 1.0f + (1.0f - float(h) / float(w)) * 0.5f;

									m_drawDetail.uv0.x = -kUvScale;
									m_drawDetail.uv1.x = 1.0f + kUvScale;
								}
							}
							else
							{
								m_drawDetail.set = VK_NULL_HANDLE;
								result = editor->getClampToTransparentBorderSet(editor->getFileImage());
							}
						}
					}
					else
					{
						if (isAssetStaticMeshMeta(extension))
						{
							m_drawDetail.set = editor->getClampToTransparentBorderSet(editor->getStaticMeshImage());
						}
						else if (isAssetMaterialMeta(extension))
						{
							m_drawDetail.set = editor->getClampToTransparentBorderSet(editor->getMaterialImage());
						}
						else if (isAssetSceneMeta(extension))
						{
							m_drawDetail.set = editor->getClampToTransparentBorderSet(editor->getSceneImage());
						}
					}
				}
				else
				{
					m_drawDetail.set = VK_NULL_HANDLE;
					result = editor->getClampToTransparentBorderSet(editor->getFileImage());
				}
			}
		}
	}

	uv0 = m_drawDetail.uv0;
	uv1 = m_drawDetail.uv1;

	return result;
}

void ProjectAssetTreeEntry::buildTreeRecursive()
{
	if (m_bFolder)
	{
		for (const auto& entry : std::filesystem::directory_iterator(m_path))
		{
			const bool bFolder = std::filesystem::is_directory(entry);
			auto u16FileNameString = entry.path().filename().replace_extension().u16string();

			if (bFolder || isEngineMetaAsset(entry.path().extension().string()))
			{
				auto child = std::make_shared<ProjectAssetTreeEntry>(utf8::utf16to8(u16FileNameString), bFolder, entry.path(), shared_from_this());
				m_children.push_back(child);

				child->buildTreeRecursive();
			}
		}

		std::sort(m_children.begin(), m_children.end(), [](const auto& A, const auto& B)
		{
			if ((A->isFoleder() && B->isFoleder()) || (!A->isFoleder() && !B->isFoleder()))
			{
				return A->getNameUtf8() < B->getNameUtf8();
			}

			return A->isFoleder();
		});
	}
}
