#include "../project_content.h"
#include "imgui/ui.h"
#include "imgui/region_string.h"
#include "../../editor.h"
#include "../../editor_asset.h"
#include <imgui/imgui.h>


using namespace engine;
using namespace engine::ui;

const std::string ProjectContentWidget::kAssetDragDropName = "ContentAssetDragDrops";


ProjectAssetTree* ProjectContentWidget::getProjectAssetTree()
{
	if(m_projectAssetTree == nullptr) 
	{
		m_projectAssetTree = std::make_unique<ProjectAssetTree>();
	}

	return m_projectAssetTree.get();
}

void ProjectContentWidget::drawContent()
{
	const float footerHeightToReserve = ImGui::GetTextLineHeight() * 1.2f;

	// Reset 
	m_treeviewHoverEntry.reset();

	if (ImGui::BeginTable("AssetContentTable", 2, ImGuiTableFlags_BordersInner | ImGuiTableFlags_Resizable))
	{
		ImGui::TableNextRow();
		ImGui::TableSetColumnIndex(0);
		{
			ImGui::PushID("##ContentViewItemTreeView");
			ImGui::BeginChild("ScrollingRegion", ImVec2(0, -footerHeightToReserve), false, ImGuiWindowFlags_HorizontalScrollbar);

			drawContentTreeView(getProjectAssetTree()->getRoot());

			ImGui::EndChild();

			if (ImGui::IsItemClicked() && (!m_treeviewHoverEntry.lock()) && (!ImGui::GetIO().KeyCtrl))
			{
				setActiveEntry(getProjectAssetTree()->getRoot());
			}

			ImGui::PopID();
		}

		ImGui::TableSetColumnIndex(1);
		{
			ImGui::PushID("##ContentViewItemInspector");
			ImGui::BeginChild("ScrollingRegion2", ImVec2(0, -footerHeightToReserve), false);

			drawContentSnapshot();

			ImGui::EndChild();
			ImGui::PopID();
		}

		ImGui::EndTable();
	}

	ImGui::Separator();
	size_t itemNum = 0;
	if (auto entry = m_workingEntry.lock())
	{
		itemNum = entry->getChildren().size();
	}
	ImGui::Text("  %d  items.", itemNum);
}

// TODO: Large tree view performance optimize, use clipper.
void ProjectContentWidget::drawContentTreeView(std::shared_ptr<ProjectAssetTreeEntry> entry)
{
	// Should we draw with tree node.
	const bool bTreeNode = entry->isFoleder() && (!entry->isChildrenEmpty());

	// Get node flags.
	ImGuiTreeNodeFlags nodeFlags = ImGuiTreeNodeFlags_AllowItemOverlap | ImGuiTreeNodeFlags_SpanFullWidth;
	nodeFlags |= bTreeNode ? ImGuiTreeNodeFlags_OpenOnArrow : ImGuiTreeNodeFlags_Leaf;

	if (m_editor->getAssetSelections().isSelected(entry->getPath())) 
	{
		nodeFlags |= ImGuiTreeNodeFlags_Selected;
	}

	// Add icon decorate.
	std::string showName = entry->getNameUtf8();
	if (entry->isFoleder())
	{
		if (entry->isChildrenEmpty())
		{
			showName = ICON_FA_FOLDER_CLOSED"  " + showName;
		}
		else
		{
			bool bOpen = entry->isFolderOpen();
			if (bOpen)
			{
				showName = ICON_FA_FOLDER_OPEN"  " + showName;
			}
			else
			{
				showName = ICON_FA_FOLDER_CLOSED"  " + showName;
			}
		}
	}
	else
	{
		showName = ICON_FA_FILE"  " + showName;
	}

	// Draw tree node.
	bool bNodeOpen = ui::treeNodeEx(entry->getNameUtf8().c_str(), showName.c_str(), nodeFlags);
	entry->setFolderOpenState(bNodeOpen);

	// Action tick.
	const auto bLeftClick = ImGui::IsMouseClicked(0);
	const auto bRightClick = ImGui::IsMouseClicked(1);
	const auto bDoubleClick = ImGui::IsMouseDoubleClicked(0);

	const bool bItemHover = ImGui::IsItemHovered(ImGuiHoveredFlags_RectOnly);

	if (ImGui::IsMouseClicked(0) && bItemHover && ImGui::GetIO().KeyCtrl)
	{
		if (m_editor->getAssetSelections().isSelected(entry->getPath()))
		{
			m_editor->getAssetSelections().removeSelect(entry->getPath());
		}
		else
		{
			m_editor->getAssetSelections().addSelected(entry->getPath());
		}
	}
	// Switch active entry.
	else if (ImGui::IsItemClicked())
	{
		setActiveEntry(entry);
	}

	if (bItemHover)
	{
		m_treeviewHoverEntry = entry;
	}

	// Recursive draw.
	if (bNodeOpen)
	{
		if (bTreeNode)
		{
			for (const auto& child : entry->getChildren())
			{
				drawContentTreeView(child);
			}
		}
		ImGui::TreePop();
	}
}

// Set active folder.
void ProjectContentWidget::setActiveEntry(std::shared_ptr<ProjectAssetTreeEntry> entry)
{
	m_editor->getAssetSelections().clearSelections();
	m_editor->getAssetSelections().addSelected(entry->getPath());

	if (entry->isFoleder())
	{
		m_workingEntry = entry;
		m_activeFolder = entry->getPath();
	}
	else
	{
		m_workingEntry = entry->getParent();
		m_activeFolder = m_workingEntry.lock()->getPath();
	}
}

// Setup project.
void ProjectContentWidget::setupProject(const std::filesystem::path& path)
{
	getProjectAssetTree()->setupProject(path);

	// Then update working entry.
	m_workingEntry = m_projectAssetTree->getRoot();
	m_activeFolder = m_projectAssetTree->getRoot()->getPath();
}

void ProjectContentWidget::drawContentSnapshot()
{
	auto workingEntry = m_workingEntry.lock();
	if (!workingEntry)
	{
		// Pre-return if no working entry.
		return;
	}
	const auto& children = workingEntry->getChildren();
	const size_t inspectItemNum = children.size();

	const auto availRegion = ImGui::GetContentRegionAvail();
	const float itemDimSize = ImGui::GetTextLineHeightWithSpacing() * m_snapshotItemIconSize;
	const uint32_t drawItemPerRow = uint32_t(math::max(1.0f, availRegion.x / itemDimSize - 1.0f));

	const size_t minDrawRowNum = size_t(math::max(1.0f, math::ceil(availRegion.y / itemDimSize)));
	const uint32_t drawRowNum = uint32_t(math::max(minDrawRowNum, inspectItemNum / drawItemPerRow + 1));


	static const ImGuiTableFlags flags = ImGuiTableFlags_ScrollY | ImGuiTableFlags_Hideable | ImGuiTableFlags_NoClip | ImGuiTableFlags_NoBordersInBody;

	if (ImGui::BeginTable("table_scrolly_snapshot", drawItemPerRow, flags, availRegion))
	{
		ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 1.0f);
		for (size_t i = 1; i < drawItemPerRow; i++)
		{
			ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, itemDimSize);
		}

		ImGuiListClipper clipper;
		clipper.Begin(int(drawRowNum));
		while (clipper.Step())
		{
			for (size_t row = clipper.DisplayStart; row < clipper.DisplayEnd; row++)
			{
				ImGui::TableNextRow();
				ImGui::PushID(int(row));

				for (size_t colum = 1; colum < drawItemPerRow; colum++)
				{
					size_t drawId = row * (drawItemPerRow - 1) + (colum - 1);

					if (drawId < inspectItemNum)
					{
						ImGui::TableSetColumnIndex(int(colum));
						drawItemSnapshot(itemDimSize, children.at(drawId));
					}
				}
				ImGui::PopID();
			}
		}
		ImGui::EndTable();
	}

}

void ProjectContentWidget::drawItemSnapshot(float drawDimSize, std::shared_ptr<ProjectAssetTreeEntry> entry)
{
	static std::hash<uint64_t> haser;
	float textH = ImGui::GetTextLineHeightWithSpacing();

	const bool bItemSeleted = m_editor->getAssetSelections().isSelected(entry->getPath());

	ImGui::PushID(int(haser(entry->getRuntimeUUID())));
	ImGui::BeginChild(entry->getNameUtf8().c_str(), { drawDimSize ,  drawDimSize + textH * 3.0f }, false, ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoScrollbar);
	ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 0);
	{
		
		bool bItemHover = ImGui::IsMouseHoveringRect(ImGui::GetCursorScreenPos(),
			ImVec2(ImGui::GetCursorScreenPos().x + drawDimSize, ImGui::GetCursorScreenPos().y + drawDimSize));

		if (bItemHover)
		{
			// Open asset.
			if (ImGui::IsMouseDoubleClicked(0))
			{
				if (entry->isFoleder())
				{
					setActiveEntry(entry);
				}
				else
				{
					auto copyPath = entry->getPath();
					if (isAssetSceneMeta(copyPath.extension().string().c_str()))
					{
						// Save assets when need open new scene.
						m_editor->saveDirtyAssetActions();
						m_sceneManager->loadScene(copyPath);
						m_editor->setTitleName();
					}
					else
					{
						const auto relativePath = buildRelativePathUtf8(m_editor->getProjectRootPathUtf16(), copyPath.replace_extension());
						m_editor->getAssetConfigManager()->openWidget(utf8::utf8to16(relativePath));
					}
				}
			}
		}

		if (ImGui::IsMouseClicked(0) && bItemHover)
		{
			if (ImGui::GetIO().KeyCtrl)
			{
				if (m_editor->getAssetSelections().isSelected(entry->getPath()))
				{
					m_editor->getAssetSelections().removeSelect(entry->getPath());
				}
				else
				{
					m_editor->getAssetSelections().addSelected(entry->getPath());
				}
			}
			else
			{
				m_editor->getAssetSelections().clearSelections();
				m_editor->getAssetSelections().addSelected(entry->getPath());
			}
		}

		if (bItemSeleted && ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID))
		{
			m_dragDropAssets.selectAssets.clear();
			for (const auto& path : m_editor->getAssetSelected())
			{
				m_dragDropAssets.selectAssets.insert(path);
			}

			ImGui::SetDragDropPayload(kAssetDragDropName.c_str(), (void*)&m_dragDropAssets, sizeof(void*));

			const auto& filterMaps = EditorAsset::get()->getRegisterMap();
			const auto& typeMaps = EditorAsset::get()->getTypeNameMap();

			for (const auto& id : m_dragDropAssets.selectAssets)
			{
				std::string showName = utf8::utf16to8(id.u16string());
				ImGui::Text(showName.c_str());
			}

			ImGui::EndDragDropSource();
		}

		// Decorated fill
		{
			ImGui::GetWindowDrawList()->AddRectFilled(
				ImGui::GetCursorScreenPos(),
				ImVec2(ImGui::GetCursorScreenPos().x + drawDimSize, ImGui::GetCursorScreenPos().y + drawDimSize + textH * 3.0f),
				bItemSeleted ? IM_COL32(88, 150, 250, 81) : IM_COL32(51, 51, 51, 190));

			ImGui::GetWindowDrawList()->AddRect(
				ImGui::GetCursorScreenPos(),
				ImVec2(ImGui::GetCursorScreenPos().x + drawDimSize, ImGui::GetCursorScreenPos().y + drawDimSize + textH * 3.0f),
				IM_COL32(255, 255, 255, 39));

			ImGui::GetWindowDrawList()->AddRect(
				ImGui::GetCursorScreenPos(),
				ImVec2(ImGui::GetCursorScreenPos().x + drawDimSize, ImGui::GetCursorScreenPos().y + drawDimSize),
				bItemHover ? IM_COL32(250, 244, 11, 255) : IM_COL32(255, 255, 255, 80), bItemHover ? 1.0f : 0.0f, 0, bItemHover ? 1.5f : 1.0f);
		}

		ImVec2 uv0{ 0.0f, 0.0f };
		ImVec2 uv1{ 1.0f, 1.0f };

		VkDescriptorSet set = entry->getSet(m_editor, uv0, uv1);
		ImGui::Image(set, { drawDimSize , drawDimSize }, uv0, uv1);

		const float indentSize = ImGui::GetFontSize() * 0.25f;
		ImGui::Indent(indentSize);
		{
			ImGui::Spacing();
			ImGui::PushTextWrapPos(ImGui::GetCursorPos().x + drawDimSize - indentSize);
			ImGui::Text(entry->getNameUtf8().c_str());
			ImGui::PopTextWrapPos();

		}
		ImGui::Unindent();
	}
	ImGui::PopStyleVar();



	ImGui::EndChild();

	ui::hoverTip(entry->getNameUtf8().c_str());

	ImGui::PopID();


	ImGui::GetWindowDrawList()->AddRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMin(), IM_COL32(88, 150, 250, 81));
}