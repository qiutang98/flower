#include "content.h"
#include "../builtin_resources.h"
#include "dockspace.h"

#include <asset/asset_texture.h>
#include <asset/asset_manager.h>
#include <nfd.h>
#include <scene/scene_manager.h>

using namespace engine;
using namespace engine::ui;

static const std::string kIconContentImport  = std::string(" ") + ICON_FA_FILE_IMPORT       + std::string(" Import ")  ;
static const std::string kIconContentNew     = std::string(" ") + ICON_FA_FILE_CIRCLE_PLUS  + std::string(" New ")     ;
static const std::string kIconContentSave    = std::string(" ") + ICON_FA_FILE_EXPORT       + std::string(" Save ")    ;
static const std::string kIconContentSaveAll = std::string(" ") + ICON_FA_FILE_SIGNATURE    + std::string(" Save All ");
static const std::string kIconContentSearch  =                    ICON_FA_MAGNIFYING_GLASS                             ; 
static const std::string kIconContentTitle   =                    ICON_FA_FOLDER_CLOSED                                ;


static const std::string kRightClickedMenuName = "##RightClickedMenu";

static const float kMinSnapShotIconSize = 2.0f;
static const float kMaxSnapShotIconSize = 8.0f;

WidgetContent::WidgetContent(size_t index, ProjectContentModel* model)
	: WidgetBase(
		combineIcon("Content", kIconContentTitle).c_str(), 
		combineIcon(combineIndex("Content", index), kIconContentTitle).c_str())
	, m_model(model)
	, m_index(index)
{

}

void WidgetContent::onInit()
{
	m_bShow = false;
	CHECK(getAssetManager()->isProjectSetup());

	setupProject();

	m_onTreeViewRebuildHandle = m_model->onProjectTreeRebuild.addRaw(this, &WidgetContent::onContentTreeRebuild);
}

void WidgetContent::onRelease()
{
	m_model->onProjectTreeRebuild.remove(m_onTreeViewRebuildHandle);
}

void WidgetContent::onTick(const RuntimeModuleTickData& tickData, VulkanContext* context)
{

}

void WidgetContent::setActiveEntry(std::shared_ptr<ProjectAssetTreeEntry> entry)
{
	if (entry->isFoleder())
	{
		m_activeFolder = entry->getPath();
	}
	else
	{
		m_activeFolder = entry->getParent().lock()->getPath();
	}

	// Update selections in model.
	const AssetSelector newSelector(entry);
	m_selections.clear();
	m_selections.add(newSelector);
}

void WidgetContent::drawRightClickedMenu()
{
	CHECK(getSelections().existElement());

	ImGui::TextDisabled("Assets Menu");
	ImGui::Separator();

	static const std::string kNewItemName     = combineIcon("New ...", ICON_NONE);
	static const std::string kImportItemName  = combineIcon("Import ...", ICON_NONE);
	static const std::string kCopyItemName    = combineIcon(" Copy", ICON_FA_COPY);
	static const std::string kPasteItemName   = combineIcon(" Paste", ICON_FA_PASTE);
	static const std::string kDeleteItemName  = combineIcon("Delete", ICON_NONE);

	const bool bSingleSelected = getSelections().getNum() == 1;
	const bool bSingleFolderSelected = getSelections().getSelections().at(0).entry.lock()->isFoleder();

	if (bSingleSelected)
	{
		if (bSingleFolderSelected)
		{
			if (ImGui::BeginMenu(kNewItemName.c_str()))
			{


				ImGui::EndMenu();
			}

			if (ImGui::BeginMenu(kImportItemName.c_str()))
			{
				drawAssetImport();
				ImGui::EndMenu();
			}
		}
	}

	if (ImGui::Selectable(kDeleteItemName.c_str()))
	{


	}

	ImGui::Separator();
	if (ImGui::Selectable(kCopyItemName.c_str()))
	{

	}
	if (ImGui::Selectable(kPasteItemName.c_str()))
	{

	}
}

void WidgetContent::drawAssetImport()
{
	ZoneScoped;

	ImGui::TextDisabled("Import  Assets:");
	ImGui::Separator();

	auto assetList = rttr::type::get<AssetInterface>().get_derived_classes();
	for (auto& assetType : assetList)
	{
		const auto& method = assetType.get_method("uiGetAssetReflectionInfo");
		if (method.is_static() && method.is_valid())
		{
			rttr::variant returnValue = method.invoke({});
			if (returnValue.is_valid() && returnValue.is_type<AssetReflectionInfo>())
			{
				const auto& meta = returnValue.get_value<AssetReflectionInfo>();
				if (meta.importConfig.bImportable)
				{
					ImGui::PushID(assetType.get_name().data());
					if (ImGui::Selectable(meta.decoratedName.c_str()))
					{
						nfdpathset_t pathSet;
						nfdresult_t result = NFD_OpenDialogMultiple(meta.importConfig.importRawAssetExtension.c_str(), NULL, &pathSet);
						if (result == NFD_OKAY)
						{
							const auto count = NFD_PathSet_GetCount(&pathSet);
							if (count > 0)
							{
								auto& contentImport = Editor::get()->getDockSpaceAndMenu()->contentAssetImport;
								if (contentImport.open())
								{
									CHECK(contentImport.importConfigs.empty());
									CHECK(contentImport.typeName.empty());
									contentImport.typeName = assetType.get_name();

									CHECK(meta.importConfig.buildAssetImportConfig);
									for (size_t i = 0; i < NFD_PathSet_GetCount(&pathSet); ++i)
									{
										nfdchar_t* path = NFD_PathSet_GetPath(&pathSet, i);
										std::string utf8Path = path;

										const std::filesystem::path u16Path = utf8::utf8to16(utf8Path);
										auto folderPath = m_activeFolder / u16Path.filename().replace_extension();
										folderPath = buildStillNonExistPath(folderPath);

										auto config = meta.importConfig.buildAssetImportConfig();
										config->path = { u16Path, folderPath };

										contentImport.importConfigs.push_back(config);
									}
								}
							}
							NFD_PathSet_Free(&pathSet);
						}
					}
					ImGui::PopID();
				}
			}
		}
	}
}


void WidgetContent::onVisibleTick(const RuntimeModuleTickData& tickData)
{
	ZoneScoped;

	ImGui::Separator();
	{
		drawMenu(tickData);
	}

	ImGui::Separator();
	const auto& projectConfig = getAssetManager()->getProjectConfig();
	{
		std::string projectName = utf8::utf16to8(projectConfig.projectName);
		std::string rootName = utf8::utf16to8(projectConfig.rootPath);
		std::string activeFolderName = utf8::utf16to8(m_activeFolder.u16string());

		ImGui::TextDisabled("Working project: %s and working path: %s.", projectName.c_str(), rootName.c_str());

		ImGui::SameLine();
		ImGui::Text("Inspecting folder path: %s.", activeFolderName.c_str());
	}

	drawContent(tickData);
}

void WidgetContent::onContentTreeRebuild()
{
	m_selections.clear();
	m_treeviewHoverEntry = {};
}

void WidgetContent::setupProject()
{
	const auto& projectConfig = getAssetManager()->getProjectConfig();
	m_activeFolder = projectConfig.assetPath;


}

void WidgetContent::drawMenu(const RuntimeModuleTickData& tickData)
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

		if (ImGui::Button((kIconContentImport).c_str()))
		{
			ImGui::OpenPopup(kImport);
		}
		hoverTip("Import new asset from disk.");

		if (ImGui::BeginPopup(kImport))
		{
			drawAssetImport();

			ImGui::EndPopup();
		}

		ImGui::TableNextColumn();

		if (ImGui::Button((kIconContentNew).c_str()))
		{
			ImGui::OpenPopup(kCreate);

		}
		hoverTip("Create new asset.");

		if (ImGui::BeginPopup(kCreate))
		{
			ImGui::TextDisabled("Create  Assets:");
			ImGui::Separator();

			// TODO:
			// drawAssetCreate();

			ImGui::EndPopup();
		}

		ImGui::TableNextColumn();

		if (ImGui::Button((kIconContentSave).c_str()))
		{
			// TODO:
			/*
			const auto& assets = m_editor->getAssetSelected();
			for (const auto& assetPath : assets)
			{
				auto pathCopy = assetPath;
				const auto relativePath = buildRelativePathUtf8(m_editor->getProjectRootPathUtf16(), pathCopy.replace_extension());
				getAssetSystem()->getAssetByRelativeMap(relativePath)->saveAction();
			}
			*/
		}
		hoverTip("Save select asset.");
		ImGui::TableNextColumn();

		if (ImGui::Button((kIconContentSaveAll).c_str()))
		{

		}
		hoverTip("Save all assets.");
		ImGui::TableNextColumn();

		m_filter.Draw((kIconContentSearch).c_str());

		ImGui::EndTable();
	}

}

void WidgetContent::drawContentTreeView(std::shared_ptr<ProjectAssetTreeEntry> entry)
{
	// Should we draw with tree node.
	const bool bTreeNode = entry->isFoleder() && (!entry->isChildrenEmpty());

	// Get node flags.
	ImGuiTreeNodeFlags nodeFlags = ImGuiTreeNodeFlags_AllowItemOverlap | ImGuiTreeNodeFlags_SpanFullWidth;
	nodeFlags |= bTreeNode ? ImGuiTreeNodeFlags_OpenOnArrow : ImGuiTreeNodeFlags_Leaf;

	const AssetSelector assetSelector(entry);
	if (getSelections().isSelected(assetSelector))
	{
		nodeFlags |= ImGuiTreeNodeFlags_Selected;
	}

	// Add icon decorate.
	u8str showName = entry->getName();
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
	bool bNodeOpen = ui::treeNodeEx(entry->getName().c_str(), showName.c_str(), nodeFlags);
	entry->setFolderOpenState(bNodeOpen);

	// Action tick.
	if (ImGui::IsItemClicked(0))
	{
		if (ImGui::GetIO().KeyCtrl)
		{
			if (getSelections().isSelected(assetSelector))
			{
				getSelections().remove(assetSelector);
			}
			else
			{
				getSelections().add(assetSelector);
			}
		}
		else
		{
			// Switch active entry.
			setActiveEntry(entry);
		}
	}

	const bool bItemHover = ImGui::IsItemHovered(ImGuiHoveredFlags_RectOnly);
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

void WidgetContent::drawContentSnapShot(std::shared_ptr<ProjectAssetTreeEntry> workingEntry)
{
	const auto& children = workingEntry->getChildren();
	const auto inspectItemNum = children.size();

	const auto availRegion = ImGui::GetContentRegionAvail();
	const float itemDimSize = ImGui::GetTextLineHeightWithSpacing() * m_snapshotItemIconSize;

	const float kPadFirstColumDimX = 1.0f;

	const uint32_t drawItemPerRow = uint32_t(math::max(1.0f, (availRegion.x - kPadFirstColumDimX - ImGui::GetTextLineHeightWithSpacing() * 2.0f) / (itemDimSize + ImGui::GetStyle().ItemSpacing.x)));

	const size_t minDrawRowNum = size_t(math::max(1.0f, math::ceil(availRegion.y / itemDimSize)));
	const uint32_t drawRowNum = uint32_t(math::max(minDrawRowNum, inspectItemNum / drawItemPerRow + 1));


	if (ImGui::BeginTable("##table_scrolly_snapshot", drawItemPerRow + 1, 
		ImGuiTableFlags_ScrollY  |
		ImGuiTableFlags_Hideable |
		ImGuiTableFlags_NoClip   |
		ImGuiTableFlags_NoBordersInBody, availRegion))
	{
		ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, kPadFirstColumDimX);
		for (size_t i = 1; i < drawItemPerRow + 1; i++)
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

				for (size_t colum = 1; colum < drawItemPerRow + 1; colum++)
				{
					size_t drawId = row * drawItemPerRow + (colum - 1);
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


	if (ImGui::IsMouseHoveringRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax()) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl))
	{
		m_snapshotItemIconSize += ImGui::GetIO().MouseWheel;
		m_snapshotItemIconSize = math::clamp(m_snapshotItemIconSize, kMinSnapShotIconSize, kMaxSnapShotIconSize);
	}

	if (ImGui::IsMouseClicked(1) && 
		ImGui::IsMouseHoveringRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax()) && 
		getSelections().existElement())
	{
		ImGui::OpenPopup(kRightClickedMenuName.c_str());
	}
	if (ImGui::BeginPopup(kRightClickedMenuName.c_str()))
	{
		drawRightClickedMenu();
		ImGui::EndPopup();
	}
}

void WidgetContent::drawItemSnapshot(float drawDimSize, std::shared_ptr<ProjectAssetTreeEntry> entry)
{
	const AssetSelector entrySelector(entry);

	static std::hash<uint64_t> haser;
	float textH = ImGui::GetTextLineHeightWithSpacing();

	const bool bItemSeleted = getSelections().isSelected(entrySelector);

	ImGui::PushID(int(haser(entry->getRuntimeUUID())));
	ImGui::BeginChild(
		entry->getName().c_str(), 
		{ drawDimSize ,  drawDimSize + textH * 3.0f }, 
		false, 
		ImGuiWindowFlags_NoScrollWithMouse | 
		ImGuiWindowFlags_NoScrollbar);

	ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 0);
	{
		bool bItemHover = ImGui::IsMouseHoveringRect(ImGui::GetCursorScreenPos(),
			ImVec2(ImGui::GetCursorScreenPos().x + drawDimSize, ImGui::GetCursorScreenPos().y + drawDimSize));

		if (bItemHover)
		{
			// Double clicked
			if (ImGui::IsMouseDoubleClicked(0))
			{
				if (entry->isFoleder())
				{
					setActiveEntry(entry);
				}
				else
				{
					auto copyPath = entry->getPath();
					if (Scene::assetIsScene(copyPath.extension().string().c_str()))
					{
						if (!getAssetManager()->getDirtyAsset<Scene>().empty())
						{
							if (Editor::get()->getDockSpaceAndMenu()->sceneAssetSave.open())
							{
								CHECK(!Editor::get()->getDockSpaceAndMenu()->sceneAssetSave.afterEventAccept);
								Editor::get()->getDockSpaceAndMenu()->sceneAssetSave.afterEventAccept = [copyPath]()
								{
									getSceneManager()->loadScene(copyPath);
								};
							}
						}
						else
						{
							getSceneManager()->loadScene(copyPath);
						}
					}
					else
					{
						// TODO: Asset edit.
						// const auto relativePath = buildRelativePathUtf8(m_editor->getProjectRootPathUtf16(), copyPath.replace_extension());
						// m_editor->getAssetConfigManager()->openWidget(utf8::utf8to16(relativePath));
					}
				}
			}

			// Single click
			if (ImGui::IsMouseClicked(0))
			{
				if (ImGui::GetIO().KeyCtrl)
				{
					if (getSelections().isSelected(entrySelector))
					{
						getSelections().remove(entrySelector);
					}
					else
					{
						getSelections().add(entrySelector);
					}
				}
				else
				{
					getSelections().clear();
					getSelections().add(entrySelector);
				}
			}
		}

		if (bItemSeleted && ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID))
		{
			auto& dragDropAssets = Editor::get()->getDragDropAssets();

			dragDropAssets.selectAssets.clear();
			for (const auto& selector : getSelections().getSelections())
			{
				if (auto entry = selector.entry.lock())
				{
					dragDropAssets.selectAssets.insert(entry->getPath());
				}
			}

			ImGui::SetDragDropPayload(
				Editor::getDragDropAssetsName(), 
				(void*)&dragDropAssets,
				sizeof(void*));

			for (const auto& id : dragDropAssets.selectAssets)
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

		auto set = entry->getSet(uv0, uv1);
		ImGui::Image(set, { drawDimSize , drawDimSize }, uv0, uv1);

		const float indentSize = ImGui::GetFontSize() * 0.25f;
		ImGui::Indent(indentSize);
		{
			ImGui::Spacing();
			ImGui::PushTextWrapPos(ImGui::GetCursorPos().x + drawDimSize - indentSize);
			ImGui::Text(entry->getName().c_str());
			ImGui::PopTextWrapPos();
		}
		ImGui::Unindent();
	}
	ImGui::PopStyleVar();
	ImGui::EndChild();

	ui::hoverTip(entry->getName().c_str());
	ImGui::PopID();
	ImGui::GetWindowDrawList()->AddRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMin(), IM_COL32(88, 150, 250, 81));
}

void WidgetContent::drawContent(const engine::RuntimeModuleTickData& tickData)
{
	const float footerHeightToReserve = ImGui::GetTextLineHeight() * 1.2f;
	const float sizeLable = ImGui::GetFontSize();

	// Reset tree view hovering entry.
	m_treeviewHoverEntry.reset();

	auto workingEntry = m_model->getTree().getEntry(m_activeFolder).lock();

	if (ImGui::BeginTable("AssetContentTable", 2, ImGuiTableFlags_BordersInner | ImGuiTableFlags_Resizable))
	{
		ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, sizeLable * 14.0f);
		ImGui::TableSetupColumn("", ImGuiTableColumnFlags_None);

		// Row #0 is tree view.
		ImGui::TableNextRow();
		ImGui::TableSetColumnIndex(0);
		{
			ImGui::PushID("##ContentViewItemTreeView");
			ImGui::BeginChild("ScrollingRegion", ImVec2(0, -footerHeightToReserve), false, ImGuiWindowFlags_HorizontalScrollbar);
			{
				drawContentTreeView(m_model->getTree().getRoot());
			}
			ImGui::EndChild();
			if (ImGui::IsItemClicked() && (!m_treeviewHoverEntry.lock()) && (!ImGui::GetIO().KeyCtrl))
			{
				setActiveEntry(m_model->getTree().getRoot());
			}
			if (ImGui::IsItemClicked(1) && getSelections().existElement())
			{
				ImGui::OpenPopup(kRightClickedMenuName.c_str());
			}
			if (ImGui::BeginPopup(kRightClickedMenuName.c_str()))
			{
				drawRightClickedMenu();
				ImGui::EndPopup();
			}
			ImGui::PopID();
		}

		// Row #1 is content snap shot.
		ImGui::TableSetColumnIndex(1);
		{
			ImGui::PushID("##ContentViewItemInspector");
			ImGui::BeginChild("ScrollingRegion2", ImVec2(0, -footerHeightToReserve), false);
			if (workingEntry)
			{
				drawContentSnapShot(workingEntry);
			}
			ImGui::EndChild();
			ImGui::PopID();
		}

		ImGui::EndTable();
	}

	ImGui::Separator();
	size_t itemNum = 0;
	if (workingEntry)
	{
		itemNum = workingEntry->getChildren().size();
	}
	ImGui::Text("  %d  items.", itemNum);
}

void ProjectAssetTree::build()
{
	CHECK(getAssetManager()->isProjectSetup());

	const auto& projectConfig = getAssetManager()->getProjectConfig();
	m_projectAssetTreeRoot = std::make_shared<ProjectAssetTreeEntry>("asset", true, projectConfig.assetPath, nullptr);

	m_pathEntryMap.clear();

	m_pathEntryMap[projectConfig.assetPath] = m_projectAssetTreeRoot;
	m_projectAssetTreeRoot->buildTreeRecursive(this);
}

ProjectAssetTreeEntry::ProjectAssetTreeEntry(
	  u8str name
	, bool bFolder
	, const std::filesystem::path& path
	, std::shared_ptr<ProjectAssetTreeEntry> parent)
	: m_name(name)
	, m_bFolder(bFolder)
	, m_path(path)
	, m_parent(parent)
	, m_runtimeUUID(engine::buildRuntimeUUID64u())
	, m_bFolderOpen(false)
{

}

// Build set for sanpshot.
ImTextureID ProjectAssetTreeEntry::getSet(ImVec2& outUv0, ImVec2& outUv1)
{
	ImTextureID result;
	ImVec2 uv0, uv1;
	{
		uv0 = { 0.0f, 0.0f };
		uv1 = { 1.0f, 1.0f };
		static const float kUvScale = 0.02f;

		uv0.x = -kUvScale;
		uv1.x = 1.0f + kUvScale;
		uv0.y = -kUvScale;
		uv1.y = 1.0f + kUvScale;

		const auto* builtinResources = Editor::get()->getBuiltinResources();
		if (m_bFolder)
		{
			result = Editor::get()->getClampToTransparentBorderImGuiTexture(builtinResources->folderImage.get());
		}
		else
		{
			result = Editor::get()->getClampToTransparentBorderImGuiTexture(builtinResources->fileImage.get());
			if (m_path.extension().string().starts_with(".dark"))
			{
				if (auto asset = getAssetManager()->getOrLoadAsset<AssetInterface>(m_path).lock())
				{
					result = 
						Editor::get()->getClampToTransparentBorderImGuiTexture(asset->getSnapshotImage());

					const auto w = asset->getSnapshotImage()->getExtent().width;
					const auto h = asset->getSnapshotImage()->getExtent().height;

					if (w < h)
					{
						uv0.x = 0.0f - (1.0f - float(w) / float(h)) * 0.5f;
						uv1.x = 1.0f + (1.0f - float(w) / float(h)) * 0.5f;

						uv0.y = -kUvScale;
						uv1.y = 1.0f + kUvScale;

					}
					else if (w > h)
					{
						uv0.y = 0.0f - (1.0f - float(h) / float(w)) * 0.5f;
						uv1.y = 1.0f + (1.0f - float(h) / float(w)) * 0.5f;

						uv0.x = -kUvScale;
						uv1.x = 1.0f + kUvScale;
					}
				}
			}

		}
	}

	outUv0 = uv0;
	outUv1 = uv1;
	return result;
}

void ProjectAssetTreeEntry::buildTreeRecursive(ProjectAssetTree* tree)
{
	if (m_bFolder)
	{
		for (const auto& entry : std::filesystem::directory_iterator(m_path))
		{
			const bool bFolder = std::filesystem::is_directory(entry);

			const bool bSkip = (!bFolder) && (!entry.path().extension().string().starts_with(".dark"));
			if (!bSkip)
			{
				auto u16FileNameString = entry.path().filename().replace_extension().u16string();

				auto child = std::make_shared<ProjectAssetTreeEntry>(
					utf8::utf16to8(u16FileNameString), 
					bFolder, 
					entry.path(), 
					shared_from_this());

				m_children.push_back(child);

				tree->m_pathEntryMap[entry.path()] = child;

				child->buildTreeRecursive(tree);
			}
		}

		std::sort(m_children.begin(), m_children.end(), [](const auto& A, const auto& B)
		{
			if ((A->isFoleder() && B->isFoleder()) || (!A->isFoleder() && !B->isFoleder()))
			{
				return A->getName() < B->getName();
			}
			return A->isFoleder();
		});
	}
}

ProjectContentModel::ProjectContentModel()
{
	rebuild();

	m_onAssetNewlySaveToDisk = getAssetManager()->onAssetNewlySavedToDisk.addLambda([this](std::shared_ptr<AssetInterface> asset)
	{
		m_bDirty = true;
	});
}

void ProjectContentModel::rebuild()
{
	m_projectAssetTree.build();
	onProjectTreeRebuild.broadcast();
}

ProjectContentModel::~ProjectContentModel()
{
	release();
}

void ProjectContentModel::release()
{
	if (m_onAssetNewlySaveToDisk.isValid())
	{
		getAssetManager()->onAssetNewlySavedToDisk.remove(m_onAssetNewlySaveToDisk);
		m_onAssetNewlySaveToDisk = {};
	}
}

void ProjectContentModel::tick()
{
	if (m_bDirty)
	{
		rebuild();
		
		m_bDirty = false;
	}
}
