	#include "Pch.h"
#include "ContentViewer.h"
#include "../Editor.h"
#include "EditorAsset.h"
#include "../EditorAsset.h"

using namespace Flower;
using namespace Flower::UI;

static const std::string CONTENTVIEWR_IconContent = ICON_FA_FOLDER_CLOSED;
static const std::string CONTENTVIEWR_ImportIcon = std::string("  ") + ICON_FA_FILE_IMPORT + std::string("  IMPORT  ");
static const std::string CONTENTVIEWR_NewIcon = std::string("  ") + ICON_FA_SQUARE_PLUS + std::string("  NEW  ");
static const std::string CONTENTVIEWR_SearchAssetIcon = ICON_FA_MAGNIFYING_GLASS;
static const std::string CONTENTVIEWR_SaveIcon = std::string("  ") + ICON_FA_FILE_SIGNATURE + std::string("  SAVE ALL  ");


static const Flower::UUID GEditorFloderIconUUID = buildUUID();
static const Flower::UUID GEditorFileIconUUID = buildUUID();

WidgetContentViewer::WidgetContentViewer()
	: Widget("  " + CONTENTVIEWR_IconContent + "  Content")
{

}

WidgetContentViewer::~WidgetContentViewer() noexcept
{

}

const std::unordered_set<Flower::UUID>& WidgetContentViewer::getSelectionAssets() const
{
	return m_dragDropObjects->selectAssets;
}

void WidgetContentViewer::onInit()
{
	m_assetTypeDrawer = std::unique_ptr<AssetTypeDrawer>(new AssetTypeDrawer(this));
	m_dragDropObjects = std::make_unique<DragAndDropAssets>();

	{
		vkDeviceWaitIdle(RHI::Device);

		auto GEditorFolderIconLoad = RawAssetTextureLoadTask::build(
			"Image/Folder.png",
			GEditorFloderIconUUID,
			VK_FORMAT_R8G8B8A8_SRGB);
		GEngine->getRuntimeModule<AssetSystem>()->addUploadTask(GEditorFolderIconLoad);

		auto GEditorFileIconLoad = RawAssetTextureLoadTask::build(
			"Image/File.png",
			GEditorFileIconUUID,
			VK_FORMAT_R8G8B8A8_SRGB);
		GEngine->getRuntimeModule<AssetSystem>()->addUploadTask(GEditorFileIconLoad);

		GEngine->getRuntimeModule<AssetSystem>()->flushUploadTask();
	}
}

void WidgetContentViewer::onRelease()
{
	m_snapshotDrawers.clear();
}

void WidgetContentViewer::onTick(const RuntimeModuleTickData& tickData)
{
	
	m_assetTypeDrawer->flushLazyCallFunctions();

	if (m_bCacheSnapShotDirty)
	{
		m_bCacheSnapShotDirty = false;
		m_snapshotDrawers.clear();

		if (!m_workingEntry.lock())
		{
			m_workingEntry = AssetRegistryManager::get()->getRoot();
		}

		if (auto workingEntry = m_workingEntry.lock())
		{
			for (auto& child : workingEntry->getChildren())
			{
				AssetSnapShotDrawer drawer(this);
				drawer.entry = child->getRegistryUUID();

				m_snapshotDrawers.push_back(drawer);
			}
		}
	}

	{
		const char* selectSceneId = "Select scene";
		if (m_bSceneSelect)
		{
			ImGui::OpenPopup(selectSceneId);

			m_bSceneSelect = false;
		}

		// Always center this window when appearing
		ImVec2 center = ImGui::GetMainViewport()->GetCenter();
		ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
		if (ImGui::BeginPopupModal(selectSceneId, NULL, ImGuiWindowFlags_AlwaysAutoResize))
		{
			ImGui::Text("Select one scene to open.\n\n");
			ImGui::Separator();

			const auto& scenes = ProjectContext::get()->project.getScenes();

			for (const auto& s : scenes)
			{
				if (ImGui::Selectable(s.c_str()))
				{
					GEngine->getRuntimeModule<SceneManager>()->loadScene(s);
					ImGui::CloseCurrentPopup();
				}
			}

			ImGui::EndPopup();
		}
	}


	const char* saveSceneId = "Save scene";

	auto saveAssetAction = [](const std::filesystem::path& sceneSavePath)
	{
		GEngine->getRuntimeModule<SceneManager>()->saveScene(sceneSavePath);
		if (AssetRegistryManager::get()->isDirty())
		{
			AssetRegistryManager::get()->save();
		}
	};

	if (m_bNeedSaveAsset)
	{
		auto* activeScene = GEngine->getRuntimeModule<SceneManager>()->getScenes();
		const bool bNeedCreateScene = activeScene->getSavePath().empty();

		if (bNeedCreateScene)
		{
			ImGui::OpenPopup(saveSceneId);
		}
		else
		{
			saveAssetAction(activeScene->getSavePath());
		}
		m_bNeedSaveAsset = false;
	}

	// Always center this window when appearing
	ImVec2 center = ImGui::GetMainViewport()->GetCenter();
	ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
	if (ImGui::BeginPopupModal(saveSceneId, NULL, ImGuiWindowFlags_AlwaysAutoResize))
	{
		ImGui::Text("Select one area to save the scene.\n\n");
		ImGui::Separator();

		ImGui::InputText("Name", m_createSceneName, GCreateScenePathSize);
		ImGui::SameLine();

		std::string sceneName = m_createSceneName;

		bool bValidCanCreate = false;
		if (!sceneName.empty())
		{
			if (!ProjectContext::get()->project.existScene(sceneName))
			{
				if (std::all_of(sceneName.begin(), sceneName.end(), [](char c)
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

		auto* assetSys = GEngine->getRuntimeModule<AssetSystem>();
		if (ImGui::Button("  Create  "))
		{
			auto sceneFolder = assetSys->getProjectSceneFolderPath();
			sceneFolder /= sceneName;

			sceneFolder = std::filesystem::relative(sceneFolder, ProjectContext::get()->path);
			saveAssetAction(sceneFolder);

			// Add scene to project.
			ProjectContext::get()->project.addScene(sceneFolder.string());

			saveActiveProject(assetSys->getProjectPathFilePath());
			

			ImGui::CloseCurrentPopup();
		}
		if (!bValidCanCreate)
		{
			ImGui::EndDisabled();
		}

		ImGui::EndPopup();
	}
}


void WidgetContentViewer::onVisibleTick(const RuntimeModuleTickData& tickData)
{
	ImGui::Separator();

	drawMenu();
	ImGui::Separator();

	if (ProjectContext::get()->isValid())
	{
		ImGui::TextDisabled("Working Project %s and Working Path: %s.", 
			ProjectContext::get()->project.getNameWithSuffix().c_str(), 
			ProjectContext::get()->path.string().c_str());
	}
	else
	{
		ImGui::TextDisabled("Please select project to open. Current no project path set.");
		ImGui::Separator();
		return;
	}

	drawContent();
}

void WidgetContentViewer::drawContent()
{
	const float footerHeightToReserve = ImGui::GetStyle().ItemSpacing.y + ImGui::GetFrameHeightWithSpacing();

	if (ImGui::BeginTable("AssetContentTable", 2, ImGuiTableFlags_BordersInner | ImGuiTableFlags_Resizable))
	{
		ImGui::TableNextRow();
		ImGui::TableSetColumnIndex(0);
		{
			ImGui::PushID("##ContentViewItemTreeView");
			ImGui::BeginChild("ScrollingRegion", ImVec2(0, -footerHeightToReserve), false, ImGuiWindowFlags_HorizontalScrollbar);

			for (const auto& child : AssetRegistryManager::get()->getRoot()->getChildren())
			{
				drawContentTreeView(child);
			}

			ImGui::EndChild();
			ImGui::PopID();
		}

		ImGui::TableSetColumnIndex(1);
		{
			ImGui::PushID("##ContentViewItemInspector");
			ImGui::BeginChild("ScrollingRegion2", ImVec2(0, -footerHeightToReserve), false, ImGuiWindowFlags_HorizontalScrollbar);

			drawContentSnapshot();

			ImGui::EndChild();
			ImGui::PopID();
		}

		ImGui::EndTable();
	}

	ImGui::Separator();
	ImGui::Text("  %d  items.", m_snapshotDrawers.size());
}

void WidgetContentViewer::drawContentTreeView(const std::shared_ptr<Flower::RegistryEntry>& entry)
{
	const bool bTreeNode = !entry->isLeaf() || entry->getAssetHeaderID().empty();

	ImGuiTreeNodeFlags nodeFlags = ImGuiTreeNodeFlags_AllowItemOverlap | ImGuiTreeNodeFlags_SpanFullWidth;
	nodeFlags |= bTreeNode ? ImGuiTreeNodeFlags_OpenOnArrow : ImGuiTreeNodeFlags_Leaf;

	std::string name = bTreeNode ? entry->getName() : entry->getHeader()->getName();

	if (const auto selectedEntry = m_selectedEntryInTreeView.lock())
	{
		nodeFlags |= selectedEntry == entry ? ImGuiTreeNodeFlags_Selected : nodeFlags;
	}

	bool bNodeOpen = ImGui::TreeNodeEx(name.c_str(), nodeFlags);

	const auto bLeftClick = ImGui::IsMouseClicked(0);
	const auto bRightClick = ImGui::IsMouseClicked(1);
	const auto bDoubleClick = ImGui::IsMouseDoubleClicked(0);

	if (bLeftClick || bDoubleClick || bRightClick)
	{
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_RectOnly))
		{
			if (entry->isLeaf())
			{
				m_selectedEntryInTreeView = entry->getParent().lock();
			}
			else
			{
				m_selectedEntryInTreeView = entry;
			}
			
			setWorkingEntry(m_selectedEntryInTreeView.lock());
		}
	}

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

void WidgetContentViewer::drawContentSnapshot()
{
	const size_t inspectItemNum = m_snapshotDrawers.size();

	const auto availRegion = ImGui::GetContentRegionAvail();
	const float itemDimSize = ImGui::GetTextLineHeightWithSpacing() * m_inspectorItemIconSize;

	const uint32_t drawItemPerRow = uint32_t(glm::max(1.0f, glm::ceil(availRegion.x / itemDimSize - 0.25f) - 1.0f));

	const size_t minDrawRowNum = size_t(glm::max(1.0f, glm::ceil(availRegion.y / itemDimSize)));
	const uint32_t drawRowNum = uint32_t(glm::max(minDrawRowNum, inspectItemNum / drawItemPerRow + 1));

	static ImGuiTableFlags flags =
		ImGuiTableFlags_ScrollY |
		ImGuiTableFlags_Hideable |
		ImGuiTableFlags_NoClip |
		ImGuiTableFlags_NoBordersInBody;

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
						m_snapshotDrawers[drawId].draw(itemDimSize);
					}
				}

				ImGui::PopID();
			}
		}

		ImGui::EndTable();
	}

}

void WidgetContentViewer::importAssetAction(EAssetType type, std::shared_ptr<Flower::RegistryEntry> entry)
{
	const auto& typeNameMap = EditorAssetSystem::get()->getTypeNameMap();
	const auto& assetMap = EditorAssetSystem::get()->getRegisterMap();

	nfdpathset_t pathSet;
	nfdresult_t result = NFD_OpenDialogMultiple(assetMap.at(typeNameMap.at(type)).rawResourceExtensions, NULL, &pathSet);
	if (result != NFD_OKAY)
	{
		return;
	}

	std::vector<std::future<void>> importFutures;

	for (size_t i = 0; i < NFD_PathSet_GetCount(&pathSet); ++i)
	{
		nfdchar_t* path = NFD_PathSet_GetPath(&pathSet, i);
		std::filesystem::path filePath = path;

		ImportOptions options{};
		if (type == EAssetType::Texture)
		{
			options.texOptions = ImportTextureOptions{};
			if (filePath.extension() == std::filesystem::path(".hdr"))
			{
				options.texOptions.value().bHdr = true;
				options.texOptions.value().bSrgb = false;
				options.texOptions.value().bBuildMipmap = false; // Current skip mipmap build.
			}
			else
			{
				options.texOptions.value().bHdr = false;
			}
		}

		// Asset import and bake can't be multi thread.
		// It's pretty hard to handle asset interdependence case. :(
		GEngine->getRuntimeModule<AssetSystem>()->importAsset(filePath, type, entry, options);
	}

	NFD_PathSet_Free(&pathSet);

	m_bCacheSnapShotDirty = true;
}

void WidgetContentViewer::drawMenu()
{
	const float sizeLable = ImGui::GetFontSize();
	if (ImGui::BeginTable("Import UIC##", 5))
	{
		ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, sizeLable * 5.5f);
		ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, sizeLable * 4.0f);
		ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, sizeLable * 4.0f);
		ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, sizeLable * 6.0f);
		ImGui::TableSetupColumn("", ImGuiTableColumnFlags_None);

		ImGui::TableNextColumn();

		if (ImGui::Button((CONTENTVIEWR_ImportIcon).c_str()))
		{
			ImGui::OpenPopup("##XAssetMenu_Import");
		}
		UIHelper::hoverTip("Import new asset from disk.");

		if (ImGui::BeginPopup("##XAssetMenu_Import"))
		{
			ImGui::TextDisabled("Import  Assets:");
			ImGui::Separator();

			m_assetTypeDrawer->drawAssetImport();
			ImGui::EndPopup();
		}

		ImGui::TableNextColumn();

		if (ImGui::Button((CONTENTVIEWR_NewIcon).c_str()))
		{
			ImGui::OpenPopup("##XAssetMenu_Create");

		}
		UIHelper::hoverTip("Create new asset.");
		if (ImGui::BeginPopup("##XAssetMenu_Create"))
		{
			ImGui::TextDisabled("Create  Assets:");
			ImGui::Separator();

			m_assetTypeDrawer->drawAssetNew();
			ImGui::EndPopup();
		}

		ImGui::TableNextColumn();

		if (ImGui::Button("Scene"))
		{
			m_bSceneSelect = true;
		}

		ImGui::TableNextColumn();

		auto* activeScene = GEngine->getRuntimeModule<SceneManager>()->getScenes();

		bool bShouldPushDisableForSave = 
			!AssetRegistryManager::get()->isDirty() && 
			!activeScene->isDirty();

		if (bShouldPushDisableForSave)
		{
			ImGui::BeginDisabled();
		}

		if (ImGui::Button((CONTENTVIEWR_SaveIcon).c_str()))
		{
			m_bNeedSaveAsset = true;
		}
		if (bShouldPushDisableForSave)
		{
			ImGui::EndDisabled();
		}

		ImGui::TableNextColumn();
		m_filter.Draw((CONTENTVIEWR_SearchAssetIcon).c_str());

		ImGui::EndTable();
	}
}

void AssetTypeDrawer::drawAssetImport()
{
	const auto& filterMaps = EditorAssetSystem::get()->getRegisterMap();

	for (const auto& pair : filterMaps)
	{
		if (pair.second.type != EAssetType::Max)
		{
			bool bPreReturn = false;
			ImGui::PushID(pair.first.c_str());
			if (ImGui::Selectable(pair.second.decoratorName.c_str()))
			{
				const auto typeCap = pair.second.type;
				lazyCallFunctions.push_back([typeCap, this]()
				{
					m_viewer->importAssetAction(typeCap);
				});

				bPreReturn = true;
			}
			ImGui::PopID();

			if (bPreReturn)
			{
				return;
			}
		}
	}
}

void AssetTypeDrawer::drawAssetNew()
{
	const auto& filterMaps = EditorAssetSystem::get()->getRegisterMap();

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

			if (bPreReturn)
			{
				return;
			}
		}
	}
}

void AssetTypeDrawer::flushLazyCallFunctions()
{
	if (m_counter >= MIKU_MAGIC_NUMBER)
	{
		for (auto& func : lazyCallFunctions)
		{
			if (func)
			{
				func();
			}
		}
		lazyCallFunctions.clear();
		m_counter = 0;
	}
	m_counter++;
}

void AssetSnapShotDrawer::draw(float drawDimSize)
{
	auto entryPtr = AssetRegistryManager::get()->getEntryMap().at(entry).lock();
	

	ImGui::BeginGroup();
	std::hash<std::string> haser;
	int hashId = int(haser(entry));
	ImGui::PushID(hashId);

	ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 0);
	float textH = ImGui::GetTextLineHeightWithSpacing();
	bool bItemHover = ImGui::IsMouseHoveringRect(ImGui::GetCursorScreenPos(),
		ImVec2(ImGui::GetCursorScreenPos().x + drawDimSize, ImGui::GetCursorScreenPos().y + drawDimSize));


	if (ImGui::IsMouseClicked(0) && bItemHover)
	{
		if (ImGui::GetIO().KeyCtrl)
		{
			if (viewer->m_dragDropObjects->selectAssets.contains(entry))
			{
				viewer->m_dragDropObjects->selectAssets.erase(entry);
			}
			else
			{
				viewer->m_dragDropObjects->selectAssets.insert(entry);
			}
		}
		else
		{
			viewer->m_dragDropObjects->selectAssets.clear();
			viewer->m_dragDropObjects->selectAssets.insert(entry);
		}
	}
	const bool bItemSeleted = viewer->m_dragDropObjects->selectAssets.contains(entry);


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
		bItemHover ? IM_COL32(250, 244, 11, 255) : IM_COL32(255, 255, 255, 80), bItemHover ? 2.0f : 0.0f, 0, bItemHover ? 5.0f : 1.0f);

	static auto* assetSystem = GEngine->getRuntimeModule<AssetSystem>();


	VkDescriptorSet set;

	ImVec2 uv0{ 0.0f, 0.0f };
	ImVec2 uv1{ 1.0f, 1.0f };
	static const float uvScale = 0.02f;

	
	set = EditorAssetSystem::get()->getSetByAssetAsSnapShot(TextureManager::get()->getImage(EngineTextures::GWhiteTextureUUID).get());

	if (!entryPtr->isFoleder())
	{
		if (auto assetHeader = AssetRegistryManager::get()->getHeaderMap().at(entryPtr->getAssetHeaderID()))
		{
			if (assetHeader->getType() == EAssetType::Texture)
			{
				auto imageAsset = std::dynamic_pointer_cast<ImageAssetHeader>(assetHeader);

				if (!cacheAsset)
				{
					cacheAsset = TextureManager::get()->getOrCreateLRUSnapShot(imageAsset);
				}

				set = EditorAssetSystem::get()->getSetByAssetAsSnapShot(cacheAsset->getReadyAsset());

				const auto w = imageAsset->getSnapShotWidth();
				const auto h = imageAsset->getSnapShotHeight();

				if (w < h)
				{
					uv0.x = 0.0f - (1.0f - float(w) / float(h)) * 0.5f;
					uv1.x = 1.0f + (1.0f - float(w) / float(h)) * 0.5f;

					uv0.y = -uvScale;
					uv1.y = 1.0f + uvScale;

				}
				else if (w > h)
				{
					uv0.y = 0.0f - (1.0f - float(h) / float(w)) * 0.5f;
					uv1.y = 1.0f + (1.0f - float(h) / float(w)) * 0.5f;

					uv0.x = -uvScale;
					uv1.x = 1.0f + uvScale;
				}
			}
			else
			{
				set = EditorAssetSystem::get()->getSetByAssetAsSnapShot(TextureManager::get()->getImage(GEditorFileIconUUID).get());

				uv0.x = -uvScale;
				uv1.x = 1.0f + uvScale;
				uv0.y = -uvScale;
				uv1.y = 1.0f + uvScale;
			}
		}
	}
	else
	{
		set = EditorAssetSystem::get()->getSetByAssetAsSnapShot(TextureManager::get()->getImage(GEditorFloderIconUUID).get());

		uv0.x = -uvScale;
		uv1.x = 1.0f + uvScale;
		uv0.y = -uvScale;
		uv1.y = 1.0f + uvScale;
	}
	

	ImGui::Image(set, { drawDimSize , drawDimSize }, uv0, uv1 );

	const float indentSize = 4.0f;
	ImGui::Indent(indentSize);
	ImGui::PushTextWrapPos(ImGui::GetCursorPos().x + drawDimSize - indentSize);

	ImGui::Text(entryPtr->getName().c_str());


	ImGui::PopTextWrapPos();
	ImGui::Unindent();

	ImGui::Spacing();
	ImGui::Spacing();
	ImGui::Spacing();
	ImGui::Spacing();

	ImGui::PopStyleVar();
	ImGui::EndGroup();


	if (bItemSeleted && ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID))
	{
		ImGui::SetDragDropPayload(EditorAsset::DragDropName.c_str(), (void*)viewer->m_dragDropObjects.get(), sizeof(void*));

		const auto& filterMaps = EditorAssetSystem::get()->getRegisterMap();
		const auto& typeMaps = EditorAssetSystem::get()->getTypeNameMap();
		static auto* assetSystem = GEngine->getRuntimeModule<AssetSystem>();

		for (const auto& id : viewer->m_dragDropObjects->selectAssets)
		{
			auto loopEntryPtr = AssetRegistryManager::get()->getEntryMap().at(id).lock();
			if (!loopEntryPtr->getAssetHeaderID().empty())
			{
				auto loopassetHeader = AssetRegistryManager::get()->getHeaderMap().at(loopEntryPtr->getAssetHeaderID());
				const std::string& typeName = typeMaps.at(loopassetHeader->getType());

				ImGui::Text(filterMaps.at(typeName).iconName.c_str());
				ImGui::SameLine();
				ImGui::Text(loopassetHeader->getName().c_str());
			}
		}

		ImGui::EndDragDropSource();
	}

	if (bItemSeleted && ImGui::IsMouseDoubleClicked(0))
	{
		if (!entryPtr->isLeaf())
		{
			viewer->setWorkingEntry(entryPtr);
		}
	}

	ImGui::PopID();
}
