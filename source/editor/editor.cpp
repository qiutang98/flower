#include "editor.h"
#include "editor_asset.h"
#include <utf8.h>
#include <utf8/cpp17.h>
#include <imgui/imgui_impl_vulkan.h>
#include <asset/asset.h>
#include <scene/scene.h>
#include <asset/asset_common.h>
#include <nfd.h>

#if _WIN32
	#include <Windows.h>
#endif

using namespace engine;
using namespace engine::ui;

Editor* Editor::get()
{
	static Editor editor;
	return &editor;
}

void Editor::initBuiltinResources()
{
	auto flushUploadImage = [&](const char* path)
	{
		int32_t texWidth, texHeight, texChannels;
		stbi_uc* pixels = stbi_load(path, &texWidth, &texHeight, &texChannels, 4);
		ASSERT(pixels, "Load builtin folder image fail, check your install path.");

		auto newImage = std::make_unique<VulkanImage>(
			m_context,
			path,
			buildBasicUploadImageCreateInfo(texWidth, texHeight));

		auto stageBuffer = std::make_unique<VulkanBuffer>( 
			m_context,
			"StageUploadBuffer",
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
			texWidth * texHeight * 4,
			pixels);

		m_context->executeImmediatelyMajorGraphics([&](VkCommandBuffer cmd)
		{
			newImage->transitionLayout(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, buildBasicImageSubresource());

			VkBufferImageCopy region{};
			region.bufferOffset = 0;
			region.bufferRowLength = 0;
			region.bufferImageHeight = 0;
			region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			region.imageSubresource.mipLevel = 0;
			region.imageSubresource.baseArrayLayer = 0;
			region.imageSubresource.layerCount = 1;
			region.imageOffset = { 0, 0, 0 };
			region.imageExtent = newImage->getExtent();
			vkCmdCopyBufferToImage(cmd, stageBuffer->getVkBuffer(), newImage->getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

			newImage->transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
		});

		stbi_image_free(pixels);
		return std::move(newImage);
	};

	m_builtinResources.folderImage = flushUploadImage("image/folder.png");
	m_builtinResources.fileImage   = flushUploadImage("image/file.png");
	m_builtinResources.materialImage = flushUploadImage("image/material.png");
	m_builtinResources.sceneImage = flushUploadImage("image/scene.png");
	m_builtinResources.meshImage = flushUploadImage("image/mesh.png");

	m_builtinResources.sunImage = flushUploadImage("image/sun.png");
	m_builtinResources.userImage = flushUploadImage("image/user.png");
	m_builtinResources.effectImage = flushUploadImage("image/effects.png");
	m_builtinResources.postImage = flushUploadImage("image/post.png");
}

void Editor::releaseBuiltinResources()
{
	// Clear all reference.
	m_builtinResources = {};
}

void Editor::shortcutHandle()
{
	// Handle undo shortcut. ctrl_z and ctrl_y.
	if (ImGui::IsKeyDown(ImGuiKey_LeftCtrl) || ImGui::IsKeyDown(ImGuiKey_RightCtrl))
	{
		if (ImGui::IsKeyPressed(ImGuiKey_Z))
		{
			if (m_undo->undo()) {}
		}
		else if (ImGui::IsKeyPressed(ImGuiKey_Y))
		{
			if (m_undo->redo()) {}
		}
		else if (ImGui::IsKeyPressed(ImGuiKey_S))
		{
			if (!m_dirtyAssets.empty())
			{
				saveDirtyAssetActions();
			}
		}
	}
}

void Editor::saveDirtyAssetActions()
{
	const auto& typeNameMap = EditorAsset::get()->getTypeNameMap();
	const auto& assetMap = EditorAsset::get()->getRegisterMap();

	std::vector<engine::UUID> saveSets; 

	// Save dirty asset to disk.
	for (auto& assetPair : m_dirtyAssets)
	{
		auto asset = assetPair.second.lock();
		if (!asset)
		{
			continue;
		}

		if (asset->savePathUnvalid())
		{
			std::string path;

			nfdchar_t* outPathChars;

			std::string suffix = std::string(asset->getSuffix()).erase(0, 1) + "\0";
			nfdchar_t* filterList = suffix.data();
			nfdresult_t result = NFD_SaveDialog(filterList, getProjectAssetPathUtf8().c_str(), &outPathChars);
			
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
				std::string assetNameUtf8 = utf8::utf16to8(assetName.u16string());

				const auto relativePath = buildRelativePathUtf8(getProjectRootPathUtf16(), fp);
				asset->setNameUtf8(assetNameUtf8);
				asset->setRelativePathUtf8(relativePath);
			}
		}

		if (asset->saveAction())
		{
			saveSets.push_back(asset->getUUID());
		}
		else
		{
			LOG_ERROR("Fail to save asset {}!", asset->getNameUtf8());
		}
	}

	m_dirtyAssets.clear();
	setupProjectDirectory(m_projectFilePathUtf16);
}



int Editor::run(int argc, char** argv)
{
	// Prepare config.
	Config config;
	config.appName = "flower";
	config.bConsole = false;
	config.bEnableLogFileOut = true;
	config.logFolder = "log";
	config.configFolder = "config";
	config.iconPath = "image/icon.png";
	config.windowInfo.bResizeable = false;
	config.windowInfo.initWidth = 800;
	config.windowInfo.initHeight = 450;
	config.windowInfo.windowShowMode = Config::InitWindowInfo::EWindowMode::Free;

	// Framework init and register module.
	Framework* app = Framework::get();
	app->initFramework(config);
	{
		app->getEngine().registerRuntimeModule<VulkanContext>();
		app->getEngine().registerRuntimeModule<SceneManager>();
		app->getEngine().registerRuntimeModule<Renderer>();
		app->getEngine().registerRuntimeModule<AssetSystem>();
	}

	// Try init app.
	if (app->init())
	{
		// Init widgets.
		init();

		// Run app loop if available.
		app->loop();

		// Release widgets.
		release();

		// Release app when loop end.
		app->release();
	}

	return 0;
}

void Editor::setTitleName() const
{
	auto activeScene = m_sceneManager->getActiveScene();

	std::string newTitleName = Framework::get()->getConfig().appName + " - " + m_projectNameUtf8 + " - " + activeScene->getNameUtf8();

	if (activeScene->isDirty())
	{
		newTitleName += " * ";
	}

	glfwSetWindowTitle(Framework::get()->getWindow(), newTitleName.c_str());
}

void Editor::setupProjectDirectory(const std::filesystem::path& inProjectPath)
{
	m_projectRootPathUtf8 = utf8::utf16to8(inProjectPath.parent_path().u16string());
	m_projectFilePathUtf8 = utf8::utf16to8(inProjectPath.u16string());
	m_projectNameUtf8     = utf8::utf16to8(inProjectPath.filename().replace_extension().u16string());
	m_projectAssetPathUtf8 = utf8::utf16to8((inProjectPath.parent_path() / "asset").u16string());

	m_projectRootPathUtf16 = inProjectPath.parent_path().u16string();
	m_projectFilePathUtf16 = inProjectPath.u16string();
	m_projectNameUtf16     = inProjectPath.filename().replace_extension().u16string();
	m_projectAssetPathUtf16 = (inProjectPath.parent_path() / "asset").u16string();

	setTitleName();

	m_assetSystem->setupProject(inProjectPath);
	m_projectContent->setupProject(inProjectPath);
}

void Editor::onAssetDirty(std::shared_ptr<engine::AssetInterface> asset)
{
	m_dirtyAssets[asset->getUUID()] = asset;

	if (asset->getType() == EAssetType::Scene)
	{
		auto scene = std::static_pointer_cast<Scene>(asset);
		if (scene == m_sceneManager->getActiveScene())
		{
			setTitleName();
		}
	}
}

VkDescriptorSet Editor::getSet(VulkanImage* image, const VkSamplerCreateInfo& sampler)
{
	const auto& uuid = image->getRuntimeUUID();
	if (!m_cacheImageSet[uuid])
	{
		m_cacheImageSet[uuid] = ImGui_ImplVulkan_AddTexture(
			m_context->getSamplerCache().createSampler(sampler), 
			image->getOrCreateView(buildBasicImageSubresource()),
			VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		);
	}

	return m_cacheImageSet.at(uuid);
}

VkDescriptorSet Editor::getClampToTransparentBorderSet(engine::VulkanImage* image)
{
	static const VkSamplerCreateInfo info = SamplerFactory::pointClampBorder0000();

	return getSet(image, info);
}

void Editor::init()
{
#if _WIN32
	SetConsoleOutputCP(CP_UTF8);
#endif // _WIN32

	m_engine = Framework::get()->getEnginePtr();

	m_context = m_engine->getRuntimeModule<VulkanContext>();
	ASSERT(m_context, "You must register one vulkan context module when require widget ui.");

	m_renderer = m_engine->getRuntimeModule<Renderer>();
	ASSERT(m_renderer, "You must register one renderer module when require widget ui.");

	m_sceneManager = m_engine->getRuntimeModule<SceneManager>();
	ASSERT(m_sceneManager, "You must register one scene module when require widget ui.");

	m_assetSystem = m_engine->getRuntimeModule<AssetSystem>();
	ASSERT(m_assetSystem, "You must register one assetsystem module when require widget ui.");

	constexpr size_t kMaxUndoItem = 100;
	m_undo = std::make_unique<Undo>(kMaxUndoItem);

	initBuiltinResources();

	// Register asset delegates
	{
		m_onAssetDirtyHandle = m_assetSystem->onAssetDirty.addRaw(this, &Editor::onAssetDirty);
	}

	// Create widgets.
	{
		m_mainDockspace = std::make_unique<MainViewportDockspaceAndMenu>(this);
		m_mainDockspace->init();

		m_downbar = std::make_unique<DownbarWidget>(this);
		m_downbar->init();

		// Render Manager before viewport to ensure change value work.
		m_renderManager = std::make_unique<RenderManagerWidget>(this);
		m_renderManager->init();


		m_console = std::make_unique<WidgetConsole>(this);
		m_console->init();

		m_hubWidget = std::make_unique<HubWidget>(this);
		m_hubWidget->init();

		m_projectContent = std::make_unique<ProjectContentWidget>(this);
		m_projectContent->init();

		m_outlinerWidget = std::make_unique<SceneOutlinerWidget>(this);
		m_outlinerWidget->init();

		m_viewport = std::make_unique<ViewportWidget>(this);
		m_viewport->init();

		m_detail = std::make_unique<WidgetDetail>(this);
		m_detail->init();


		m_assetConfigs = std::make_unique<AssetConfigWidgetManager>(this);
	}

	// Register tick function on renderer delegate.
	m_tickFunctionHandle = m_renderer->tickFunctions.addRaw(this, &Editor::tick);
	m_tickCmdFunctionHandle = m_renderer->tickCmdFunctions.addRaw(this, &Editor::tickWithCmd);


}

void Editor::release()
{
	// unregister tick function.
	CHECK(m_renderer->tickFunctions.remove(m_tickFunctionHandle));
	CHECK(m_renderer->tickCmdFunctions.remove(m_tickCmdFunctionHandle));
	CHECK(m_assetSystem->onAssetDirty.remove(m_onAssetDirtyHandle));

	m_context->waitDeviceIdle();

	// Release widgets.
	m_mainDockspace->release();
	m_console->release();
	m_downbar->release();
	m_hubWidget->release();
	m_projectContent->release();
	m_outlinerWidget->release();
	m_viewport->release();
	m_detail->release();
	m_renderManager->release();

	m_assetConfigs->release();

	releaseBuiltinResources();
}

void Editor::tick(const RuntimeModuleTickData& tickData, VulkanContext* context)
{
	if (m_bShouldSetFocus)
	{
		m_bShouldSetFocus = false;
		ImGui::SetWindowFocus(m_focusWindow.c_str());
	}

	if (m_bHubWidgetActive)
	{
		m_hubWidget->tick(tickData, context);
	}
	else
	{
		m_mainDockspace->tick(tickData, context);
		m_console->tick(tickData, context);
		m_downbar->tick(tickData, context);
		m_projectContent->tick(tickData, context);
		m_outlinerWidget->tick(tickData, context);
		m_viewport->tick(tickData, context);
		m_detail->tick(tickData, context);
		m_renderManager->tick(tickData, context);
		m_assetConfigs->tick(tickData, context);
	}

	tickFunctions.broadcast(tickData, context);

	shortcutHandle();


}

void Editor::tickWithCmd(const RuntimeModuleTickData& tickData, VkCommandBuffer cmd, VulkanContext* context)
{
	if (m_bHubWidgetActive)
	{
		m_hubWidget->tickWithCmd(tickData, cmd, context);
	}
	else
	{
		m_mainDockspace->tickWithCmd(tickData, cmd, context);
		m_console->tickWithCmd(tickData, cmd, context);
		m_downbar->tickWithCmd(tickData, cmd, context);
		m_projectContent->tickWithCmd(tickData, cmd, context);
		m_outlinerWidget->tickWithCmd(tickData, cmd, context);
		m_viewport->tickWithCmd(tickData, cmd, context);
		m_detail->tickWithCmd(tickData, cmd, context);
		m_renderManager->tickWithCmd(tickData, cmd, context);
		m_assetConfigs->tickCmd(tickData, cmd, context);
	}

	tickCmdFunctions.broadcast(tickData, cmd, context);
}
