#pragma once

#include "../widget.h"
#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>
#include <asset/asset.h>
#include <utf8/cpp17.h>

class AssetConfigWidgetManager;
class Editor;

class WidgetAssetConfig
{
public:
	WidgetAssetConfig(Editor* editor, AssetConfigWidgetManager* manager, const std::filesystem::path& path);
	virtual ~WidgetAssetConfig() noexcept;

	bool shouldClosed() const { return !m_bRun; }

	// event always tick.
	void tick(const engine::RuntimeModuleTickData& tickData, engine::VulkanContext* context);

	void tickWithCmd(const engine::RuntimeModuleTickData& tickData, VkCommandBuffer cmd, engine::VulkanContext* context) { }

	const std::string& getNameUtf8() const { return m_nameUTF8; }
private:
	AssetConfigWidgetManager* m_manager;

	Editor* m_editor;

	bool m_bRun;

	std::string m_nameUTF8;
};

class AssetConfigWidgetManager
{
public:
	AssetConfigWidgetManager(Editor* editor);

	WidgetAssetConfig* openWidget(const std::filesystem::path& path);

	void tick(const engine::RuntimeModuleTickData& tickData, engine::VulkanContext* context);
	void tickCmd(const engine::RuntimeModuleTickData& tickData, VkCommandBuffer cmd, engine::VulkanContext* context);
	void release();

	ImGuiViewport* getTickDrawViewport() { return m_tickDrawCtx.viewport; }
	void setTickDrawViewport(ImGuiViewport* vp, ImGuiID dockID) { m_tickDrawCtx.viewport = vp; m_tickDrawCtx.dockId = dockID; }
	ImGuiID getTickDrawDockID() const { return m_tickDrawCtx.dockId; }
private:
	Editor* m_editor;


	std::unordered_map<std::filesystem::path, std::unique_ptr<WidgetAssetConfig>> m_widgets;

	// Cache viewport for tick switch.
	struct
	{
		ImGuiViewport* viewport;
		ImGuiID dockId;
	} m_tickDrawCtx;
	
};