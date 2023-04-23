#pragma once

#include <util/util.h>
#include <renderer/renderer.h>
#include <scene/scene.h>
#include <asset/asset_system.h>
#include <imgui/imgui.h>
#include "undo.h"

class Widget
{
public:
	Widget(class Editor* editor, const std::string& name);
	virtual ~Widget() = default;

	// Get widget name.
	const std::string& getName() const { return m_name; }

	// Visible state set and get.
	void setVisible(bool bVisible) { m_bShow = bVisible; }
	bool getVisible() const { return m_bShow; }

	void init() { onInit(); }
	void release() { onRelease(); }

	void tick(const engine::RuntimeModuleTickData& tickData, engine::VulkanContext* context);
	void tickWithCmd(const engine::RuntimeModuleTickData& tickData, VkCommandBuffer cmd, engine::VulkanContext* context);

protected:
	// event init.
	virtual void onInit() { }

	// event on widget visible state change. sync on tick function first.
	virtual void onHide(const engine::RuntimeModuleTickData& tickData) {  }
	virtual void onShow(const engine::RuntimeModuleTickData& tickData) {  }

	// evetn before tick.
	virtual void beforeTick(const engine::RuntimeModuleTickData& tickData) {}

	// event always tick.
	virtual void onTick(const engine::RuntimeModuleTickData& tickData, engine::VulkanContext* context) {  }

	// event when widget visible tick, draw imgui logic here.
	virtual void onVisibleTick(const engine::RuntimeModuleTickData& tickData) {  }

	virtual void afterTick(const engine::RuntimeModuleTickData& tickData) { }

	// Tick with graphics command.
	virtual void onTickCmd(const engine::RuntimeModuleTickData& tickData, VkCommandBuffer cmd, engine::VulkanContext* context) {  }
	virtual void onVisibleTickCmd(const engine::RuntimeModuleTickData& tickData, VkCommandBuffer cmd) {  }

	// event release.
	virtual void onRelease() {  }

protected:
	Editor* m_editor;

	// Cache renderer module.
	engine::Renderer* m_renderer;

	// Cache engine handle.
	engine::Engine* m_engine;

	// Cache vulkan context.
	engine::VulkanContext* m_context;

	// Cache scene manager.
	engine::SceneManager* m_sceneManager = nullptr;

	engine::AssetSystem* m_assetSystem = nullptr;

	Undo* m_undo;

	// Widget show state.
	bool m_bShow;

	// Widget name.
	std::string m_name;

	// Widget prev frame show state.
	bool m_bPrevShow;

	// Window show flags.
	ImGuiWindowFlags m_flags = 0;
};