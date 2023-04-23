#pragma once

#include <util/framework.h>
#include <rhi/rhi.h>
#include <renderer/renderer.h>
#include <imgui/ui.h>
#include "../widget.h"

// Control main viewport dockspace of the windows.
class MainViewportDockspaceAndMenu : public Widget
{
public:
	MainViewportDockspaceAndMenu(Editor* editor);

	static void dockspace(bool bNewWindow, const std::string& name, const engine::RuntimeModuleTickData& tickData, engine::VulkanContext* context, ImGuiViewport* viewport, std::function<void()>&& menu);

protected:
	virtual void onTick(const engine::RuntimeModuleTickData& tickData, engine::VulkanContext* context) override;



	void drawDockspaceMenu();
};