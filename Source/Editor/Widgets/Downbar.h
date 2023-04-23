#pragma once

#include <util/framework.h>
#include <rhi/rhi.h>
#include <renderer/renderer.h>
#include <imgui/ui.h>
#include "../widget.h"

class DownbarWidget : public Widget
{
public:
	DownbarWidget(Editor* editor);

	static void draw(bool bNewWindow, const engine::RuntimeModuleTickData& tickData, engine::VulkanContext* context, const std::string& name, ImGuiID ID);

	virtual void onTick(const engine::RuntimeModuleTickData& tickData, engine::VulkanContext* context) override;
};