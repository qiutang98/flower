#pragma once

#include "../widget.h"
#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>
#include <asset/asset.h>
#include <utf8/cpp17.h>

#include <scene/scene.h>

class RenderManagerWidget : public Widget
{
public:
	RenderManagerWidget(Editor* editor);

	// event init.
	virtual void onInit() override;

	// event always tick.
	virtual void onTick(const engine::RuntimeModuleTickData& tickData, engine::VulkanContext* context) override;

	// event release.
	virtual void onRelease() override;

	// event when widget visible tick.
	virtual void onVisibleTick(const engine::RuntimeModuleTickData& tickData) override;


private:


};