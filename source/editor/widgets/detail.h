#pragma once

#include "../widget.h"
#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>
#include <asset/asset.h>
#include <utf8/cpp17.h>

class WidgetDetail : public Widget
{
public:
	WidgetDetail(Editor* editor);
	virtual ~WidgetDetail() noexcept;

protected:
	// event init.
	virtual void onInit() override;

	// event always tick.
	virtual void onTick(const engine::RuntimeModuleTickData& tickData, engine::VulkanContext* context) override;

	// event release.
	virtual void onRelease() override;

	// event when widget visible tick.
	virtual void onVisibleTick(const engine::RuntimeModuleTickData& tickData) override;

	void drawComponent(std::shared_ptr<engine::SceneNode> node);

private:
	ImGuiTextFilter m_filter;
};

